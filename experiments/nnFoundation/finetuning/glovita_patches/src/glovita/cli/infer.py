from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import torch
from pydantic import BaseModel, Field

from glovita.configs.cli import parse_cli
from glovita.configs.data import DataConfig
from glovita.configs.dataloading import DataloadingConfig
from glovita.configs.root import resolve_encoder_input_shape
from glovita.datasets.factory import build_dataloaders
from glovita.models.factory import use_finetuned_checkpoint
from glovita.models.preprocessing import resolve_encoder_preprocessing_defaults


class InferConfig(BaseModel):
    """Checkpoint-based inference and evaluation configuration."""

    exp_dir: Path = Field(
        default=Path("./experiments"),
        description="Run directory containing checkpoints/ or fold subdirectories with checkpoints/.",
    )
    data: DataConfig = Field(
        description="Dataset configuration used to build the evaluation dataloader.",
    )
    dataloading: DataloadingConfig = Field(
        default_factory=DataloadingConfig,
        description="Dataloader settings for inference.",
    )
    metrics: List[str] = Field(
        default_factory=lambda: ["acc", "f1"],
        description="Metric names to compute on the test set. Common values: acc, f1, mse, mae.",
    )
    fold: Optional[str] = Field(
        default=None,
        description="Specific fold to evaluate. If unset, infer scans all fold subdirectories and ensembles their checkpoints when available.",
    )
    checkpoint: Literal["last", "best"] = Field(
        default="last",
        description="Which checkpoint to evaluate: last.pt or best.pt.",
    )
    pred_output: Optional[Path] = Field(
        default=None,
        description="Optional path to save predictions, labels, probabilities, and case ids (if the dataset provides them) as a torch file.",
    )


def _collect_checkpoints(exp_dir: Path, fold: Optional[str], checkpoint: str = "last") -> List[Path]:
    filename = f"{checkpoint}.pt"
    if fold is not None:
        candidates = list((exp_dir / fold / "checkpoints").glob(filename))
    else:
        candidates = list(exp_dir.glob(f"*/checkpoints/{filename}"))
        if not candidates:
            candidates = list((exp_dir / "checkpoints").glob(filename))
    if not candidates:
        raise FileNotFoundError(f"No '{filename}' checkpoints found under {exp_dir}")
    return sorted(candidates)


def _load_run_config(ckpt_path: Path):
    run_dir = ckpt_path.parent.parent
    config_file = run_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found at {config_file}")
    from glovita.configs.root import RootConfig

    return RootConfig.model_validate_json(config_file.read_text())


def _load_trainable_param_stats(ckpt_path: Path, state: dict) -> Dict[str, Any]:
    """Prefer training-time counts stored in the checkpoint / sidecar (not a live recount)."""
    stats: Dict[str, Any] = {}
    for key in ("trainable_params", "total_params", "trainable_fraction"):
        if key in state and state[key] is not None:
            stats[key] = state[key]

    sidecar = ckpt_path.parent / "best_trainable_params.json"
    if ckpt_path.name == "best.pt" and sidecar.exists():
        try:
            side = json.loads(sidecar.read_text())
        except json.JSONDecodeError:
            side = {}
        for key in ("trainable_params", "total_params", "trainable_fraction", "epoch"):
            if key in side and key not in stats:
                stats[key] = side[key]
        if "epoch" in side:
            stats.setdefault("best_epoch", side["epoch"])

    if "epoch" in state and "best_epoch" not in stats:
        stats["best_epoch"] = state["epoch"]

    # Normalize types for JSON.
    if "trainable_params" in stats:
        stats["trainable_params"] = int(stats["trainable_params"])
    if "total_params" in stats:
        stats["total_params"] = int(stats["total_params"])
    if "trainable_fraction" in stats:
        stats["trainable_fraction"] = float(stats["trainable_fraction"])
    elif "trainable_params" in stats and "total_params" in stats and stats["total_params"]:
        stats["trainable_fraction"] = float(stats["trainable_params"]) / float(stats["total_params"])

    return stats


def _load_model(ckpt_path: Path) -> torch.nn.Module:
    from glovita.models.factory import build_model
    from glovita.models.peft.registry import apply_peft

    run_config = _load_run_config(ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu", mmap=True)
    output_dim = getattr(run_config.data, "num_classes", 1)
    model = build_model(use_finetuned_checkpoint(run_config.model, ckpt_path, state), output_dim=output_dim)
    model = apply_peft(model, run_config.peft)

    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def run_inference(config: InferConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_paths = _collect_checkpoints(config.exp_dir, config.fold, config.checkpoint)
    print(f"Found {len(ckpt_paths)} checkpoint(s).")

    reference_state = torch.load(ckpt_paths[0], map_location="cpu", mmap=True)
    trainable_stats = _load_trainable_param_stats(ckpt_paths[0], reference_state)
    if trainable_stats:
        tp = trainable_stats.get("trainable_params")
        tot = trainable_stats.get("total_params")
        if tp is not None and tot is not None:
            print(f"Trainable params (from checkpoint): {tp:,} / {tot:,}")

    reference_run_config = _load_run_config(ckpt_paths[0])
    reference_model_config = use_finetuned_checkpoint(
        reference_run_config.model, ckpt_paths[0], reference_state
    )
    resolve_encoder_input_shape(config.data, reference_model_config)
    encoder_preprocessing = resolve_encoder_preprocessing_defaults(reference_model_config.encoder).as_kwargs()
    _, _, test_loader = build_dataloaders(
        config.data,
        config.dataloading,
        encoder_preprocessing=encoder_preprocessing,
    )
    if test_loader is None:
        raise ValueError(f"Dataset {config.data.dataset!r} has no test split to evaluate.")

    task = reference_run_config.data.task
    subtask = reference_run_config.data.subtask
    num_classes = reference_run_config.data.num_classes

    all_logits: List[torch.Tensor] = []
    all_labels: Optional[torch.Tensor] = None

    for ckpt_path in ckpt_paths:
        print(f"  Loading {ckpt_path}")
        model = _load_model(ckpt_path)
        model.to(device)

        batch_logits, batch_labels = [], []
        use_amp = device.type == "cuda"
        for x, y in test_loader:
            x = x.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            batch_logits.append(logits.float().cpu())
            batch_labels.append(y)

        all_logits.append(torch.cat(batch_logits))
        if all_labels is None:
            all_labels = torch.cat(batch_labels)

    assert all_labels is not None

    summed = torch.sum(torch.stack(all_logits), dim=0)

    if task == "Regression":
        preds = summed.squeeze(-1)
        probs = None
    elif subtask == "multilabel":
        probs = torch.stack([logits.sigmoid() for logits in all_logits]).mean(dim=0)
        preds = (probs > 0.5).long()
    else:
        probs = torch.stack([logits.softmax(dim=1) for logits in all_logits]).mean(dim=0)
        preds = torch.argmax(probs, dim=1)

    from torchmetrics import Accuracy, F1Score, MeanAbsoluteError, MeanSquaredError, MetricCollection

    metrics_dict = {}
    if task == "Regression":
        if "mse" in config.metrics:
            metrics_dict["MSE"] = MeanSquaredError()
        if "mae" in config.metrics:
            metrics_dict["MAE"] = MeanAbsoluteError()
    else:
        metric_task = subtask
        if "acc" in config.metrics:
            metrics_dict["Accuracy"] = Accuracy(
                task=metric_task, num_classes=num_classes, num_labels=num_classes
            )
        if "f1" in config.metrics:
            metrics_dict["F1"] = F1Score(
                task=metric_task,
                num_classes=num_classes,
                num_labels=num_classes,
                average="macro",
            )

    collection = MetricCollection(metrics_dict)
    results = collection(preds, all_labels)

    print("\nTest results:")
    for k, v in results.items():
        print(f"  {k}: {v.item():.4f}")

    results_payload: Dict[str, Any] = {
        k: float(v.item()) for k, v in results.items()
    }
    results_payload.update(trainable_stats)
    results_payload["checkpoint"] = config.checkpoint
    results_payload["checkpoint_paths"] = [str(p) for p in ckpt_paths]

    if config.pred_output is not None:
        config.pred_output.parent.mkdir(parents=True, exist_ok=True)
        outputs: Dict[str, Any] = {"preds": preds, "labels": all_labels}
        if probs is not None:
            outputs["probs"] = probs
        case_ids = getattr(test_loader.dataset, "case_ids", None)
        if case_ids is not None:
            outputs["case_ids"] = list(case_ids)
        outputs.update(trainable_stats)
        torch.save(outputs, config.pred_output)
        print(f"Predictions saved to {config.pred_output}")

        results_json = config.pred_output.with_name(
            config.pred_output.stem + "_results.json"
        )
        results_json.write_text(json.dumps(results_payload, indent=2) + "\n")
        print(f"Results JSON saved to {results_json}")


def main(config: InferConfig | None = None) -> None:
    if config is None:
        config = parse_cli(InferConfig)
    run_inference(config)


if __name__ == "__main__":
    main()
