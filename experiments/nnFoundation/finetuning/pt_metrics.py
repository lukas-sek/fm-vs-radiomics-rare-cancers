"""Compute and save classification metrics from a glovita_infer predictions.pt.

Point estimates use the full test set. ``*_std`` values are bootstrap SDs
(resample cases with replacement).

Also embeds ``trainable_params`` / ``total_params`` when available from the
predictions.pt, a companion ``*_results.json``, or the training checkpoint /
``best_trainable_params.json`` (via ``--exp-dir``).

Example:

    python experiments/nnFoundation/finetuning/pt_metrics.py \\
      Results/nnFoundation/finetune/infer_Liver_test.pt \\
      --exp-dir Results/nnFoundation/finetune_vit_lora/... \\
      --fold 1 --checkpoint best
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

_PARAM_KEYS = ("trainable_params", "total_params", "trainable_fraction", "best_epoch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Acc/BCA/F1/Precision/Recall/Spec/NPV/AUROC (+ bootstrap std) from predictions.pt"
    )
    parser.add_argument("pt_file", type=Path, help="Path to glovita_infer predictions.pt")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON output path (default: <pt_file stem>_metrics.json)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Bootstrap resamples for metric std (0 disables). Default: 1000",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for bootstrap")
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=None,
        help="Training run dir (used to load trainable param counts from checkpoints).",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default=None,
        help="Fold subdirectory under exp-dir (e.g. 1). Optional.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        choices=("best", "last"),
        help="Which checkpoint sidecar/state to read for trainable params. Default: best",
    )
    parser.add_argument(
        "--param-source",
        type=Path,
        default=None,
        help="Explicit best_trainable_params.json or checkpoint .pt to read counts from.",
    )
    return parser.parse_args()


def _safe_rate(num: float, den: float) -> float:
    if den <= 0:
        return float("nan")
    return float(num) / float(den)


def _metrics_from_arrays(
    labels: np.ndarray,
    preds: np.ndarray,
    scores: np.ndarray,
    probs: np.ndarray,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "Accuracy": float(accuracy_score(labels, preds)),
        "Balanced_Accuracy": float(balanced_accuracy_score(labels, preds)),
        "F1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "Precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "Recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
    }

    if len(np.unique(labels)) <= 2 and set(np.unique(labels)).issubset({0, 1}):
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        metrics["Sensitivity"] = _safe_rate(tp, tp + fn)
        metrics["Specificity"] = _safe_rate(tn, tn + fp)
        metrics["NPV"] = _safe_rate(tn, tn + fn)
        try:
            metrics["AUROC"] = float(roc_auc_score(labels, scores))
        except ValueError:
            metrics["AUROC"] = float("nan")
    else:
        try:
            metrics["AUROC"] = float(
                roc_auc_score(labels, probs, multi_class="ovr", average="macro")
            )
        except ValueError:
            metrics["AUROC"] = float("nan")

    return metrics


def _normalize_param_stats(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "trainable_params" in raw and raw["trainable_params"] is not None:
        out["trainable_params"] = int(raw["trainable_params"])
    if "total_params" in raw and raw["total_params"] is not None:
        out["total_params"] = int(raw["total_params"])
    if "trainable_fraction" in raw and raw["trainable_fraction"] is not None:
        out["trainable_fraction"] = float(raw["trainable_fraction"])
    elif "trainable_params" in out and "total_params" in out and out["total_params"]:
        out["trainable_fraction"] = float(out["trainable_params"]) / float(out["total_params"])
    if "best_epoch" in raw and raw["best_epoch"] is not None:
        out["best_epoch"] = int(raw["best_epoch"])
    elif "epoch" in raw and raw["epoch"] is not None:
        out["best_epoch"] = int(raw["epoch"])
    return out


def _load_param_stats_from_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if path.suffix == ".json":
        try:
            return _normalize_param_stats(json.loads(path.read_text()))
        except json.JSONDecodeError:
            return {}
    if path.suffix == ".pt":
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        if not isinstance(state, dict):
            return {}
        return _normalize_param_stats(state)
    return {}


def _resolve_checkpoint_dir(exp_dir: Path, fold: str | None) -> Path | None:
    candidates: list[Path] = []
    if fold is not None:
        candidates.append(exp_dir / str(fold) / "checkpoints")
    candidates.append(exp_dir / "checkpoints")
    # Nested run dirs: exp_dir/<something>/checkpoints or exp_dir/<fold>/checkpoints
    if fold is not None:
        candidates.extend(sorted(exp_dir.glob(f"*/{fold}/checkpoints")))
    candidates.extend(sorted(exp_dir.glob("*/checkpoints")))
    for path in candidates:
        if path.is_dir():
            return path
    return None


def load_trainable_param_stats(
    *,
    pt_data: dict[str, Any],
    pt_path: Path,
    exp_dir: Path | None = None,
    fold: str | None = None,
    checkpoint: str = "best",
    param_source: Path | None = None,
) -> dict[str, Any]:
    """Resolve trainable counts: .pt → results json → explicit source → exp-dir ckpt."""
    stats = _normalize_param_stats(pt_data)
    if "trainable_params" in stats:
        return stats

    for companion in (
        pt_path.with_name(pt_path.stem + "_results.json"),
        pt_path.with_suffix(".json"),
    ):
        stats = _load_param_stats_from_file(companion)
        if "trainable_params" in stats:
            return stats

    if param_source is not None:
        stats = _load_param_stats_from_file(param_source)
        if "trainable_params" in stats:
            return stats

    if exp_dir is not None:
        ckpt_dir = _resolve_checkpoint_dir(exp_dir, fold)
        if ckpt_dir is not None:
            if checkpoint == "best":
                stats = _load_param_stats_from_file(ckpt_dir / "best_trainable_params.json")
                if "trainable_params" in stats:
                    return stats
            stats = _load_param_stats_from_file(ckpt_dir / f"{checkpoint}.pt")
            if "trainable_params" in stats:
                return stats

    return {}


def compute_metrics(
    pt_path: Path,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
    exp_dir: Path | None = None,
    fold: str | None = None,
    checkpoint: str = "best",
    param_source: Path | None = None,
) -> dict[str, Any]:
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    for key in ("preds", "labels", "probs"):
        if key not in data:
            raise SystemExit(f"{pt_path} missing key {key!r}; found {list(data)}")

    preds = np.asarray(data["preds"]).reshape(-1).astype(int)
    labels = np.asarray(data["labels"]).reshape(-1).astype(int)
    probs = np.asarray(data["probs"])
    if probs.ndim == 1:
        scores = probs
        probs_2d = probs.reshape(-1, 1)
    else:
        scores = probs[:, 1] if probs.shape[1] == 2 else probs
        probs_2d = probs

    metrics: dict[str, Any] = _metrics_from_arrays(labels, preds, scores, probs_2d)
    metric_keys = list(metrics.keys())

    if n_bootstrap > 0 and labels.size > 1:
        rng = np.random.default_rng(seed)
        samples = {k: [] for k in metric_keys}
        n = labels.size
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot = _metrics_from_arrays(
                labels[idx],
                preds[idx],
                scores[idx] if np.ndim(scores) == 1 else scores[idx],
                probs_2d[idx],
            )
            for k in metric_keys:
                v = boot[k]
                if v == v:  # skip nan
                    samples[k].append(v)
        for k in metric_keys:
            vals = samples[k]
            metrics[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) >= 2 else float("nan")

    param_stats = load_trainable_param_stats(
        pt_data=data if isinstance(data, dict) else {},
        pt_path=pt_path,
        exp_dir=exp_dir,
        fold=fold,
        checkpoint=checkpoint,
        param_source=param_source,
    )
    metrics.update(param_stats)
    return metrics


def main() -> None:
    args = parse_args()
    metrics = compute_metrics(
        args.pt_file,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        exp_dir=args.exp_dir,
        fold=args.fold,
        checkpoint=args.checkpoint,
        param_source=args.param_source,
    )
    out = args.out or args.pt_file.with_name(args.pt_file.stem + "_metrics.json")

    print("Full metrics (point estimate ± bootstrap std):")
    base_keys = [k for k in metrics if not k.endswith("_std") and k not in _PARAM_KEYS]
    for key in base_keys:
        value = metrics[key]
        std = metrics.get(f"{key}_std")
        if isinstance(value, float) and value != value:
            print(f"  {key}: nan")
        elif std is None or (isinstance(std, float) and std != std):
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.4f} ± {std:.4f}")

    serializable = {}
    for k, v in metrics.items():
        if isinstance(v, float) and v != v:
            serializable[k] = None
        elif isinstance(v, (int, float, str, bool)) or v is None:
            serializable[k] = v
        else:
            serializable[k] = v
    serializable["n_bootstrap"] = args.n_bootstrap
    serializable["bootstrap_seed"] = args.seed
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(serializable, indent=2) + "\n")
    if "trainable_params" in serializable and "total_params" in serializable:
        print(
            f"  trainable_params: {serializable['trainable_params']:,} / "
            f"{serializable['total_params']:,}"
        )
    else:
        print(
            "  trainable_params: missing "
            "(pass --exp-dir / --param-source, or re-infer with patched glovita_infer)"
        )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
