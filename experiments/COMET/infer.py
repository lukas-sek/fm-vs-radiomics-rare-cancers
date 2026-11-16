"""Infer from a COMET ``*_task.pt`` export and write nnFoundation-style metrics JSON.

Loads the pickled ``TaskModule`` from ``finetune.py`` (``save_task_native``),
scores the matching WORC test fold on cached unaugmented tokens, writes a
predictions ``.pt`` (``preds`` / ``labels`` / ``probs`` / ``case_ids``), then
computes Acc/BCA/F1/… (+ bootstrap std) like ``nnFoundation/finetuning/pt_metrics.py``.

Example:
    python experiments/COMET/infer.py \\
      --task-pt Results/COMET/Finetune/worc_Liver_tokens_split00_task.pt \\
      --dataset Liver --split 0 --no-cudnn
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MMM_LICENSE_ACCEPTED", "i accept")
os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")

import numpy as np
import torch
from mmm.mtl_modules.shared_blocks.Grouper import Grouper

COMET_DIR = Path(__file__).resolve().parent
REPO_ROOT = COMET_DIR.parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
NNF_FT_DIR = REPO_ROOT / "experiments" / "nnFoundation" / "finetuning"
for path in (COMET_DIR, SCRIPTS_DIR, NNF_FT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import finetune as ft  # noqa: E402
from build_nnFoundation_lesion_patches import DATASETS  # noqa: E402
from pt_metrics import _PARAM_KEYS, compute_metrics  # noqa: E402

DEFAULT_OUTPUT_DIR = REPO_ROOT / "Results" / "COMET" / "Finetune"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--task-pt",
        type=Path,
        required=True,
        help="Native task export from finetune.py (…_task.pt).",
    )
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument(
        "--split",
        type=int,
        default=None,
        help="WORC fold index. Default: parse from …_splitNN_task.pt filename.",
    )
    parser.add_argument("--features-csv", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: next to --task-pt.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-cudnn", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--predictions-pt",
        type=Path,
        default=None,
        help="Optional path for predictions.pt (default: infer_<dataset>_splitNN.pt).",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Optional metrics JSON path (default: <predictions stem>_metrics.json).",
    )
    return parser.parse_args()


def parse_split_from_task_pt(path: Path) -> int | None:
    match = re.search(r"_split(\d+)_task\.pt$", path.name)
    return int(match.group(1)) if match else None


def count_params(module: torch.nn.Module) -> dict[str, float | int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {
        "trainable_params": int(trainable),
        "total_params": int(total),
        "trainable_fraction": float(trainable) / float(total) if total else 0.0,
    }


@torch.inference_mode()
def predict_test_fold(
    model: torch.nn.Module,
    cases: list[list[dict]],
    cache_dir: Path,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor | list[str]]:
    """Run TaskModule on unaugmented bags; return lesion-level preds for pt_metrics."""
    model = model.to(device).eval()
    task_name = model.task.get_name()
    grouper_key = model.task.args.grouper_key
    grouper: Grouper | None = model.shared_modules.get("grouper")  # type: ignore[assignment]

    dataset = ft.TokenLesionDataset(cases, cache_dir, random_aug=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ft.token_collate,
        num_workers=0,
    )

    case_ids: list[str] = []
    patient_ids: list[str] = []
    labels: list[int] = []
    scores: list[float] = []
    probs_rows: list[np.ndarray] = []

    for batch in loader:
        images = batch["image"].to(device)
        y = batch["class"].to(device)
        metas = batch.get("meta", [{} for _ in range(images.shape[0])])
        group_ids = [meta.get("group_id") for meta in metas]
        contexts = [meta.get("context") for meta in metas]
        supercase_indices = Grouper.extract_ids_from_batch(
            group_ids, for_task_name=task_name
        ).to(device)

        if grouper is not None and grouper_key.grouper_key:
            grouped_y = grouper.group_targets(y, supercase_indices, grouper_key)
        else:
            grouped_y = y

        logits = model.forward((images, supercase_indices, contexts))
        prob = torch.softmax(logits.float(), dim=1).cpu().numpy()
        if prob.shape[0] == y.shape[0]:
            targets = y.detach().cpu().numpy().astype(int)
        else:
            targets = grouped_y.detach().cpu().numpy().astype(int)

        for row in ft.lesion_rows_from_batch(metas, supercase_indices, prob, targets):
            case_ids.append(row["case_id"])
            patient_ids.append(row["patient_id"])
            labels.append(int(row["label"]))
            scores.append(float(row["score"]))
            # Rebuild a 2-class prob row from the lesion score for pt_metrics.
            probs_rows.append(np.asarray([1.0 - row["score"], row["score"]], dtype=np.float32))

    # Patient-level aggregation (mean lesion score), matching finetune reporting.
    frame_labels = np.asarray(labels, dtype=np.int64)
    frame_scores = np.asarray(scores, dtype=np.float64)
    frame_probs = np.stack(probs_rows, axis=0)
    unique_patients: list[str] = []
    patient_label: list[int] = []
    patient_score: list[float] = []
    patient_prob: list[np.ndarray] = []
    patient_case: list[str] = []
    for pid in dict.fromkeys(patient_ids):
        idxs = [i for i, p in enumerate(patient_ids) if p == pid]
        unique_patients.append(pid)
        patient_label.append(int(frame_labels[idxs[0]]))
        patient_score.append(float(frame_scores[idxs].mean()))
        patient_prob.append(frame_probs[idxs].mean(axis=0))
        patient_case.append(",".join(case_ids[i] for i in idxs))

    probs = np.stack(patient_prob, axis=0)
    preds = (np.asarray(patient_score) >= 0.5).astype(np.int64)
    return {
        "preds": torch.as_tensor(preds, dtype=torch.long),
        "labels": torch.as_tensor(patient_label, dtype=torch.long),
        "probs": torch.as_tensor(probs, dtype=torch.float32),
        "case_ids": patient_case,
        "patient_ids": unique_patients,
    }


def write_metrics_json(predictions_pt: Path, metrics_out: Path, n_bootstrap: int, seed: int) -> dict:
    metrics = compute_metrics(predictions_pt, n_bootstrap=n_bootstrap, seed=seed)
    serializable: dict = {}
    for key, value in metrics.items():
        if isinstance(value, float) and value != value:
            serializable[key] = None
        else:
            serializable[key] = value
    serializable["n_bootstrap"] = n_bootstrap
    serializable["bootstrap_seed"] = seed
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(serializable, indent=2) + "\n")
    return serializable


def main() -> None:
    args = parse_args()
    task_pt = ft.resolve_cli_path(args.task_pt)
    if not task_pt.is_file():
        raise SystemExit(f"Task export not found: {task_pt}")

    split_index = args.split if args.split is not None else parse_split_from_task_pt(task_pt)
    if split_index is None:
        raise SystemExit("Could not infer --split from filename; pass --split explicitly")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    device = ft.resolve_cuda_device(args.device)
    ft.prepare_cuda(device, no_cudnn=args.no_cudnn)

    cache_dir = (
        ft.resolve_cli_path(args.cache_dir)
        if args.cache_dir
        else ft.DEFAULT_CACHE_ROOT / f"{args.dataset}_tokens"
    )
    features_csv = (
        ft.resolve_cli_path(args.features_csv)
        if args.features_csv
        else ft.default_features_csv(args.dataset)
    )
    splits_path = (
        ft.resolve_cli_path(args.splits)
        if args.splits
        else REPO_ROOT / "crossvalidationsplits" / f"crossvalidationsplits_{args.dataset}.csv"
    )
    output_dir = (
        ft.resolve_cli_path(args.output_dir) if args.output_dir else task_pt.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    folds = ft.load_worc_splits(splits_path)
    if split_index < 0 or split_index >= len(folds):
        raise SystemExit(f"--split {split_index} out of range (0..{len(folds) - 1})")
    _, test_patients = folds[split_index]

    index = ft.load_token_index(features_csv, cache_dir, patients=None)
    test_cases = ft.cases_from_index(index, test_patients, originals_only=True)
    if not test_cases:
        raise SystemExit(f"No unaugmented test bags for split {split_index:02d}")

    print(f"Loading {task_pt}")
    model = torch.load(task_pt, map_location=device, weights_only=False)
    model.eval()
    param_stats = count_params(model)

    predictions = predict_test_fold(model, test_cases, cache_dir, args.batch_size, device)
    predictions.update(param_stats)

    pred_path = (
        ft.resolve_cli_path(args.predictions_pt)
        if args.predictions_pt
        else output_dir / f"infer_{args.dataset}_split{split_index:02d}.pt"
    )
    torch.save(predictions, pred_path)
    print(f"Wrote {pred_path} ({len(predictions['case_ids'])} patients)")

    metrics_out = (
        ft.resolve_cli_path(args.metrics_out)
        if args.metrics_out
        else pred_path.with_name(pred_path.stem + "_metrics.json")
    )
    metrics = write_metrics_json(pred_path, metrics_out, args.n_bootstrap, args.seed)

    print("Patient-level metrics (point estimate ± bootstrap std):")
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
    print(
        f"  trainable_params: {param_stats['trainable_params']:,} / "
        f"{param_stats['total_params']:,}"
    )
    print(f"Wrote {metrics_out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
