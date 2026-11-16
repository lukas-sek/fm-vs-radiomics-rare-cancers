"""Fine-tune UMedPT grouper + head on cached frozen slice tokens.

Loads 512-d tokens from ``feature_extraction.py`` (encoder/squeezer already
applied). Train samples one cached volume augmentation per lesion each access.
Inner val is a patient holdout from WORC train (unaugmented bags); WORC test is
scored once after the last loop, never used for checkpoint selection.
Schedule follows CoM³eT partial FT: 15 loops × 100 steps, 8 lesions/step,
optional 50–100 token subsample on train.

Example:
    python experiments/COMET/finetune.py --dataset Liver --max-splits 1
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MMM_LICENSE_ACCEPTED", "i accept")
os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")
os.environ.setdefault("WANDB_MODE", "offline")

import json

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from mmm.api.M3Model import DEFAULT_MODEL, M3_MODELS
from mmm.data_loading.MTLDataset import flatten_list, mtl_collate
from mmm.interactive import api, blocks, configs as cfs, data, tasks, training
from mmm.mmm_types.GroupUsage import GroupingStrategy, GroupUsage, MaskingStrategy
from mmm.mtl_modules.shared_blocks.Grouper import Grouper, MHAReducer

COMET_DIR = Path(__file__).resolve().parent
REPO_ROOT = COMET_DIR.parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (COMET_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_nnFoundation_lesion_patches import DATASETS  # noqa: E402

CLASS_NAMES = ["class0", "class1"]
DEFAULT_ARCHIVE = REPO_ROOT / "data" / "comet" / DEFAULT_MODEL
DEFAULT_CACHE_ROOT = REPO_ROOT / "data" / "comet" / "cache"
DEFAULT_FEATURES_DIR = REPO_ROOT / "Results" / "COMET" / "Features"
SUMMARY_METRICS = (
    ("AUROC", "auroc"),
    ("AP", "ap"),
    ("Accuracy", "accuracy"),
    ("BCA", "bca"),
    ("F1", "f1"),
    ("Precision", "precision"),
    ("Sensitivity", "sensitivity"),
    ("Specificity", "specificity"),
    ("NPV", "npv"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--features-csv", type=Path, default=None)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory of token .pt files. Default: data/comet/cache/<dataset>_tokens.",
    )
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "Results" / "COMET" / "Finetune")
    parser.add_argument("--max-splits", type=int, default=None)
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=15,
        help="CoM³eT loops. Default 15.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=100,
        help="Update steps per loop. Default 100 (CoM³eT). -1 = one full pass.",
    )
    parser.add_argument("--val-steps", type=int, default=-1, help="-1 = full inner val each loop.")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of WORC train patients held out as inner val (not the test fold).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Lesion bags per update step. Default 8 (CoM³eT multidimensional).",
    )
    parser.add_argument(
        "--token-subsample-min",
        type=int,
        default=50,
        help="If a train bag is larger than --token-subsample-max, sample this many tokens at least.",
    )
    parser.add_argument(
        "--token-subsample-max",
        type=int,
        default=100,
        help="If a train bag is larger, subsample to a random count in [min, max]. 0 disables.",
    )
    parser.add_argument(
        "--grouper-heads",
        type=int,
        default=8,
        help="Must divide the 512-d embedding; mmm's default 12 does not.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-cudnn",
        action="store_true",
        help="Disable cuDNN (workaround for CUDNN_STATUS_NOT_INITIALIZED / driver mismatches).",
    )
    parser.add_argument("--patients", nargs="+", default=None, help="Optional subset for smoke tests.")
    return parser.parse_args()


def resolve_cli_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def resolve_cuda_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def prepare_cuda(device: torch.device, no_cudnn: bool) -> None:
    """Init CUDA context before loading Swin; optional cuDNN disable for bad drivers."""
    if device.type != "cuda":
        return
    torch.cuda.init()
    torch.cuda.set_device(device)
    torch.empty(1, device=device).fill_(0)
    if no_cudnn:
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled (--no-cudnn)")
    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"device={torch.cuda.get_device_name(device)}"
    )


def normalize_pid(value: object) -> str:
    patient_id = str(value).strip()
    return patient_id[:-2] if patient_id.endswith(".0") else patient_id


def default_features_csv(dataset: str) -> Path:
    augmented = DEFAULT_FEATURES_DIR / f"{dataset}_augmented_features_comet.csv"
    original = DEFAULT_FEATURES_DIR / f"{dataset}_features_comet.csv"
    if augmented.is_file():
        return augmented
    return original


def load_worc_splits(path: Path) -> list[tuple[set[str], set[str]]]:
    if not path.is_file():
        raise SystemExit(f"Split file not found: {path}")
    frame = pd.read_csv(path)
    fold_ids = sorted(
        {
            int(match.group(1))
            for column in frame.columns
            if (match := re.fullmatch(r"(\d+)_train", str(column)))
        }
    )
    if not fold_ids:
        raise SystemExit(f"No '{{i}}_train' columns found in {path}")
    folds: list[tuple[set[str], set[str]]] = []
    for fold_id in fold_ids:
        test_column = f"{fold_id}_test"
        if test_column not in frame.columns:
            raise SystemExit(f"Missing {test_column} in {path}")
        train_patients = {
            normalize_pid(value)
            for value in frame[f"{fold_id}_train"].dropna()
            if str(value).strip()
        }
        test_patients = {
            normalize_pid(value)
            for value in frame[test_column].dropna()
            if str(value).strip()
        }
        folds.append((train_patients, test_patients))
    return folds


def load_token_index(csv_path: Path, cache_dir: Path, patients: list[str] | None) -> pd.DataFrame:
    if not csv_path.is_file():
        raise SystemExit(f"Feature CSV not found: {csv_path}")
    if not cache_dir.is_dir():
        raise SystemExit(f"Token cache directory not found: {cache_dir}")
    frame = pd.read_csv(csv_path)
    required = {"case_id", "patient_id", "label", "augmentation_id", "file"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"{csv_path} is missing columns: {sorted(missing)}")
    frame["patient_id"] = frame["patient_id"].map(normalize_pid)
    frame["augmentation_id"] = frame["augmentation_id"].astype(int)
    if patients is not None:
        wanted = {normalize_pid(pid) for pid in patients}
        frame = frame[frame["patient_id"].isin(wanted)]
    if frame.empty:
        raise SystemExit(f"No token rows left in {csv_path} after filtering")
    return frame.reset_index(drop=True)


def cases_from_index(
    frame: pd.DataFrame,
    patients: set[str],
    originals_only: bool,
) -> list[list[dict]]:
    """Group cached bags by lesion.

    Train keeps all K+1 cached variants per lesion so ``__getitem__`` can
    sample a different aug each epoch. Test/val keep only ``augmentation_id == 0``.
    """
    subset = frame[frame["patient_id"].isin(patients)]
    if originals_only:
        subset = subset[subset["augmentation_id"] == 0]
    cases: list[list[dict]] = []
    for _, group in subset.groupby("case_id", sort=False):
        rows = group.to_dict(orient="records")
        if rows:
            cases.append(rows)
    return cases


def patient_labels(index: pd.DataFrame, patients: set[str]) -> pd.Series:
    subset = index[(index["patient_id"].isin(patients)) & (index["augmentation_id"] == 0)]
    if subset.empty:
        subset = index[index["patient_id"].isin(patients)]
    return subset.groupby("patient_id", sort=False)["label"].first()


def holdout_from_train(
    index: pd.DataFrame,
    train_patients: set[str],
    fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """Patient-level stratified holdout from WORC train. Test fold is untouched."""
    if not 0 < fraction < 1:
        raise SystemExit("--val-fraction must be in (0, 1)")
    labels = patient_labels(index, train_patients)
    patients = np.array(labels.index, dtype=object)
    y = labels.to_numpy(dtype=int)
    if len(patients) < 4:
        raise ValueError(f"Need at least 4 train patients for inner val, got {len(patients)}")
    n_val = max(2, int(round(len(patients) * fraction)))
    n_val = min(n_val, len(patients) - 2)
    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=n_val, random_state=seed)
        fit_idx, val_idx = next(splitter.split(patients, y))
    except ValueError:
        rng = np.random.RandomState(seed)
        val_idx = rng.choice(len(patients), size=n_val, replace=False)
        fit_idx = np.setdiff1d(np.arange(len(patients)), val_idx)
    fit_patients = set(patients[fit_idx].tolist())
    val_patients = set(patients[val_idx].tolist())
    return fit_patients, val_patients


def subsample_token_indices(n_tokens: int, subsample_min: int, subsample_max: int) -> list[int]:
    """CoM³eT: if more than ``subsample_max`` tokens, keep a random 50–100."""
    if subsample_max <= 0 or n_tokens <= subsample_max:
        return list(range(n_tokens))
    keep = random.randint(min(subsample_min, subsample_max), subsample_max)
    keep = min(keep, n_tokens)
    return sorted(random.sample(range(n_tokens), keep))


def load_token_bag(
    row: dict,
    cache_dir: Path,
    subsample_min: int = 0,
    subsample_max: int = 0,
) -> list[dict]:
    token_path = cache_dir / str(row["file"])
    if not token_path.is_file():
        raise FileNotFoundError(f"Missing token file: {token_path}")
    entry = torch.load(token_path, map_location="cpu", weights_only=True)
    tokens = torch.as_tensor(entry["tokens"]).float()
    # Keep bags as (S, D) so ClassificationTask.batch_is_compressed skips the encoder.
    if tokens.ndim > 2:
        tokens = tokens.flatten(1)
    if tokens.ndim != 2:
        raise ValueError(f"{token_path}: expected tokens (S, D), got {tuple(tokens.shape)}")
    indices = subsample_token_indices(tokens.shape[0], subsample_min, subsample_max)
    case_id = str(row["case_id"])
    group_id = f"{case_id}__aug{int(row['augmentation_id'])}"
    class_idx = int(row["label"])
    return [
        {
            "image": tokens[i].reshape(-1),
            "class": class_idx,
            "meta": {
                "group_id": group_id,
                "caption": f"{group_id} ({CLASS_NAMES[class_idx]})",
                "patient_id": str(row["patient_id"]),
                "case_id": case_id,
                "context": (i,),
            },
        }
        for i in indices
    ]


class TokenLesionDataset(torch.utils.data.Dataset):
    """One item = one lesion; train randomly picks a cached aug each access."""

    def __init__(
        self,
        cases: list[list[dict]],
        cache_dir: Path,
        random_aug: bool,
        subsample_min: int = 0,
        subsample_max: int = 0,
    ):
        self.cases = cases
        self.cache_dir = cache_dir
        self.random_aug = random_aug
        self.subsample_min = subsample_min
        self.subsample_max = subsample_max

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> list[dict]:
        variants = self.cases[index]
        if self.random_aug and len(variants) > 1:
            row = random.choice(variants)
        else:
            row = next(
                (variant for variant in variants if int(variant["augmentation_id"]) == 0),
                variants[0],
            )
        return load_token_bag(
            row,
            self.cache_dir,
            subsample_min=self.subsample_min,
            subsample_max=self.subsample_max,
        )


def token_collate(batch: list) -> dict:
    return mtl_collate(flatten_list(batch))


def make_cohort(
    train_cases: list[list[dict]],
    val_cases: list[list[dict]],
    cache_dir: Path,
    batch_size: int,
    subsample_min: int,
    subsample_max: int,
) -> data.TrainValCohort:
    sizes = (batch_size, batch_size)
    return data.TrainValCohort(
        data.TrainValCohort.Config(batch_size=sizes, num_workers=0),
        train_ds=data.ClassificationDataset(
            TokenLesionDataset(
                train_cases,
                cache_dir,
                random_aug=True,
                subsample_min=subsample_min,
                subsample_max=subsample_max,
            ),
            class_names=CLASS_NAMES,
            collate_fn=token_collate,
            verify_cases=0,
        ),
        val_ds=data.ClassificationDataset(
            TokenLesionDataset(val_cases, cache_dir, random_aug=False),
            class_names=CLASS_NAMES,
            collate_fn=token_collate,
            verify_cases=0,
        ),
    )


def ensure_checkpoint(path: Path) -> Path:
    path = resolve_cli_path(path)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {M3_MODELS[DEFAULT_MODEL]} -> {path}")
        torch.hub.download_url_to_file(M3_MODELS[DEFAULT_MODEL], str(path))
    return path


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator) / float(denominator)


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    predicted = (scores >= 0.5).astype(np.int64)
    if len(np.unique(labels)) < 2:
        auc = float("nan")
        ap = float("nan")
    else:
        auc = float(roc_auc_score(labels, scores))
        ap = float(average_precision_score(labels, scores))
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    sensitivity = _safe_rate(tp, tp + fn)
    specificity = _safe_rate(tn, tn + fp)
    precision = _safe_rate(tp, tp + fp)
    npv = _safe_rate(tn, tn + fn)
    return {
        "auroc": auc,
        "ap": ap,
        "accuracy": float((predicted == labels).mean()) if labels.size else float("nan"),
        "bca": float(balanced_accuracy_score(labels, predicted)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "precision": float(precision_score(labels, predicted, zero_division=0))
        if not np.isnan(precision)
        else precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "npv": npv,
        "recall": float(recall_score(labels, predicted, zero_division=0)),
    }


def metric_summary(per_split: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for display_name, column in SUMMARY_METRICS:
        if column not in per_split.columns:
            continue
        values = per_split[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            lower, upper = np.percentile(finite, [2.5, 97.5])
        else:
            lower, upper = float("nan"), float("nan")
        rows.append(
            {
                "metric": display_name,
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "median": float(np.nanmedian(values)),
                "ci_95_lower": float(lower),
                "ci_95_upper": float(upper),
                "n_splits": len(per_split),
            }
        )
    return pd.DataFrame(rows)


def patient_prediction_bundle(predictions: list[dict]) -> dict:
    """Patient-level tensors for an nnFoundation-style predictions.pt payload."""
    pred_frame = pd.DataFrame(predictions)
    patient = pred_frame.groupby("patient_id", sort=False).agg(
        label=("label", "first"),
        score=("score", "mean"),
        case_id=("case_id", lambda s: ",".join(map(str, s))),
    )
    scores = patient["score"].to_numpy(dtype=np.float64)
    probs = np.stack([1.0 - scores, scores], axis=1)
    preds = (scores >= 0.5).astype(np.int64)
    return {
        "preds": torch.as_tensor(preds, dtype=torch.long),
        "labels": torch.as_tensor(patient["label"].to_numpy(), dtype=torch.long),
        "probs": torch.as_tensor(probs, dtype=torch.float32),
        "case_ids": patient["case_id"].tolist(),
        "patient_ids": patient.index.tolist(),
        "n_patients": int(len(patient)),
    }


def patient_metrics(rows: list[dict]) -> dict[str, float]:
    frame = pd.DataFrame(rows)
    grouped = frame.groupby("patient_id", sort=False).agg(label=("label", "first"), score=("score", "mean"))
    return classification_metrics(grouped["label"].to_numpy(), grouped["score"].to_numpy())


def lesion_rows_from_batch(
    metas: list[dict],
    supercase_indices: torch.Tensor,
    probs: np.ndarray,
    targets: np.ndarray,
) -> list[dict]:
    """Collapse token- or group-level model outputs to one row per lesion bag.

    ``grouping=single`` → one logit per unique supercase.
    ``grouping=full`` (mmm default) → one logit per token; mean-pool scores per bag.
    """
    scores = probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else np.asarray(probs).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    indices = supercase_indices.detach().cpu().tolist()
    n_tokens = len(metas)
    if len(scores) not in (n_tokens, len(set(indices))):
        raise ValueError(
            f"Cannot align outputs: n_scores={len(scores)}, n_tokens={n_tokens}, "
            f"n_groups={len(set(indices))}"
        )

    # Map each unique supercase id -> token positions / representative meta.
    groups: dict[int, list[int]] = {}
    for token_i, case_i in enumerate(indices):
        groups.setdefault(int(case_i), []).append(token_i)

    rows: list[dict] = []
    unique_sorted = sorted(groups)
    for rank, case_i in enumerate(unique_sorted):
        token_is = groups[case_i]
        meta = metas[token_is[0]]
        if len(scores) == n_tokens:
            score = float(np.mean(scores[token_is]))
            label = int(targets[token_is[0]])
        else:
            score = float(scores[rank])
            label = int(targets[rank])
        rows.append(
            {
                "patient_id": str(meta.get("patient_id")),
                "case_id": str(meta.get("case_id")),
                "label": label,
                "score": score,
            }
        )
    return rows


@torch.no_grad()
def evaluate_cases(
    cases: list[list[dict]],
    cache_dir: Path,
    task: tasks.ClassificationTask,
    grouper: Grouper,
    encoder,
    squeezer,
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict], dict[str, float]]:
    """Score unaugmented bags with the last in-memory weights (no checkpoint selection)."""
    dataset = TokenLesionDataset(cases, cache_dir, random_aug=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=token_collate,
        num_workers=0,
    )
    task.eval()
    grouper.eval()
    shared = {"encoder": encoder, "squeezer": squeezer, "grouper": grouper}
    rows: list[dict] = []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["class"].to(device)
        metas = batch.get("meta", [{} for _ in range(images.shape[0])])
        group_ids = [meta.get("group_id") for meta in metas]
        contexts = [meta.get("context") for meta in metas]
        supercase_indices = Grouper.extract_ids_from_batch(
            group_ids, for_task_name=task.get_name()
        ).to(device)
        logits = task.forward((images, supercase_indices, contexts), shared)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        # Prefer raw token labels when outputs are per-token (grouping=full).
        if probs.shape[0] == labels.shape[0]:
            targets = labels.detach().cpu().numpy()
        else:
            targets = grouper.group_targets(
                labels, supercase_indices, task.args.grouper_key
            ).detach().cpu().numpy()
        rows.extend(lesion_rows_from_batch(metas, supercase_indices, probs, targets))
    return rows, patient_metrics(rows)


def run_split(
    split_index: int,
    train_cases: list[list[dict]],
    val_cases: list[list[dict]],
    test_cases: list[list[dict]],
    args: argparse.Namespace,
    cache_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> dict:
    checkpoint = ensure_checkpoint(args.checkpoint)
    foundation_model = api.M3Model(str(checkpoint), device_identifier=str(device))
    encoder: blocks.PyramidEncoder = foundation_model["encoder"]
    squeezer: blocks.Squeezer = foundation_model["squeezer"]
    encoder.freeze_all_parameters()
    squeezer.freeze_all_parameters()

    grouper = Grouper(
        Grouper.Config(reducer=MHAReducer.Config(num_heads=args.grouper_heads)),
        embedding_dim=squeezer.get_hidden_dim(),
    )

    cohort = make_cohort(
        train_cases,
        val_cases,
        cache_dir,
        args.batch_size,
        args.token_subsample_min,
        args.token_subsample_max,
    )
    task = tasks.ClassificationTask(
        hidden_dim=squeezer.get_hidden_dim(),
        args=tasks.ClassificationTask.Config(
            module_name=f"worc_{args.dataset}_split{split_index:02d}",
            grouper_key=GroupUsage(
                grouper_key="grouper",
                masking=MaskingStrategy.fullattention,
                # One pooled embedding per lesion bag (not per slice / mmm default "full").
                grouping=GroupingStrategy.single,
            ),
        ),
        cohort=cohort,
    )

    experiment_name = f"worc_{args.dataset}_tokens_split{split_index:02d}"
    trainer = training.MTLTrainer(
        training.MTLTrainer.Config(
            checkpoint_cache_folder=output_dir / "checkpoints" / experiment_name,
            train_device="cuda" if device.type == "cuda" else "cpu",
            mtl_train_loop=cfs.TrainLoopConfig(max_steps=args.train_steps),
            mtl_val_loop=cfs.ValLoopConfig(max_steps=args.val_steps),
            max_epochs=args.max_epochs,
        ),
        experiment_name=cfs.remove_wandb_special_chars(experiment_name),
        clear_checkpoints=True,
    )
    trainer.add_shared_blocks(
        [foundation_model[k] for k in foundation_model.get_sharedblock_keys()] + [grouper]
    )
    trainer.add_mtl_task(task)
    n_augs = max((len(case) for case in train_cases), default=0)
    print(
        f"Split {split_index:02d}: {len(train_cases)} train / {len(val_cases)} inner-val / "
        f"{len(test_cases)} test lesions (~{n_augs} cached augs on train); "
        f"{args.max_epochs} loops × {args.train_steps} steps, batch={args.batch_size} "
        f"-> {experiment_name}"
    )
    wandb.init(
        project="comet-worc",
        name=cfs.remove_wandb_special_chars(experiment_name),
        reinit=True,
        settings=wandb.Settings(quiet=True),
    )
    try:
        trainer.fit()
        export_path = output_dir / f"{experiment_name}_task.pt"
        trainer.save_task_native(task.get_name(), export_path, only_inference=True)
        print(f"Wrote {export_path}")
        predictions, metrics = evaluate_cases(
            test_cases,
            cache_dir,
            task,
            grouper,
            encoder,
            squeezer,
            args.batch_size,
            device,
        )
        bundle = patient_prediction_bundle(predictions)
        pred_path = output_dir / f"infer_{args.dataset}_split{split_index:02d}.pt"
        torch.save(bundle, pred_path)
        print(f"Wrote {pred_path}")
        print(
            f"Split {split_index:02d} test (last ckpt, patient-level): "
            f"AUROC={metrics['auroc']:.4f} AP={metrics['ap']:.4f} BCA={metrics['bca']:.4f}"
        )
        return {
            "metrics": {
                "split": split_index,
                "n_train_lesions": len(train_cases),
                "n_val_lesions": len(val_cases),
                "n_test_lesions": len(test_cases),
                "n_test_patients": bundle["n_patients"],
                **metrics,
            },
            "predictions": bundle,
        }
    finally:
        wandb.finish()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    device = resolve_cuda_device(args.device)
    prepare_cuda(device, no_cudnn=args.no_cudnn)

    cache_dir = resolve_cli_path(args.cache_dir) if args.cache_dir else (
        DEFAULT_CACHE_ROOT / f"{args.dataset}_tokens"
    )
    features_csv = resolve_cli_path(args.features_csv) if args.features_csv else default_features_csv(args.dataset)
    splits_path = resolve_cli_path(args.splits) if args.splits else (
        REPO_ROOT / "crossvalidationsplits" / f"crossvalidationsplits_{args.dataset}.csv"
    )
    output_dir = resolve_cli_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index = load_token_index(features_csv, cache_dir, args.patients)
    folds = load_worc_splits(splits_path)
    if args.max_splits is not None:
        if args.max_splits < 1:
            raise SystemExit("--max-splits must be at least 1")
        folds = folds[: args.max_splits]

    print(
        f"{args.dataset}: {index['case_id'].nunique()} lesions, {len(index)} bags, "
        f"{len(folds)} splits, csv={features_csv}, cache={cache_dir}, device={device}"
    )

    per_split_rows: list[dict] = []
    split_payloads: list[dict] = []
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_probs: list[torch.Tensor] = []
    all_case_ids: list[str] = []
    all_patient_ids: list[str] = []
    all_split_ids: list[int] = []

    for split_index, (train_patients, test_patients) in enumerate(folds):
        try:
            fit_patients, val_patients = holdout_from_train(
                index, train_patients, args.val_fraction, args.seed + split_index
            )
        except ValueError as error:
            warnings.warn(f"Split {split_index:02d}: {error}, skipped", stacklevel=1)
            continue
        train_cases = cases_from_index(index, fit_patients, originals_only=False)
        val_cases = cases_from_index(index, val_patients, originals_only=True)
        test_cases = cases_from_index(index, test_patients, originals_only=True)
        if not train_cases or not val_cases or not test_cases:
            warnings.warn(
                f"Split {split_index:02d} has empty train/val/test after filtering, skipped",
                stacklevel=1,
            )
            continue
        result = run_split(
            split_index,
            train_cases,
            val_cases,
            test_cases,
            args,
            cache_dir,
            output_dir,
            device,
        )
        metrics_row = result["metrics"]
        bundle = result["predictions"]
        per_split_rows.append(metrics_row)
        split_payloads.append(
            {
                "split": split_index,
                "metrics": metrics_row,
                "preds": bundle["preds"],
                "labels": bundle["labels"],
                "probs": bundle["probs"],
                "case_ids": bundle["case_ids"],
                "patient_ids": bundle["patient_ids"],
            }
        )
        n = int(bundle["preds"].shape[0])
        all_preds.append(bundle["preds"])
        all_labels.append(bundle["labels"])
        all_probs.append(bundle["probs"])
        all_case_ids.extend(bundle["case_ids"])
        all_patient_ids.extend(bundle["patient_ids"])
        all_split_ids.extend([split_index] * n)

    if not per_split_rows:
        raise SystemExit("No splits produced test metrics")

    per_split = pd.DataFrame(per_split_rows)
    summary = metric_summary(per_split)
    summary_records = summary.to_dict(orient="records")
    prefix = f"{args.dataset}_tokens"
    per_split_path = output_dir / f"{prefix}_cv_per_split.csv"
    summary_csv_path = output_dir / f"{prefix}_cv_summary.csv"
    summary_json_path = output_dir / f"{prefix}_cv_summary.json"
    all_splits_pt = output_dir / f"{prefix}_cv_all_splits.pt"

    per_split.to_csv(per_split_path, index=False)
    summary.to_csv(summary_csv_path, index=False)
    summary_json_path.write_text(json.dumps(summary_records, indent=2) + "\n")
    torch.save(
        {
            "dataset": args.dataset,
            "n_splits": len(split_payloads),
            "preds": torch.cat(all_preds, dim=0),
            "labels": torch.cat(all_labels, dim=0),
            "probs": torch.cat(all_probs, dim=0),
            "case_ids": all_case_ids,
            "patient_ids": all_patient_ids,
            "splits": torch.as_tensor(all_split_ids, dtype=torch.long),
            "per_split": split_payloads,
            "per_split_metrics": per_split.to_dict(orient="records"),
            "summary": summary_records,
        },
        all_splits_pt,
    )

    print("\n--- Summary across splits ---")
    for _, row in summary.iterrows():
        print(
            f"{row['metric']} 95%: {row['mean']:.4f} "
            f"({row['ci_95_lower']:.4f}-{row['ci_95_upper']:.4f}); "
            f"mean ± SD {row['mean']:.4f} ± {row['std']:.4f}"
        )
    print(f"Wrote {per_split_path}")
    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {summary_json_path}")
    print(f"Wrote {all_splits_pt}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
