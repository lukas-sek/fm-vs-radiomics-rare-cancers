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

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from mmm.api.M3Model import DEFAULT_MODEL, M3_MODELS
from mmm.data_loading.MTLDataset import flatten_list, mtl_collate
from mmm.interactive import api, blocks, configs as cfs, data, tasks, training
from mmm.mmm_types.GroupUsage import GroupUsage, MaskingStrategy
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
    return {
        "auroc": auc,
        "ap": ap,
        "bca": float(balanced_accuracy_score(labels, predicted)),
        "accuracy": float((predicted == labels).mean()) if labels.size else float("nan"),
    }


def patient_metrics(rows: list[dict]) -> dict[str, float]:
    frame = pd.DataFrame(rows)
    grouped = frame.groupby("patient_id", sort=False).agg(label=("label", "first"), score=("score", "mean"))
    return classification_metrics(grouped["label"].to_numpy(), grouped["score"].to_numpy())


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
        grouped_labels = grouper.group_targets(labels, supercase_indices, task.args.grouper_key)
        logits = task.forward((images, supercase_indices, contexts), shared)
        probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
        targets = grouped_labels.detach().cpu().numpy()
        # One row per grouped lesion (first token of each group_id in batch order).
        seen: dict[str, int] = {}
        order: list[str] = []
        for meta in metas:
            group_id = str(meta.get("group_id"))
            if group_id not in seen:
                seen[group_id] = len(order)
                order.append(group_id)
        for group_id, score, label in zip(order, probs, targets, strict=True):
            meta = next(m for m in metas if str(m.get("group_id")) == group_id)
            rows.append(
                {
                    "patient_id": str(meta.get("patient_id")),
                    "case_id": str(meta.get("case_id")),
                    "label": int(label),
                    "score": float(score),
                }
            )
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
            grouper_key=GroupUsage(grouper_key="grouper", masking=MaskingStrategy.fullattention),
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
        print(
            f"Split {split_index:02d} test (last ckpt, patient-level): "
            f"AUROC={metrics['auroc']:.4f} AP={metrics['ap']:.4f} BCA={metrics['bca']:.4f}"
        )
        return {
            "split": split_index,
            "n_train_lesions": len(train_cases),
            "n_val_lesions": len(val_cases),
            "n_test_lesions": len(test_cases),
            "n_test_patients": pd.DataFrame(predictions)["patient_id"].nunique(),
            **metrics,
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
        per_split_rows.append(
            run_split(
                split_index,
                train_cases,
                val_cases,
                test_cases,
                args,
                cache_dir,
                output_dir,
                device,
            )
        )

    if not per_split_rows:
        raise SystemExit("No splits produced test metrics")
    per_split = pd.DataFrame(per_split_rows)
    per_split_path = output_dir / f"{args.dataset}_tokens_cv_per_split.csv"
    per_split.to_csv(per_split_path, index=False)
    print(f"Wrote {per_split_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
