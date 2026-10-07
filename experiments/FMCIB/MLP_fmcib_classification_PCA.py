"""Classify Melanoma BRAF status from FMCIB features across WORC CV splits.

Train: Melanoma_fmcib_augmentation_features.csv (train patients, all aug slots)
Test:  Melanoma_Fmcib_original_features.csv (test patients only)
Labels: metadata/pinfo_Melanoma.csv (Patient -> Diagnosis)
Splits: crossvalidationsplits/crossvalidationsplits_Melanoma.csv

Parameters are loaded from a YAML config (default: mlp_config.yaml next to this script).
CLI flags override individual YAML values when provided.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent if _HERE.name.upper() == "FMCIB" else _HERE
PATH_ROOT = Path.home()
DEFAULT_CONFIG = _HERE / "mlp_config.yaml"
SEED = 42
PCA_COMPONENTS = (5, 10, 20, 40, 60)


class MLP(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        if in_features == 4096:
            self.net = nn.Sequential(
                nn.Linear(4096, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_features, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_file(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def normalize_pid(val: object) -> str:
    value = str(val).strip()
    return value[:-2] if value.endswith(".0") else value


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if pred_cols:
        # Sort by numeric index so pred_2 follows pred_1, not pred_10
        return sorted(pred_cols, key=lambda c: int(c.split("_", 1)[1]))

    feature_cols = [c for c in df.columns if c.startswith("feature_") or c.startswith("emb_")]
    if feature_cols:
        return feature_cols

    metadata_cols = {
        "patient_id",
        "lesion_id",
        "aug_id",
        "aug_type",
        "label",
        "image_path",
        "coordX",
        "coordY",
        "coordZ",
        "dataset",
        "mask_source",
    }
    return [c for c in df.columns if c not in metadata_cols]


def load_pinfo(path: Path) -> dict[str, int]:
    pinfo = pd.read_csv(path)
    if "Patient" not in pinfo.columns or "Diagnosis" not in pinfo.columns:
        raise SystemExit(f"pinfo CSV must have Patient and Diagnosis columns: {path}")
    return {normalize_pid(row["Patient"]): int(row["Diagnosis"]) for _, row in pinfo.iterrows()}


def attach_pinfo_labels(df: pd.DataFrame, pinfo: dict[str, int]) -> pd.DataFrame:
    out = df.copy()
    if "patient_id" not in out.columns:
        raise SystemExit("Feature CSV must contain patient_id")
    out["patient_id"] = out["patient_id"].apply(normalize_pid)
    missing = sorted(set(out["patient_id"]) - set(pinfo))
    if missing:
        raise SystemExit(f"Missing pinfo labels for {len(missing)} patients (e.g. {missing[:5]})")
    out["label"] = out["patient_id"].map(pinfo).astype(int)
    return out


def load_worc_splits(path: Path) -> list[tuple[set[str], set[str]]]:
    """Parse WORC wide CV CSV into (train_pids, test_pids) per fold."""
    splits_df = pd.read_csv(path)
    fold_ids = sorted(
        {
            int(m.group(1))
            for col in splits_df.columns
            if (m := re.fullmatch(r"(\d+)_train", str(col)))
        }
    )
    if not fold_ids:
        raise SystemExit(f"No '{{i}}_train' columns found in {path}")

    folds: list[tuple[set[str], set[str]]] = []
    for i in fold_ids:
        train_col, test_col = f"{i}_train", f"{i}_test"
        if test_col not in splits_df.columns:
            raise SystemExit(f"Missing column {test_col} in {path}")
        train_pids = {normalize_pid(x) for x in splits_df[train_col].dropna() if str(x).strip()}
        test_pids = {normalize_pid(x) for x in splits_df[test_col].dropna() if str(x).strip()}
        folds.append((train_pids, test_pids))
    return folds


def patient_balanced_weights(groups: np.ndarray) -> np.ndarray:
    """Give every patient the same total training weight."""
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return (weights / weights.mean()).astype(np.float32)


def patient_mean_features(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Average feature rows so every patient contributes once to PCA."""
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    means = np.empty((len(unique_groups), x.shape[1]), dtype=np.float64)
    for group_idx in range(len(unique_groups)):
        means[group_idx] = x[inverse == group_idx].mean(axis=0)
    return means


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Config not found: {path}")
    with open(path, encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"Config must be a mapping: {path}")
    return cfg


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_weight = 0.0
    for x_b, y_b, weight_b in loader:
        x_b, y_b, weight_b = x_b.to(device), y_b.to(device), weight_b.to(device)
        optimizer.zero_grad()
        logits = model(x_b)
        losses = criterion(logits, y_b)
        loss = (losses * weight_b).sum() / weight_b.sum()
        loss.backward()
        optimizer.step()
        total_loss += float((losses.detach() * weight_b).sum().item())
        total_weight += float(weight_b.sum().item())
    return total_loss / total_weight


@torch.no_grad()
def evaluate(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[float, float, float]:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=batch_size,
        shuffle=False,
    )

    preds = []
    for (x_b,) in loader:
        logits = model(x_b.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
        preds.extend(probs)

    eval_df = pd.DataFrame({
        "patient_id": patient_ids,
        "label": y,
        "prob": preds,
    })

    # Aggregate lesion probabilities per patient
    patient_df = eval_df.groupby("patient_id")[["label", "prob"]].mean()
    y_true = patient_df["label"].to_numpy(dtype=np.int64)
    y_pred = patient_df["prob"].to_numpy(dtype=np.float64)

    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
    else:
        auc, ap = np.nan, np.nan

    return auc, ap, float(np.std(y_pred))


def run_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    scheduler_step_size: int,
    scheduler_gamma: float,
    device: torch.device,
    n_components: int,
    seed: int,
) -> tuple[float, float, float, float]:
    seed_everything(seed)
    x_train_raw = train_df[feature_cols].to_numpy(dtype=np.float32)
    x_test_raw = test_df[feature_cols].to_numpy(dtype=np.float32)
    train_pids = train_df["patient_id"].to_numpy()
    patient_means = patient_mean_features(x_train_raw, train_pids)
    if n_components >= len(patient_means):
        raise ValueError(
            f"PCA components ({n_components}) must be smaller than the number "
            f"of training patients ({len(patient_means)})"
        )

    scaler = StandardScaler()
    scaler.fit(patient_means)
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(scaler.transform(patient_means))
    x_train = pca.transform(scaler.transform(x_train_raw)).astype(np.float32)
    x_test = pca.transform(scaler.transform(x_test_raw)).astype(np.float32)
    explained_variance = float(pca.explained_variance_ratio_.sum())

    y_train = train_df["label"].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=np.float32)
    test_pids = test_df["patient_id"].to_numpy()
    train_weights = patient_balanced_weights(train_pids)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train),
            torch.from_numpy(train_weights),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    model = MLP(in_features=n_components).to(device)
    # Add weight decay and reduce learning rate to mitigate overfitting
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    for _ in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

    auc, ap, prob_std = evaluate(
        model=model,
        x=x_test,
        y=y_test,
        patient_ids=test_pids,
        device=device,
        batch_size=batch_size,
    )
    return auc, ap, prob_std, explained_variance


def select_pca_components(
    train_df: pd.DataFrame,
    original_df: pd.DataFrame,
    feature_cols: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    scheduler_step_size: int,
    scheduler_gamma: float,
    device: torch.device,
    seed: int,
) -> tuple[int, float]:
    """Select PCA dimensionality by patient-grouped inner cross-validation."""
    y = train_df["label"].to_numpy(dtype=np.int64)
    groups = train_df["patient_id"].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    inner_splits = list(sgkf.split(train_df, y, groups=groups))
    min_inner_patients = min(len(np.unique(groups[tr_idx])) for tr_idx, _ in inner_splits)
    candidates = [n for n in PCA_COMPONENTS if n < min_inner_patients]
    if not candidates:
        raise ValueError(
            f"No PCA candidate is smaller than the minimum inner-training "
            f"patient count ({min_inner_patients})"
        )

    scores = {n: [] for n in candidates}
    for fold_idx, (tr_idx, val_idx) in enumerate(inner_splits):
        inner_train = train_df.iloc[tr_idx]
        validation_pids = set(groups[val_idx])
        inner_validation = original_df[
            original_df["patient_id"].isin(validation_pids)
        ].copy()
        if inner_validation.empty:
            raise ValueError(f"Inner fold {fold_idx} has no original validation rows")

        for n_components in candidates:
            auc, _, _, _ = run_split(
                train_df=inner_train,
                test_df=inner_validation,
                feature_cols=feature_cols,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                scheduler_step_size=scheduler_step_size,
                scheduler_gamma=scheduler_gamma,
                device=device,
                n_components=n_components,
                seed=seed + fold_idx,
            )
            if not np.isnan(auc):
                scores[n_components].append(auc)

    mean_scores = {
        n: float(np.mean(fold_scores)) if fold_scores else float("-inf")
        for n, fold_scores in scores.items()
    }
    best_n_components = max(candidates, key=lambda n: (mean_scores[n], -n))
    return best_n_components, mean_scores[best_n_components]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Melanoma BRAF MLP on FMCIB features with WORC CV splits."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config path (default: {DEFAULT_CONFIG})",
    )
    # Optional overrides (None = keep YAML value)
    parser.add_argument("--orig-features", type=Path, default=None)
    parser.add_argument("--aug-features", type=Path, default=None)
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--pinfo", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()


def build_settings(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_yaml_config(args.config)
    paths = cfg.get("paths") or {}
    training = cfg.get("training") or {}

    def pick(cli_val: Any, *keys: str, default: Any = None) -> Any:
        if cli_val is not None:
            return cli_val
        section = paths if keys[0] in paths else training
        cur: Any = section
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur if cur is not None else default

    settings = {
        "orig_features": resolve_path(
            pick(args.orig_features, "orig_features"),
            PATH_ROOT,
        ),
        "aug_features": resolve_path(
            pick(args.aug_features, "aug_features"),
            PATH_ROOT,
        ),
        "splits": resolve_path(pick(args.splits, "splits"), PATH_ROOT),
        "pinfo": resolve_path(pick(args.pinfo, "pinfo"), PATH_ROOT),
        "output_dir": resolve_path(
            pick(args.output_dir, "output_dir", default="experiments/Results/FMCIB"),
            PATH_ROOT,
        ),
        "epochs": int(pick(args.epochs, "epochs", default=100)),
        "batch_size": int(pick(args.batch_size, "batch_size", default=32)),
        "lr": float(pick(args.lr, "lr", default=0.0125)),
        "scheduler_step_size": int(training.get("scheduler_step_size", 30)),
        "scheduler_gamma": float(training.get("scheduler_gamma", 0.1)),
    }
    return settings


def main() -> None:
    args = parse_args()
    cfg = build_settings(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Config: {args.config.resolve() if args.config.exists() else args.config}")
    print(f"Device: {device}")
    print(f"pinfo:          {cfg['pinfo']}")
    print(f"orig_features:  {cfg['orig_features']}")
    print(f"aug_features:   {cfg['aug_features']}")
    print(f"splits:         {cfg['splits']}")
    print(f"output_dir:     {cfg['output_dir']}")
    print(
        f"Training: epochs={cfg['epochs']} batch_size={cfg['batch_size']} "
        f"lr={cfg['lr']} step={cfg['scheduler_step_size']} gamma={cfg['scheduler_gamma']}"
    )

    pinfo = load_pinfo(cfg["pinfo"])
    df_orig = attach_pinfo_labels(load_file(cfg["orig_features"]), pinfo)
    df_aug = attach_pinfo_labels(load_file(cfg["aug_features"]), pinfo)

    feature_cols = get_feature_cols(df_orig)
    aug_cols = get_feature_cols(df_aug)
    if feature_cols != aug_cols:
        raise SystemExit("Feature column mismatch between original and augmented files")
    if len(feature_cols) != 4096:
        print(f"Warning: expected 4096 pred_* features, got {len(feature_cols)}")

    folds = load_worc_splits(cfg["splits"])
    print(f"Loaded {len(folds)} WORC folds from {cfg['splits']}")
    print(f"Train source: {cfg['aug_features']}")
    print(f"Test source:  {cfg['orig_features']}")

    per_split_rows: list[dict] = []

    for split_idx, (train_pids, test_pids) in enumerate(folds):
        train_data = df_aug[df_aug["patient_id"].isin(train_pids)].copy()
        test_data = df_orig[df_orig["patient_id"].isin(test_pids)].copy()

        if train_data.empty or test_data.empty:
            print(f"Split {split_idx:02d}: skipped (empty train/test)")
            continue

        best_n_components, inner_cv_auc = select_pca_components(
            train_df=train_data,
            original_df=df_orig,
            feature_cols=feature_cols,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            scheduler_step_size=cfg["scheduler_step_size"],
            scheduler_gamma=cfg["scheduler_gamma"],
            device=device,
            seed=SEED,
        )

        auc, ap, prob_std, explained_variance = run_split(
            train_df=train_data,
            test_df=test_data,
            feature_cols=feature_cols,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            scheduler_step_size=cfg["scheduler_step_size"],
            scheduler_gamma=cfg["scheduler_gamma"],
            device=device,
            n_components=best_n_components,
            seed=SEED + split_idx,
        )
        per_split_rows.append(
            {
                "split": split_idx,
                "n_train": len(train_data),
                "n_test": len(test_data),
                "n_train_patients": len(train_pids),
                "n_test_patients": len(test_pids),
                "auroc": auc,
                "ap": ap,
                "prob_std": prob_std,
                "best_n_components": best_n_components,
                "explained_variance": explained_variance,
                "inner_cv_auroc": inner_cv_auc,
            }
        )
        print(
            f"Split {split_idx:02d}: AUROC = {auc:.4f} | AP = {ap:.4f} | "
            f"PCA = {best_n_components} ({explained_variance:.1%} variance) | "
            f"Inner AUROC = {inner_cv_auc:.4f}"
        )

    if not per_split_rows:
        raise SystemExit("No splits produced results")

    per_split_df = pd.DataFrame(per_split_rows)
    split_aucs = per_split_df["auroc"].to_numpy(dtype=float)
    split_aps = per_split_df["ap"].to_numpy(dtype=float)

    summary_df = pd.DataFrame(
        [
            {
                "metric": "auroc",
                "mean": float(np.nanmean(split_aucs)),
                "std": float(np.nanstd(split_aucs)),
                "median": float(np.nanmedian(split_aucs)),
                "n_splits": len(per_split_df),
            },
            {
                "metric": "ap",
                "mean": float(np.nanmean(split_aps)),
                "std": float(np.nanstd(split_aps)),
                "median": float(np.nanmedian(split_aps)),
                "n_splits": len(per_split_df),
            },
        ]
    )

    out_dir = cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    per_split_path = out_dir / "Melanoma_mlp_pca_cv_per_split.csv"
    summary_path = out_dir / "Melanoma_mlp_pca_cv_summary.csv"
    per_split_df.to_csv(per_split_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n--- Summary across splits ---")
    print(
        f"AUROC: {summary_df.loc[summary_df['metric'] == 'auroc', 'mean'].iloc[0]:.4f} ± "
        f"{summary_df.loc[summary_df['metric'] == 'auroc', 'std'].iloc[0]:.4f} "
        f"(Median: {summary_df.loc[summary_df['metric'] == 'auroc', 'median'].iloc[0]:.4f})"
    )
    print(
        f"AP:    {summary_df.loc[summary_df['metric'] == 'ap', 'mean'].iloc[0]:.4f} ± "
        f"{summary_df.loc[summary_df['metric'] == 'ap', 'std'].iloc[0]:.4f} "
        f"(Median: {summary_df.loc[summary_df['metric'] == 'ap', 'median'].iloc[0]:.4f})"
    )
    print(f"Wrote {per_split_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
