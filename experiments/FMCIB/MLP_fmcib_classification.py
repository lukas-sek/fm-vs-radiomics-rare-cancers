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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent if _HERE.name.upper() == "FMCIB" else _HERE
PATH_ROOT = Path.home()
DEFAULT_CONFIG = _HERE / "mlp_config.yaml"


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
                nn.Linear(in_features, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_file(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


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
    return {str(row["Patient"]): int(row["Diagnosis"]) for _, row in pinfo.iterrows()}


def attach_pinfo_labels(df: pd.DataFrame, pinfo: dict[str, int]) -> pd.DataFrame:
    out = df.copy()
    if "patient_id" not in out.columns:
        raise SystemExit("Feature CSV must contain patient_id")
    missing = sorted(set(out["patient_id"].astype(str)) - set(pinfo))
    if missing:
        raise SystemExit(f"Missing pinfo labels for {len(missing)} patients (e.g. {missing[:5]})")
    out["label"] = out["patient_id"].astype(str).map(pinfo).astype(int)
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
        train_pids = {str(x) for x in splits_df[train_col].dropna() if str(x).strip()}
        test_pids = {str(x) for x in splits_df[test_col].dropna() if str(x).strip()}
        folds.append((train_pids, test_pids))
    return folds


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
    for x_b, y_b in loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = model(x_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_b)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_targets = [], []
    for x_b, y_b in loader:
        logits = model(x_b.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(y_b.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
    else:
        auc, ap = np.nan, np.nan

    return auc, ap, y_true, y_pred


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
) -> tuple[float, float]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=np.float32))
    x_test = scaler.transform(test_df[feature_cols].to_numpy(dtype=np.float32))

    y_train = train_df["label"].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
        batch_size=batch_size,
        shuffle=False,
    )

    model = MLP(in_features=len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

    # Fixed protocol: evaluate once after the final epoch (no test-set model selection).
    auc, ap, _, _ = evaluate(model, test_loader, device)
    return auc, ap


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
            pick(args.output_dir, "output_dir", default="experiments/results/fmcib"),
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
        train_data = df_aug[df_aug["patient_id"].astype(str).isin(train_pids)].copy()
        test_data = df_orig[df_orig["patient_id"].astype(str).isin(test_pids)].copy()

        if train_data.empty or test_data.empty:
            print(f"Split {split_idx:02d}: skipped (empty train/test)")
            continue

        auc, ap = run_split(
            train_df=train_data,
            test_df=test_data,
            feature_cols=feature_cols,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            scheduler_step_size=cfg["scheduler_step_size"],
            scheduler_gamma=cfg["scheduler_gamma"],
            device=device,
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
            }
        )
        print(f"Split {split_idx:02d}: AUROC = {auc:.4f} | AP = {ap:.4f}")

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
    per_split_path = out_dir / "Melanoma_mlp_cv_per_split.csv"
    summary_path = out_dir / "Melanoma_mlp_cv_summary.csv"
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
