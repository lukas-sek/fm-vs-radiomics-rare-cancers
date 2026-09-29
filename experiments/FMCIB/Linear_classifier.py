"""Classify Melanoma BRAF status from FMCIB features with Optuna-tuned Logistic Regression.

Train: Melanoma_fmcib_augmentation_features.csv (train patients, all aug slots)
Test:  Melanoma_Fmcib_original_features.csv (test patients only)
Labels: metadata/pinfo_Melanoma.csv (Patient -> Diagnosis)
Splits: crossvalidationsplits/crossvalidationsplits_Melanoma.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42
_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent if _HERE.name.upper() == "FMCIB" else _HERE


def load_file(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if pred_cols:
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


def run_optuna_tuning(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    scoring: str,
    n_trials: int,
    n_jobs: int,
    seed: int,
) -> dict:
    def objective(trial: optuna.Trial) -> float:
        c = trial.suggest_float("C", 1e-4, 1e2, log=True)
        clf = LogisticRegression(
            C=c,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=seed,
        )
        cv = StratifiedGroupKFold(n_splits=5)
        scores = cross_val_score(
            clf,
            x_train,
            y_train,
            groups=groups,
            scoring=scoring,
            cv=cv,
            n_jobs=1,
        )
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    return study.best_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Melanoma BRAF linear probe on FMCIB features with WORC CV splits."
    )
    parser.add_argument(
        "--orig-features",
        type=Path,
        default=Path.home() / "experiments" / "Results" / "FMCIB" / "Melanoma_Fmcib_original_features.csv",
        help="Test-set features (original CTs)",
    )
    parser.add_argument(
        "--aug-features",
        type=Path,
        default=Path.home() / "experiments" / "Results" / "FMCIB" / "Melanoma_fmcib_augmentation_features.csv",
        help="Train-set features (augmented patches)",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path.home() / "crossvalidationsplits" / "crossvalidationsplits_Melanoma.csv",
        help="WORC wide CV CSV (i_train / i_test columns)",
    )
    parser.add_argument(
        "--pinfo",
        type=Path,
        default=Path.home() / "metadata" / "pinfo_Melanoma.csv",
        help="Patient-level Diagnosis labels (/home/lukas/metadata/...)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "experiments" / "Results" / "FMCIB",
        help="Directory for metrics CSVs (default: ~/experiments/Results/FMCIB)",
    )
    parser.add_argument("--scoring", type=str, default="roc_auc", help="Optuna optimization metric")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials per split")
    parser.add_argument("--n-jobs", type=int, default=4, help="Parallel jobs for Optuna")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pinfo = load_pinfo(args.pinfo)
    df_orig = attach_pinfo_labels(load_file(args.orig_features), pinfo)
    df_aug = attach_pinfo_labels(load_file(args.aug_features), pinfo)

    feature_cols = get_feature_cols(df_orig)
    aug_cols = get_feature_cols(df_aug)
    if feature_cols != aug_cols:
        raise SystemExit("Feature column mismatch between original and augmented files")
    if len(feature_cols) != 4096:
        print(f"Warning: expected 4096 pred_* features, got {len(feature_cols)}")

    folds = load_worc_splits(args.splits)
    print(f"Loaded {len(folds)} WORC folds from {args.splits}")
    print(f"Train source: {args.aug_features}")
    print(f"Test source:  {args.orig_features}")

    per_split_rows: list[dict] = []

    for split_idx, (train_pids, test_pids) in enumerate(folds):
        train_df = df_aug[df_aug["patient_id"].astype(str).isin(train_pids)].copy()
        test_df = df_orig[df_orig["patient_id"].astype(str).isin(test_pids)].copy()

        if train_df.empty or test_df.empty:
            print(f"Split {split_idx:02d}: skipped (empty train/test)")
            continue

        x_train_raw = train_df[feature_cols].to_numpy(dtype=np.float32)
        y_train = train_df["label"].to_numpy(dtype=np.int64)
        groups = train_df["patient_id"].astype(str).to_numpy()

        x_test_raw = test_df[feature_cols].to_numpy(dtype=np.float32)
        y_test = test_df["label"].to_numpy(dtype=np.int64)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train_raw)
        x_test = scaler.transform(x_test_raw)

        best_params = run_optuna_tuning(
            x_train=x_train,
            y_train=y_train,
            groups=groups,
            scoring=args.scoring,
            n_trials=args.trials,
            n_jobs=args.n_jobs,
            seed=SEED,
        )

        classifier = LogisticRegression(
            **best_params,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=SEED,
        )
        classifier.fit(x_train, y_train)

        probs = classifier.predict_proba(x_test)[:, 1]

        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, probs)
            ap = average_precision_score(y_test, probs)
        else:
            auc, ap = np.nan, np.nan

        per_split_rows.append(
            {
                "split": split_idx,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "n_train_patients": len(train_pids),
                "n_test_patients": len(test_pids),
                "auroc": auc,
                "ap": ap,
                "best_C": best_params.get("C"),
            }
        )
        print(
            f"Split {split_idx:02d}: AUROC = {auc:.4f} | AP = {ap:.4f} | "
            f"Best C = {best_params.get('C'):.5f}"
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

    per_split_path = args.output_dir / "Melanoma_linear_cv_per_split.csv"
    summary_path = args.output_dir / "Melanoma_linear_cv_summary.csv"
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
