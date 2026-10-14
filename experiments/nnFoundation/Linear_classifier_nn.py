"""Linear logistic regression on frozen nnFoundation lesion embeddings.

Training uses augmented feature rows for WORC training-fold patients.
Evaluation uses original feature rows for held-out patients and aggregates
lesion probabilities per patient.
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

DATASETS = ("Lipo", "Desmoid", "Liver", "GIST", "CRLM", "Melanoma")
BACKBONES = ("CNN", "ViT")
SEED = 42
BOOTSTRAP_SAMPLES = 10_000
# Keep C moderate: very large C + high-dim embeddings triggers lbfgs line-search failures.
C_SEARCH_LOW = 1e-6
C_SEARCH_HIGH = 10.0


def make_logistic_regression(c: float, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=c,
        l1_ratio=0.0,
        solver="newton-cg",
        max_iter=1000,
        random_state=seed,
    )


def discover_repo_root() -> Path:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (
            (parent / "crossvalidationsplits").is_dir()
            and (parent / "metadata").is_dir()
        ):
            return parent
    return script_path.parents[2]


REPO_ROOT = discover_repo_root()


def feature_stem(backbone: str) -> str:
    return "nncnn" if backbone == "CNN" else "nnvit"


def default_feature_paths(
    dataset: str, backbone: str
) -> tuple[Path, Path]:
    stem = feature_stem(backbone)
    root = REPO_ROOT / "Results" / "nnFoundation"
    return (
        root / f"{dataset}_{stem}_features.csv",
        root / f"{dataset}_{stem}_augmented_features.csv",
    )


def build_settings(args: argparse.Namespace) -> dict[str, Path | str | int]:
    orig_default, aug_default = default_feature_paths(args.dataset, args.backbone)
    feature_root = REPO_ROOT / "Results" / "nnFoundation"
    return {
        "dataset": args.dataset,
        "backbone": args.backbone,
        "orig_features": resolve_path(args.orig_features or orig_default),
        "aug_features": resolve_path(args.aug_features or aug_default),
        "splits": resolve_path(
            args.splits
            or REPO_ROOT
            / "crossvalidationsplits"
            / f"crossvalidationsplits_{args.dataset}.csv"
        ),
        "pinfo": resolve_path(
            args.pinfo or REPO_ROOT / "metadata" / f"pinfo_{args.dataset}.csv"
        ),
        "output_dir": resolve_path(args.output_dir or feature_root),
        "trials": int(args.trials),
        "n_jobs": int(args.n_jobs),
    }


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (REPO_ROOT / candidate).resolve()
    )


def normalize_pid(val: object) -> str:
    s = str(val).strip()
    return s[:-2] if s.endswith(".0") else s


def load_file(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Feature file not found: {path}")
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    if feature_cols:
        return sorted(
            feature_cols,
            key=lambda name: int(name.split("_", 1)[1]),
        )

    metadata_cols = {
        "case_id",
        "patient_id",
        "lesion_id",
        "augmentation_id",
        "augmentation_type",
        "augmentation_seed",
        "label",
        "dataset",
        "mask_source",
        "source_image_path",
        "source_mask_path",
        "segmentation_label",
        "component_id",
        "component_voxels",
        "center_x",
        "center_y",
        "center_z",
        "image_path",
    }
    return [c for c in df.columns if c not in metadata_cols]


def load_pinfo(path: Path) -> dict[str, int]:
    pinfo = pd.read_csv(path)
    if "Patient" not in pinfo.columns or "Diagnosis" not in pinfo.columns:
        raise SystemExit(f"pinfo CSV must have Patient and Diagnosis columns: {path}")
    return {
        normalize_pid(row["Patient"]): int(row["Diagnosis"])
        for _, row in pinfo.iterrows()
    }


def attach_pinfo_labels(df: pd.DataFrame, pinfo: dict[str, int]) -> pd.DataFrame:
    out = df.copy()
    if "patient_id" not in out.columns:
        raise SystemExit("Feature CSV must contain patient_id")
    out["patient_id"] = out["patient_id"].apply(normalize_pid)
    missing = sorted(set(out["patient_id"]) - set(pinfo))
    if missing:
        raise SystemExit(
            f"Missing pinfo labels for {len(missing)} patients (e.g. {missing[:5]})"
        )
    out["label"] = out["patient_id"].map(pinfo).astype(int)
    return out


def load_worc_splits(path: Path) -> list[tuple[set[str], set[str]]]:
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
        train_pids = {
            normalize_pid(x)
            for x in splits_df[train_col].dropna()
            if str(x).strip()
        }
        test_pids = {
            normalize_pid(x)
            for x in splits_df[test_col].dropna()
            if str(x).strip()
        }
        folds.append((train_pids, test_pids))
    return folds


def patient_balanced_weights(groups: np.ndarray) -> np.ndarray:
    """Give every patient the same total weight, regardless of row count."""
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights / weights.mean()


def bootstrap_patient_metric_cis(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Stratified patient bootstrap CIs for AUROC and average precision."""
    class_indices = [np.flatnonzero(y_true == label) for label in np.unique(y_true)]
    if len(class_indices) != 2:
        raise ValueError("Bootstrap confidence intervals require two classes")

    rng = np.random.default_rng(seed)
    bootstrap_aucs = np.empty(n_bootstrap, dtype=np.float64)
    bootstrap_aps = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        sampled_indices = np.concatenate(
            [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in class_indices
            ]
        )
        bootstrap_aucs[i] = roc_auc_score(
            y_true[sampled_indices], y_score[sampled_indices]
        )
        bootstrap_aps[i] = average_precision_score(
            y_true[sampled_indices], y_score[sampled_indices]
        )

    auc_bounds = np.clip(np.percentile(bootstrap_aucs, [2.5, 97.5]), 0.0, 1.0)
    ap_bounds = np.clip(np.percentile(bootstrap_aps, [2.5, 97.5]), 0.0, 1.0)
    return (
        (float(auc_bounds[0]), float(auc_bounds[1])),
        (float(ap_bounds[0]), float(ap_bounds[1])),
    )


def run_optuna_tuning(
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    n_trials: int,
    n_jobs: int,
    seed: int,
) -> dict[str, float]:
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    inner_splits = list(sgkf.split(x_train_raw, y_train, groups=groups))

    def objective(trial: optuna.Trial) -> float:
        c = trial.suggest_float("C", C_SEARCH_LOW, C_SEARCH_HIGH, log=True)
        fold_aucs: list[float] = []
        for tr_idx, val_idx in inner_splits:
            scaler = StandardScaler()
            x_inner_train = scaler.fit_transform(x_train_raw[tr_idx])
            x_inner_val = scaler.transform(x_train_raw[val_idx])

            clf = make_logistic_regression(c, seed)
            clf.fit(
                x_inner_train,
                y_train[tr_idx],
                sample_weight=patient_balanced_weights(groups[tr_idx]),
            )
            val_probs = clf.predict_proba(x_inner_val)[:, 1]

            val_df = pd.DataFrame(
                {
                    "patient_id": groups[val_idx],
                    "label": y_train[val_idx],
                    "prob": val_probs,
                }
            )
            patient_val = val_df.groupby("patient_id")[["label", "prob"]].mean()
            y_val = patient_val["label"].to_numpy(dtype=np.int64)
            p_val = patient_val["prob"].to_numpy(dtype=np.float64)
            if len(np.unique(y_val)) > 1:
                fold_aucs.append(roc_auc_score(y_val, p_val))

        return float(np.mean(fold_aucs)) if fold_aucs else 0.5

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    return study.best_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dataset-generic linear classification of nnFoundationCNN or "
            "nnFoundationViT embeddings over WORC CV splits."
        )
    )
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument(
        "--backbone",
        required=True,
        choices=BACKBONES,
        help="Feature family: CNN -> *_nncnn_*.csv, ViT -> *_nnvit_*.csv.",
    )
    parser.add_argument("--orig-features", type=Path, default=None)
    parser.add_argument("--aug-features", type=Path, default=None)
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--pinfo", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = build_settings(args)
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset:   {settings['dataset']}")
    print(f"Backbone:  {settings['backbone']}")
    print(f"Original:  {settings['orig_features']}")
    print(f"Augmented: {settings['aug_features']}")

    pinfo = load_pinfo(Path(settings["pinfo"]))
    df_orig = attach_pinfo_labels(load_file(Path(settings["orig_features"])), pinfo)
    df_aug = attach_pinfo_labels(load_file(Path(settings["aug_features"])), pinfo)

    feature_cols = get_feature_cols(df_orig)
    if get_feature_cols(df_aug) != feature_cols:
        raise SystemExit("Original and augmented feature columns do not match")
    folds = load_worc_splits(Path(settings["splits"]))

    per_split_rows: list[dict] = []
    patient_prediction_frames: list[pd.DataFrame] = []

    for split_idx, (train_pids, test_pids) in enumerate(folds):
        train_df = df_aug[df_aug["patient_id"].isin(train_pids)].copy()
        test_df = df_orig[df_orig["patient_id"].isin(test_pids)].copy()

        if train_df.empty or test_df.empty:
            print(f"Split {split_idx:02d}: skipped (empty train/test)")
            continue

        x_train_raw = train_df[feature_cols].to_numpy(dtype=np.float32)
        y_train = train_df["label"].to_numpy(dtype=np.int64)
        groups = train_df["patient_id"].to_numpy()
        x_test_raw = test_df[feature_cols].to_numpy(dtype=np.float32)

        best_params = run_optuna_tuning(
            x_train_raw=x_train_raw,
            y_train=y_train,
            groups=groups,
            n_trials=int(settings["trials"]),
            n_jobs=int(settings["n_jobs"]),
            seed=SEED,
        )

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train_raw)
        x_test = scaler.transform(x_test_raw)

        classifier = make_logistic_regression(float(best_params["C"]), SEED)
        classifier.fit(
            x_train,
            y_train,
            sample_weight=patient_balanced_weights(groups),
        )

        test_df["prob"] = classifier.predict_proba(x_test)[:, 1]
        patient_test = test_df.groupby("patient_id")[["label", "prob"]].mean()
        split_predictions = patient_test.reset_index()
        split_predictions.insert(0, "split", split_idx)
        patient_prediction_frames.append(split_predictions)

        y_true = patient_test["label"].to_numpy(dtype=np.int64)
        y_pred = patient_test["prob"].to_numpy(dtype=np.float64)
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_pred)
            ap = average_precision_score(y_true, y_pred)
        else:
            auc, ap = np.nan, np.nan

        per_split_rows.append(
            {
                "split": split_idx,
                "n_train_samples": len(train_df),
                "n_test_patients": len(patient_test),
                "auroc": auc,
                "ap": ap,
                "prob_std": float(np.std(y_pred)),
                "best_C": best_params.get("C"),
            }
        )
        print(
            f"Split {split_idx:02d}: AUROC = {auc:.4f} | AP = {ap:.4f} | "
            f"Best C = {best_params.get('C'):.6f} | Pred Std = {np.std(y_pred):.4f}"
        )

    if not per_split_rows:
        raise SystemExit("No splits produced results")

    stem = feature_stem(str(settings["backbone"]))
    prefix = f"{settings['dataset']}_{stem}_linear"
    per_split_df = pd.DataFrame(per_split_rows)
    per_split_path = output_dir / f"{prefix}_cv_per_split.csv"
    per_split_df.to_csv(per_split_path, index=False)

    patient_predictions = pd.concat(patient_prediction_frames, ignore_index=True)
    predictions_path = output_dir / f"{prefix}_cv_patient_predictions.csv"
    patient_predictions.to_csv(predictions_path, index=False)

    patient_oof = patient_predictions.groupby("patient_id").agg(
        label=("label", "first"),
        prob=("prob", "mean"),
        n_test_appearances=("split", "size"),
    )
    y_oof = patient_oof["label"].to_numpy(dtype=np.int64)
    p_oof = patient_oof["prob"].to_numpy(dtype=np.float64)

    pooled_auc = roc_auc_score(y_oof, p_oof)
    pooled_ap = average_precision_score(y_oof, p_oof)
    auc_ci, ap_ci = bootstrap_patient_metric_cis(
        y_true=y_oof,
        y_score=p_oof,
        n_bootstrap=BOOTSTRAP_SAMPLES,
        seed=SEED,
    )

    summary_df = pd.DataFrame(
        [
            {
                "metric": "auroc",
                "split_mean": per_split_df["auroc"].mean(),
                "split_std": per_split_df["auroc"].std(ddof=0),
                "pooled_oof": pooled_auc,
                "ci_95_lower": auc_ci[0],
                "ci_95_upper": auc_ci[1],
                "n_patients": len(patient_oof),
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
            },
            {
                "metric": "ap",
                "split_mean": per_split_df["ap"].mean(),
                "split_std": per_split_df["ap"].std(ddof=0),
                "pooled_oof": pooled_ap,
                "ci_95_lower": ap_ci[0],
                "ci_95_upper": ap_ci[1],
                "n_patients": len(patient_oof),
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
            },
        ]
    )
    summary_path = output_dir / f"{prefix}_cv_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n--- Summary ---")
    print(
        f"Patient AUROC: {pooled_auc:.4f} "
        f"(95% CI {auc_ci[0]:.4f}-{auc_ci[1]:.4f}); "
        f"split mean ± SD {summary_df.iloc[0]['split_mean']:.4f} ± "
        f"{summary_df.iloc[0]['split_std']:.4f}"
    )
    print(
        f"Patient AP: {pooled_ap:.4f} "
        f"(95% CI {ap_ci[0]:.4f}-{ap_ci[1]:.4f}); "
        f"split mean ± SD {summary_df.iloc[1]['split_mean']:.4f} ± "
        f"{summary_df.iloc[1]['split_std']:.4f}"
    )
    print(f"Wrote {per_split_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
