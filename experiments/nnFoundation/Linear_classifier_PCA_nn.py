"""PCA + linear logistic regression on frozen nnFoundation lesion embeddings.

Training uses augmented feature rows for WORC training-fold patients.
Inner CV and outer evaluation use original feature rows for held-out patients.
PCA components are selected jointly with logistic C via Optuna.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from Linear_classifier_nn import (
    BACKBONES,
    BOOTSTRAP_SAMPLES,
    C_SEARCH_HIGH,
    C_SEARCH_LOW,
    DATASETS,
    SEED,
    attach_pinfo_labels,
    bootstrap_patient_metric_cis,
    build_settings,
    feature_stem,
    get_feature_cols,
    load_file,
    load_pinfo,
    load_worc_splits,
    make_logistic_regression,
    patient_balanced_weights,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
PCA_COMPONENTS = (5, 10, 20, 40, 60)


def patient_mean_features(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Average feature rows so every patient contributes once to PCA."""
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    means = np.empty((len(unique_groups), x.shape[1]), dtype=np.float64)
    for group_idx in range(len(unique_groups)):
        means[group_idx] = x[inverse == group_idx].mean(axis=0)
    return means


def run_optuna_tuning(
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    x_original_raw: np.ndarray,
    y_original: np.ndarray,
    original_groups: np.ndarray,
    n_trials: int,
    n_jobs: int,
    seed: int,
) -> dict[str, float | int]:
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    inner_splits = list(sgkf.split(x_train_raw, y_train, groups=groups))
    min_inner_patients = min(
        len(np.unique(groups[tr_idx])) for tr_idx, _ in inner_splits
    )
    valid_components = [n for n in PCA_COMPONENTS if n < min_inner_patients]
    if not valid_components:
        raise ValueError(
            f"No PCA candidate is smaller than the minimum inner-training "
            f"patient count ({min_inner_patients})"
        )

    inner_fold_data: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    max_components = max(valid_components)
    for tr_idx, val_idx in inner_splits:
        validation_pids = np.unique(groups[val_idx])
        original_val_idx = np.flatnonzero(np.isin(original_groups, validation_pids))
        if len(original_val_idx) == 0:
            raise ValueError("Inner fold has no original validation rows")

        patient_means = patient_mean_features(x_train_raw[tr_idx], groups[tr_idx])
        scaler = StandardScaler()
        scaler.fit(patient_means)

        pca = PCA(n_components=max_components, svd_solver="full")
        pca.fit(scaler.transform(patient_means))
        x_inner_train = pca.transform(scaler.transform(x_train_raw[tr_idx]))
        x_inner_val = pca.transform(
            scaler.transform(x_original_raw[original_val_idx])
        )
        inner_fold_data.append(
            (
                x_inner_train,
                x_inner_val,
                y_train[tr_idx],
                groups[tr_idx],
                y_original[original_val_idx],
                original_groups[original_val_idx],
            )
        )

    def objective(trial: optuna.Trial) -> float:
        c = trial.suggest_float("C", C_SEARCH_LOW, C_SEARCH_HIGH, log=True)
        n_components = trial.suggest_categorical("n_components", valid_components)
        fold_aucs: list[float] = []
        for (
            x_inner_train,
            x_inner_val,
            y_inner_train,
            inner_groups,
            y_inner_val,
            inner_val_groups,
        ) in inner_fold_data:
            clf = make_logistic_regression(c, seed)
            clf.fit(
                x_inner_train[:, :n_components],
                y_inner_train,
                sample_weight=patient_balanced_weights(inner_groups),
            )
            val_probs = clf.predict_proba(x_inner_val[:, :n_components])[:, 1]

            val_df = pd.DataFrame(
                {
                    "patient_id": inner_val_groups,
                    "label": y_inner_val,
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
            "Dataset-generic PCA + linear classification of nnFoundationCNN "
            "or nnFoundationViT embeddings over WORC CV splits."
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

        original_train_df = df_orig[df_orig["patient_id"].isin(train_pids)]
        x_original_train_raw = original_train_df[feature_cols].to_numpy(
            dtype=np.float32
        )
        y_original_train = original_train_df["label"].to_numpy(dtype=np.int64)
        original_train_groups = original_train_df["patient_id"].to_numpy()

        best_params = run_optuna_tuning(
            x_train_raw=x_train_raw,
            y_train=y_train,
            groups=groups,
            x_original_raw=x_original_train_raw,
            y_original=y_original_train,
            original_groups=original_train_groups,
            n_trials=int(settings["trials"]),
            n_jobs=int(settings["n_jobs"]),
            seed=SEED,
        )

        best_c = float(best_params["C"])
        best_n_components = int(best_params["n_components"])

        outer_patient_means = patient_mean_features(x_train_raw, groups)
        scaler = StandardScaler()
        scaler.fit(outer_patient_means)
        pca = PCA(n_components=best_n_components, svd_solver="full")
        pca.fit(scaler.transform(outer_patient_means))
        x_train = pca.transform(scaler.transform(x_train_raw))
        x_test = pca.transform(scaler.transform(x_test_raw))
        explained_variance = float(pca.explained_variance_ratio_.sum())

        classifier = make_logistic_regression(best_c, SEED)
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
                "best_C": best_c,
                "best_n_components": best_n_components,
                "explained_variance": explained_variance,
            }
        )
        print(
            f"Split {split_idx:02d}: AUROC = {auc:.4f} | AP = {ap:.4f} | "
            f"Best C = {best_c:.6f} | PCA = {best_n_components} "
            f"({explained_variance:.1%} variance) | Pred Std = {np.std(y_pred):.4f}"
        )

    if not per_split_rows:
        raise SystemExit("No splits produced results")

    stem = feature_stem(str(settings["backbone"]))
    prefix = f"{settings['dataset']}_{stem}_linear_pca"
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
