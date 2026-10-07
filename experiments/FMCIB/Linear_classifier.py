from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42
BOOTSTRAP_SAMPLES = 10_000


def normalize_pid(val: object) -> str:
    s = str(val).strip()
    return s[:-2] if s.endswith(".0") else s


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
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        bootstrap_aucs[i] = roc_auc_score(y_true[sampled_indices], y_score[sampled_indices])
        bootstrap_aps[i] = average_precision_score(
            y_true[sampled_indices], y_score[sampled_indices]
        )

    auc_bounds = np.clip(np.percentile(bootstrap_aucs, [2.5, 97.5]), 0.0, 1.0)
    ap_bounds = np.clip(np.percentile(bootstrap_aps, [2.5, 97.5]), 0.0, 1.0)
    auc_ci = (float(auc_bounds[0]), float(auc_bounds[1]))
    ap_ci = (float(ap_bounds[0]), float(ap_bounds[1]))
    return auc_ci, ap_ci


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
        c = trial.suggest_float("C", 1e-6, 1e3, log=True)
        fold_aucs: list[float] = []
        for tr_idx, val_idx in inner_splits:
            # Fit preprocessing only on this inner-training fold.
            scaler = StandardScaler()
            x_inner_train = scaler.fit_transform(x_train_raw[tr_idx])
            x_inner_val = scaler.transform(x_train_raw[val_idx])

            clf = LogisticRegression(
                C=c,
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                random_state=seed,
            )
            clf.fit(
                x_inner_train,
                y_train[tr_idx],
                sample_weight=patient_balanced_weights(groups[tr_idx]),
            )
            val_probs = clf.predict_proba(x_inner_val)[:, 1]

            # Aggregate validation crops/lesions by patient
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orig-features",
        type=Path,
        default=Path.home() / "experiments" / "Results" / "FMCIB" / "Melanoma_fmcib_original_features.csv",
    )
    parser.add_argument(
        "--aug-features",
        type=Path,
        default=Path.home() / "experiments" / "Results" / "FMCIB" / "Melanoma_fmcib_augmentation_features.csv",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path.home() / "crossvalidationsplits" / "crossvalidationsplits_Melanoma.csv",
    )
    parser.add_argument(
        "--pinfo",
        type=Path,
        default=Path.home() / "metadata" / "pinfo_Melanoma.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "experiments" / "Results" / "FMCIB",
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pinfo = load_pinfo(args.pinfo)
    df_orig = attach_pinfo_labels(load_file(args.orig_features), pinfo)
    df_aug = attach_pinfo_labels(load_file(args.aug_features), pinfo)

    feature_cols = get_feature_cols(df_orig)
    folds = load_worc_splits(args.splits)

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
            n_trials=args.trials,
            n_jobs=args.n_jobs,
            seed=SEED,
        )

        # Refit preprocessing and classifier on this outer training split only.
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train_raw)
        x_test = scaler.transform(x_test_raw)

        classifier = LogisticRegression(
            **best_params,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=SEED,
        )
        classifier.fit(
            x_train,
            y_train,
            sample_weight=patient_balanced_weights(groups),
        )

        # Aggregate test lesion predictions per patient
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

    per_split_df = pd.DataFrame(per_split_rows)
    per_split_path = args.output_dir / "Melanoma_linear_cv_per_split.csv"
    per_split_df.to_csv(per_split_path, index=False)

    patient_predictions = pd.concat(patient_prediction_frames, ignore_index=True)
    predictions_path = args.output_dir / "Melanoma_linear_cv_patient_predictions.csv"
    patient_predictions.to_csv(predictions_path, index=False)

    # Each patient can occur in several predefined test splits. Average those
    # out-of-fold probabilities before bootstrapping patients.
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
    summary_path = args.output_dir / "Melanoma_linear_cv_summary.csv"
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