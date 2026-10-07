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

SEED = 42


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


def run_optuna_tuning(
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    n_trials: int,
    n_jobs: int,
    seed: int,
) -> tuple[dict[str, float], float]:
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    inner_splits = list(sgkf.split(x_train_raw, y_train, groups=groups))

    def objective(trial: optuna.Trial) -> float:
        c = trial.suggest_float("C", 1e-6, 1e1, log=True)
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

        mean_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.5
        standard_error = (
            float(np.std(fold_aucs, ddof=1) / np.sqrt(len(fold_aucs)))
            if len(fold_aucs) > 1
            else 0.0
        )
        trial.set_user_attr("standard_error", standard_error)
        return mean_auc

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # One-standard-error rule: among trials statistically tied with the best
    # mean AUROC, select the strongest regularization (smallest C).
    threshold = study.best_value - float(
        study.best_trial.user_attrs.get("standard_error", 0.0)
    )
    eligible_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and trial.value >= threshold
        and "C" in trial.params
    ]
    selected_trial = min(eligible_trials, key=lambda trial: float(trial.params["C"]))
    return (
        {"C": float(selected_trial.params["C"])},
        float(selected_trial.value),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orig-features",
        type=Path,
        default=Path.home() / "experiments" / "Results" / "FMCIB" / "Melanoma_Fmcib_original_features.csv",
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
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel Optuna trials; 1 keeps seeded TPE reproducible",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials < 1:
        raise SystemExit("--trials must be at least 1")
    if args.n_jobs == 0:
        raise SystemExit("--n-jobs must not be 0")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pinfo = load_pinfo(args.pinfo)
    df_orig = attach_pinfo_labels(load_file(args.orig_features), pinfo)
    df_aug = attach_pinfo_labels(load_file(args.aug_features), pinfo)

    feature_cols = get_feature_cols(df_orig)
    folds = load_worc_splits(args.splits)

    per_split_rows: list[dict] = []

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

        best_params, inner_cv_auc = run_optuna_tuning(
            x_train_raw=x_train_raw,
            y_train=y_train,
            groups=groups,
            n_trials=args.trials,
            n_jobs=args.n_jobs,
            seed=SEED,
        )

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
                "inner_cv_auroc": inner_cv_auc,
            }
        )
        print(
            f"Split {split_idx:02d}: AUROC = {auc:.4f} | AP = {ap:.4f} | "
            f"Best C = {best_params.get('C'):.6g} | "
            f"Inner AUROC = {inner_cv_auc:.4f} | Pred Std = {np.std(y_pred):.4f}"
        )

    if not per_split_rows:
        raise SystemExit("No splits produced results")

    per_split_df = pd.DataFrame(per_split_rows)
    per_split_path = args.output_dir / "Melanoma_linear_cv_per_split.csv"
    per_split_df.to_csv(per_split_path, index=False)

    valid_aucs = per_split_df["auroc"].dropna().to_numpy()
    print("\n--- Summary ---")
    print(f"Mean Patient AUROC: {np.mean(valid_aucs):.4f} ± {np.std(valid_aucs):.4f}")
    print(f"Wrote {per_split_path}")


if __name__ == "__main__":
    main()