"""Optuna sweep for PCA-reduced nnFoundation MLP hyperparameters.

Each trial suggests training hyperparameters and a PCA dimensionality, then
scores mean patient-level AUROC on inner validation folds built only from
WORC training patients (original rows). Outer test folds are never used.

Example:

    python experiments/nnFoundation/MLP_nn_hyperparam_sweep_PCA.py `
      --dataset Liver --backbone CNN --trials 30
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedGroupKFold

import MLP_nn_classification as base
import MLP_nn_classification_PCA as pca_clf

optuna.logging.set_verbosity(optuna.logging.WARNING)

INNER_FOLDS = 3
DEFAULT_TRIALS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Nested Optuna sweep for PCA-reduced nnFoundation MLP "
            "hyperparameters. Scores inner-fold patient AUROC; does not use "
            "WORC test folds."
        )
    )
    parser.add_argument("--dataset", required=True, choices=base.DATASETS)
    parser.add_argument("--backbone", required=True, choices=base.BACKBONES)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "YAML to read (and optionally update). Default depends on "
            "--backbone: CNN -> mlp_CNN.yaml, ViT -> mlp_ViT.yaml."
        ),
    )
    parser.add_argument("--orig-features", type=Path, default=None)
    parser.add_argument("--aug-features", type=Path, default=None)
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--pinfo", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--max-splits",
        type=int,
        default=None,
        help="Use only the first N outer WORC folds for inner CV (default: all).",
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument(
        "--inner-folds",
        type=int,
        default=INNER_FOLDS,
        help="Patient-grouped inner folds per outer training set (default: 3).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Optuna parallelism. Keep 1 when using CUDA.",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Overwrite the training block of --config with the best params.",
    )
    return parser.parse_args()


def suggest_hyperparams(
    trial: optuna.Trial, pca_candidates: list[int]
) -> dict[str, Any]:
    epochs = trial.suggest_categorical("epochs", [20, 40])
    gamma = trial.suggest_categorical("scheduler_gamma", [1.0, 0.5])
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_categorical(
            "weight_decay", [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
        ),
        "dropout": trial.suggest_categorical("dropout", [0.0, 0.1, 0.3]),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "epochs": epochs,
        "scheduler_gamma": gamma,
        "scheduler_step_size": epochs if gamma == 1.0 else max(1, epochs // 2),
        "n_components": trial.suggest_categorical(
            "n_components", pca_candidates
        ),
    }


def apply_trial_params(
    settings: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    updated = copy.deepcopy(settings)
    for key, value in params.items():
        if key == "n_components":
            continue
        updated[key] = value
    return updated


def build_inner_evaluations(
    folds: list[tuple[set[str], set[str]]],
    augmented: pd.DataFrame,
    original: pd.DataFrame,
    inner_folds: int,
    seed: int,
) -> list[tuple[int, int, pd.DataFrame, pd.DataFrame]]:
    if inner_folds < 2:
        raise SystemExit("--inner-folds must be at least 2")
    evaluations: list[tuple[int, int, pd.DataFrame, pd.DataFrame]] = []
    for outer_index, (train_patients, _) in enumerate(folds):
        training_frame = augmented[
            augmented["patient_id"].isin(train_patients)
        ].copy()
        if training_frame.empty:
            raise RuntimeError(
                f"Outer split {outer_index:02d} has empty training data"
            )
        labels = training_frame["label"].to_numpy(dtype=np.int64)
        groups = training_frame["patient_id"].to_numpy(dtype=str)
        splitter = StratifiedGroupKFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=seed + outer_index,
        )
        for inner_index, (train_indices, validation_indices) in enumerate(
            splitter.split(training_frame, labels, groups=groups)
        ):
            inner_training = training_frame.iloc[train_indices].copy()
            validation_patients = set(groups[validation_indices])
            inner_validation = original[
                original["patient_id"].isin(validation_patients)
            ].copy()
            if inner_training.empty or inner_validation.empty:
                continue
            evaluations.append(
                (outer_index, inner_index, inner_training, inner_validation)
            )
    if not evaluations:
        raise SystemExit(
            "No inner train/validation evaluations could be constructed"
        )
    return evaluations


def eligible_components_for_evaluations(
    evaluations: list[tuple[int, int, pd.DataFrame, pd.DataFrame]],
    feature_dim: int,
) -> list[int]:
    minimum_training_patients = min(
        train_frame["patient_id"].nunique()
        for _, _, train_frame, _ in evaluations
    )
    return pca_clf.eligible_pca_components(
        minimum_training_patients,
        feature_dim=feature_dim,
    )


def write_best_config(
    path: Path,
    config: dict[str, Any],
    best_params: dict[str, Any],
) -> None:
    output = copy.deepcopy(config)
    training = dict(output.get("training") or {})
    training_params = {
        key: value
        for key, value in best_params.items()
        if key != "n_components"
    }
    training.update(training_params)
    output["training"] = training
    output["pca"] = {"n_components": int(best_params["n_components"])}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(output, handle, sort_keys=False, default_flow_style=False)


def main() -> None:
    args = parse_args()
    if args.trials < 1:
        raise SystemExit("--trials must be at least 1")
    settings = base.build_settings(args)
    device = torch.device(settings["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")

    config_path = Path(settings["config"])
    source_config = base.load_yaml_config(config_path)
    pinfo = base.load_pinfo(settings["pinfo"])
    original = base.attach_pinfo_labels(
        base.load_file(settings["orig_features"]), pinfo
    )
    augmented = base.attach_pinfo_labels(
        base.load_file(settings["aug_features"]), pinfo
    )
    feature_columns = base.validate_feature_frames(
        original, augmented, expected_features=int(settings["feature_dim"])
    )
    folds = base.select_folds(
        base.load_worc_splits(settings["splits"]), settings["max_splits"]
    )
    evaluations = build_inner_evaluations(
        folds,
        augmented,
        original,
        inner_folds=args.inner_folds,
        seed=int(settings["seed"]),
    )
    pca_candidates = eligible_components_for_evaluations(
        evaluations, feature_dim=int(settings["feature_dim"])
    )

    print(f"Dataset:        {settings['dataset']}")
    print(f"Backbone:       {settings['backbone']}")
    print(f"Device:         {device}")
    print(f"Outer folds:    {len(folds)}")
    print(f"Inner evals:    {len(evaluations)}")
    print(f"PCA candidates: {pca_candidates}")
    print(f"Trials:         {args.trials}")
    print(f"Config:         {config_path}")

    def objective(trial: optuna.Trial) -> float:
        params = suggest_hyperparams(trial, pca_candidates)
        trial_settings = apply_trial_params(settings, params)
        n_components = int(params["n_components"])
        aucs: list[float] = []
        for outer_index, inner_index, train_frame, val_frame in evaluations:
            try:
                auc, _, _, _ = pca_clf.run_split(
                    train_frame,
                    val_frame,
                    feature_columns,
                    trial_settings,
                    n_components=n_components,
                    fold_seed=int(settings["seed"])
                    + 1000 * outer_index
                    + inner_index,
                )
            except ValueError:
                continue
            if not np.isnan(auc):
                aucs.append(float(auc))
        if not aucs:
            return 0.5
        mean_auc = float(np.mean(aucs))
        trial.set_user_attr("n_eval_folds", len(aucs))
        trial.set_user_attr("std_auc", float(np.std(aucs)))
        return mean_auc

    sampler = optuna.samplers.TPESampler(seed=int(settings["seed"]))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.trials, n_jobs=args.n_jobs)

    best_params = dict(study.best_params)
    best_params["scheduler_step_size"] = (
        int(best_params["epochs"])
        if float(best_params["scheduler_gamma"]) == 1.0
        else max(1, int(best_params["epochs"]) // 2)
    )

    output_dir = settings["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = base.feature_stem(str(settings["backbone"]))
    prefix = f"{settings['dataset']}_{stem}_mlp_pca_hparam_sweep"
    trials_path = output_dir / f"{prefix}_trials.csv"
    best_config_path = output_dir / f"{prefix}_best.yaml"
    study.trials_dataframe().to_csv(trials_path, index=False)
    write_best_config(best_config_path, source_config, best_params)
    if args.update_config:
        write_best_config(config_path, source_config, best_params)

    print("\n--- Best inner-CV PCA hyperparameters ---")
    print(f"mean AUROC: {study.best_value:.4f}")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"Wrote {trials_path}")
    print(f"Wrote {best_config_path}")
    if args.update_config:
        print(f"Updated {config_path}")
    print(
        "Re-run PCA classification with:\n"
        f"  python experiments/nnFoundation/MLP_nn_classification_PCA.py "
        f"--dataset {settings['dataset']} --backbone {settings['backbone']} "
        f"--config {best_config_path}"
    )
    print(
        "Note: final PCA runs still select n_components per outer fold via "
        "inner CV; the swept n_components is recorded under pca.n_components."
    )


if __name__ == "__main__":
    main()
