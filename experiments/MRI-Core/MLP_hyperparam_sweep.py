"""Optuna sweep for MRI-CORE Slideaggregation MLP training hyperparameters.

Each trial is scored by mean patient-level AUROC on inner validation folds
built only from WORC training patients (original rows). Outer test folds are
never used, so the selected values can be written back to ``mlp.yaml``.

Example:
    python experiments/MRI-Core/MLP_hyperparam_sweep.py \\
      --dataset Liver --trials 30 --update-config
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedGroupKFold

MRI_CORE_DIR = Path(__file__).resolve().parent
if str(MRI_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(MRI_CORE_DIR))

import MLP_classification as base  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

INNER_FOLDS = 3
DEFAULT_TRIALS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Nested Optuna sweep for MRI-CORE Slideaggregation MLP hyperparameters. "
            "Scores inner-fold patient AUROC; does not use WORC test folds."
        )
    )
    parser.add_argument("--dataset", required=True, choices=base.DATASETS)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"YAML to read (and optionally update). Default: {base.DEFAULT_CONFIG.name}.",
    )
    parser.add_argument("--orig-features", type=Path, default=None)
    parser.add_argument(
        "--aug-features",
        type=Path,
        default=None,
        help="Cached K-augmentation CSV for training (implies --use-cached-augs).",
    )
    parser.add_argument(
        "--use-cached-augs",
        action="store_true",
        help=(
            "Sweep with COMET-style K-cached augmentations. Requires "
            "<dataset>_slideagg_augmented_features.csv (or --aug-features)."
        ),
    )
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--pinfo", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    base.add_hidden_layer_argument(parser)
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


def suggest_hyperparams(trial: optuna.Trial) -> dict[str, Any]:
    epochs = trial.suggest_categorical("epochs", [20, 40])
    gamma = trial.suggest_categorical("scheduler_gamma", [1.0, 0.5])
    hidden_layer = trial.suggest_categorical("hidden_layer", list(base.HIDDEN_LAYERS))
    params: dict[str, Any] = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_categorical(
            "weight_decay", [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
        ),
        "dropout": trial.suggest_categorical("dropout", [0.0, 0.1, 0.3]),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "epochs": epochs,
        "scheduler_gamma": gamma,
        "scheduler_step_size": epochs if gamma == 1.0 else max(1, epochs // 2),
        "hidden_layer": hidden_layer,
    }
    if hidden_layer == "none":
        params["num_hidden_layers"] = 1
    else:
        params["num_hidden_layers"] = trial.suggest_categorical(
            "num_hidden_layers", [1, 2]
        )
    return params


def apply_trial_params(
    settings: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    updated = copy.deepcopy(settings)
    updated.update(params)
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
            raise RuntimeError(f"Outer split {outer_index:02d} has empty training data")
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
        raise SystemExit("No inner train/validation evaluations could be constructed")
    return evaluations


def write_training_config(
    path: Path,
    config: dict[str, Any],
    best_params: dict[str, Any],
) -> None:
    output = copy.deepcopy(config)
    training = dict(output.get("training") or {})
    training.update(best_params)
    # Keep seed if present in the source config / leave unset otherwise.
    if "seed" in training or "seed" in (config.get("training") or {}):
        training.setdefault("seed", (config.get("training") or {}).get("seed", 42))
    output["training"] = training
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

    print(f"Dataset:        {settings['dataset']}")
    print(f"Features:       {settings['backbone']} ({settings['feature_dim']}-d)")
    print(f"Device:         {device}")
    print(
        "Train mode:     "
        + (
            "K-cached augs (COMET-style)"
            if settings.get("use_cached_augs")
            else "no-aug linear probing"
        )
    )
    if settings.get("use_cached_augs"):
        max_k, mean_k = base.require_cached_augmentations(original, augmented)
        print(f"Cached augs:    max K={max_k}, mean K={mean_k:.2f} per lesion")
    print(f"Outer folds:    {len(folds)}")
    print(f"Inner evals:    {len(evaluations)}")
    print(f"Trials:         {args.trials}")
    print(f"Config:         {config_path}")

    def objective(trial: optuna.Trial) -> float:
        params = suggest_hyperparams(trial)
        trial_settings = apply_trial_params(settings, params)
        aucs: list[float] = []
        for outer_index, inner_index, train_frame, val_frame in evaluations:
            result = base.run_split(
                train_frame,
                val_frame,
                feature_columns,
                trial_settings,
                fold_seed=int(settings["seed"])
                + 1000 * outer_index
                + inner_index,
            )
            auc = result[0]
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
    if best_params.get("hidden_layer") == "none":
        best_params["num_hidden_layers"] = 1
    elif "num_hidden_layers" not in best_params:
        best_params["num_hidden_layers"] = 1

    output_dir = settings["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{settings['dataset']}_{base.FEATURE_STEM}_mlp_hparam_sweep"
    trials_path = output_dir / f"{prefix}_trials.csv"
    best_config_path = output_dir / f"{prefix}_best.yaml"
    study.trials_dataframe().to_csv(trials_path, index=False)
    write_training_config(best_config_path, source_config, best_params)
    if args.update_config:
        write_training_config(config_path, source_config, best_params)

    print("\n--- Best inner-CV hyperparameters ---")
    print(f"mean AUROC: {study.best_value:.4f}")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"Wrote {trials_path}")
    print(f"Wrote {best_config_path}")
    if args.update_config:
        print(f"Updated {config_path}")
    print(
        "Re-run classification with:\n"
        f"  python experiments/MRI-Core/MLP_classification.py "
        f"--dataset {settings['dataset']} --config {best_config_path}"
    )


if __name__ == "__main__":
    main()
