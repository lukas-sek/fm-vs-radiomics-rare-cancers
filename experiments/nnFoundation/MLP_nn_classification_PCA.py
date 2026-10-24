"""Classify WORC outcomes from PCA-reduced nnFoundation embeddings.

PCA dimensionality is selected independently in every outer WORC fold using
patient-grouped inner cross-validation. Training samples one cached embedding
per unique lesion each epoch (K-crop caching). Inner validation and outer
testing use original feature rows only.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR

import MLP_nn_classification as base

PCA_COMPONENTS = (5, 10, 20, 40, 60)
INNER_FOLDS = 5
ClassificationHead = base.ClassificationHead
MLP = base.MLP
aggregate_patient_predictions = base.aggregate_patient_predictions
validate_feature_frames = base.validate_feature_frames

_PCA_CONFIG_DIR = (
    base.REPO_ROOT / "Results" / "nnFoundation"
)
CNN_PCA_DEFAULT_CONFIG = (
    _PCA_CONFIG_DIR / "Liver_nncnn_mlp_pca_hparam_sweep_best.yaml"
)
VIT_PCA_DEFAULT_CONFIG = (
    _PCA_CONFIG_DIR / "Liver_nnvit_mlp_pca_hparam_sweep_best.yaml"
)


def default_config_path(backbone: str) -> Path:
    """Return the PCA-specific MLP training YAML for this backbone."""
    if backbone == "CNN":
        return CNN_PCA_DEFAULT_CONFIG
    if backbone == "ViT":
        return VIT_PCA_DEFAULT_CONFIG
    raise ValueError(f"Unknown backbone {backbone!r}; choose from {base.BACKBONES}")


def build_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Like base.build_settings, but defaults to the Liver PCA sweep YAMLs."""
    if getattr(args, "config", None) is None:
        args.config = default_config_path(args.backbone)
    return base.build_settings(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dataset-generic MLP classification of PCA-reduced "
            "nnFoundationCNN or nnFoundationViT embeddings over WORC CV splits."
        )
    )
    parser.add_argument("--dataset", required=True, choices=base.DATASETS)
    parser.add_argument(
        "--backbone",
        required=True,
        choices=base.BACKBONES,
        help="Feature family: CNN -> *_nncnn_*.csv, ViT -> *_nnvit_*.csv.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "YAML training config. Default depends on --backbone: "
            "CNN -> Liver_nncnn_mlp_pca_hparam_sweep_best.yaml, "
            "ViT -> Liver_nnvit_mlp_pca_hparam_sweep_best.yaml."
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
        help="Torch device (default: cuda when available, otherwise cpu).",
    )
    parser.add_argument(
        "--max-splits",
        type=int,
        default=None,
        help="Run only the first N WORC CV splits (default: all).",
    )
    return parser.parse_args()


def patient_mean_features(
    features: np.ndarray, patient_ids: np.ndarray
) -> np.ndarray:
    """Average rows so every patient contributes once when fitting PCA."""
    unique_patients, inverse = np.unique(patient_ids, return_inverse=True)
    means = np.empty(
        (len(unique_patients), features.shape[1]), dtype=np.float64
    )
    for patient_index in range(len(unique_patients)):
        means[patient_index] = features[inverse == patient_index].mean(axis=0)
    return means


def eligible_pca_components(
    minimum_training_patients: int,
    feature_dim: int,
    candidates: tuple[int, ...] = PCA_COMPONENTS,
) -> list[int]:
    eligible = [
        component_count
        for component_count in candidates
        if component_count < minimum_training_patients
        and component_count <= feature_dim
    ]
    if not eligible:
        raise ValueError(
            "No PCA candidate is smaller than the minimum inner-training "
            f"patient count ({minimum_training_patients})"
        )
    return eligible


def fit_pca_projection(
    training_features: np.ndarray,
    training_patients: np.ndarray,
    n_components: int,
) -> tuple[StandardScaler, PCA, float]:
    patient_means = patient_mean_features(
        training_features, training_patients
    )
    if n_components >= len(patient_means):
        raise ValueError(
            f"PCA components ({n_components}) must be smaller than the number "
            f"of training patients ({len(patient_means)})"
        )
    scaler = StandardScaler()
    standardized_means = scaler.fit_transform(patient_means)
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(standardized_means)
    explained_variance = float(pca.explained_variance_ratio_.sum())
    return scaler, pca, explained_variance


def transform_features(
    features: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
) -> np.ndarray:
    return pca.transform(scaler.transform(features)).astype(np.float32)


def run_split(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    settings: dict[str, Any],
    n_components: int,
    fold_seed: int,
) -> tuple[float, float, pd.DataFrame, float]:
    base.seed_everything(fold_seed)
    raw_train = train_frame[feature_columns].to_numpy(dtype=np.float32)
    raw_test = test_frame[feature_columns].to_numpy(dtype=np.float32)
    train_patients = train_frame["patient_id"].to_numpy(dtype=str)
    test_patients = test_frame["patient_id"].to_numpy(dtype=str)

    scaler, pca, explained_variance = fit_pca_projection(
        raw_train,
        train_patients,
        n_components,
    )
    train_features = transform_features(raw_train, scaler, pca)
    test_features = transform_features(raw_test, scaler, pca)
    test_labels = test_frame["label"].to_numpy(dtype=np.int64)
    loader = base.make_cached_crop_loader(
        base.group_cached_crops(train_features, train_frame),
        batch_size=int(settings["batch_size"]),
        seed=fold_seed,
    )
    device = torch.device(settings["device"])
    model = MLP(
        n_components,
        dropout=float(settings.get("dropout", 0.0)),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings["lr"],
        weight_decay=settings["weight_decay"],
    )
    scheduler = StepLR(
        optimizer,
        step_size=settings["scheduler_step_size"],
        gamma=settings["scheduler_gamma"],
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    for _ in range(settings["epochs"]):
        base.train_epoch(model, loader, optimizer, criterion, device)
        scheduler.step()

    probabilities = base.predict_probabilities(
        model,
        test_features,
        settings["batch_size"],
        device,
    )
    patient_predictions = base.aggregate_patient_predictions(
        test_patients,
        test_labels,
        probabilities,
    )
    y_true = patient_predictions["label"].to_numpy(dtype=np.int64)
    y_score = patient_predictions["probability"].to_numpy(dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        auc, ap = np.nan, np.nan
    else:
        auc = float(roc_auc_score(y_true, y_score))
        ap = float(average_precision_score(y_true, y_score))
    return auc, ap, patient_predictions, explained_variance


def select_pca_components(
    training_frame: pd.DataFrame,
    original_frame: pd.DataFrame,
    feature_columns: list[str],
    settings: dict[str, Any],
    seed: int,
) -> tuple[int, float]:
    """Select PCA dimensionality without exposing outer-test patients."""
    labels = training_frame["label"].to_numpy(dtype=np.int64)
    groups = training_frame["patient_id"].to_numpy(dtype=str)
    splitter = StratifiedGroupKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=seed,
    )
    inner_splits = list(
        splitter.split(training_frame, labels, groups=groups)
    )
    minimum_training_patients = min(
        len(np.unique(groups[train_indices]))
        for train_indices, _ in inner_splits
    )
    candidates = eligible_pca_components(
        minimum_training_patients,
        feature_dim=int(settings["feature_dim"]),
    )
    scores: dict[int, list[float]] = {
        component_count: [] for component_count in candidates
    }

    for inner_index, (train_indices, validation_indices) in enumerate(
        inner_splits
    ):
        inner_training = training_frame.iloc[train_indices]
        validation_patients = set(groups[validation_indices])
        inner_validation = original_frame[
            original_frame["patient_id"].isin(validation_patients)
        ].copy()
        if inner_validation.empty:
            raise ValueError(
                f"Inner fold {inner_index} has no original validation rows"
            )

        for component_count in candidates:
            auc, _, _, _ = run_split(
                inner_training,
                inner_validation,
                feature_columns,
                settings,
                n_components=component_count,
                fold_seed=seed + inner_index,
            )
            if not np.isnan(auc):
                scores[component_count].append(auc)

    mean_scores = {
        component_count: (
            float(np.mean(component_scores))
            if component_scores
            else float("-inf")
        )
        for component_count, component_scores in scores.items()
    }
    best_components = max(
        candidates,
        key=lambda component_count: (
            mean_scores[component_count],
            -component_count,
        ),
    )
    return best_components, mean_scores[best_components]


def main() -> None:
    args = parse_args()
    settings = build_settings(args)
    device = torch.device(settings["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")

    print(f"Dataset:        {settings['dataset']}")
    print(f"Backbone:       {settings['backbone']} ({settings['feature_dim']}-d)")
    print(f"Config:         {settings['config']}")
    print(f"Device:         {device}")
    print(f"Original:       {settings['orig_features']}")
    print(f"Augmented:      {settings['aug_features']}")
    print(f"Splits:         {settings['splits']}")
    print(f"Labels:         {settings['pinfo']}")
    print(f"Output:         {settings['output_dir']}")
    print(
        f"Training: epochs={settings['epochs']} "
        f"batch_size={settings['batch_size']} lr={settings['lr']} "
        f"step={settings['scheduler_step_size']} "
        f"gamma={settings['scheduler_gamma']} "
        f"weight_decay={settings['weight_decay']} seed={settings['seed']}"
    )

    labels = base.load_pinfo(settings["pinfo"])
    original = base.attach_pinfo_labels(
        base.load_file(settings["orig_features"]), labels
    )
    augmented = base.attach_pinfo_labels(
        base.load_file(settings["aug_features"]), labels
    )
    feature_columns = base.validate_feature_frames(
        original, augmented, expected_features=int(settings["feature_dim"])
    )
    folds = base.select_folds(
        base.load_worc_splits(settings["splits"]), settings["max_splits"]
    )
    print(f"Running splits: {len(folds)}")

    per_split_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for split_index, (train_patients, test_patients) in enumerate(folds):
        training_frame = augmented[
            augmented["patient_id"].isin(train_patients)
        ].copy()
        test_frame = original[
            original["patient_id"].isin(test_patients)
        ].copy()
        if training_frame.empty or test_frame.empty:
            raise RuntimeError(
                f"Split {split_index:02d} has empty training or test data"
            )

        best_components, inner_auc = select_pca_components(
            training_frame,
            original,
            feature_columns,
            settings,
            seed=settings["seed"] + split_index,
        )
        auc, ap, predictions, explained_variance = run_split(
            training_frame,
            test_frame,
            feature_columns,
            settings,
            n_components=best_components,
            fold_seed=settings["seed"] + split_index,
        )
        predictions.insert(0, "split", split_index)
        prediction_frames.append(predictions)
        metrics = base.metrics_from_patient_predictions(predictions)
        per_split_rows.append(
            {
                "split": split_index,
                "n_train_rows": len(training_frame),
                "n_test_rows": len(test_frame),
                "n_train_patients": training_frame["patient_id"].nunique(),
                "n_test_patients": test_frame["patient_id"].nunique(),
                "auroc": auc,
                **metrics,
                "best_n_components": best_components,
                "explained_variance": explained_variance,
                "inner_cv_auroc": inner_auc,
            }
        )
        print(
            f"Split {split_index:02d}: AUROC={auc:.4f} AP={ap:.4f} "
            f"BCA={metrics['bca']:.4f} PCA={best_components} "
            f"({explained_variance:.1%} variance) "
            f"inner_AUROC={inner_auc:.4f}"
        )

    per_split = pd.DataFrame(per_split_rows)
    summary = base.metric_summary(per_split)
    patient_predictions = pd.concat(prediction_frames, ignore_index=True)
    output_dir = settings["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = base.feature_stem(str(settings["backbone"]))
    prefix = f"{settings['dataset']}_{stem}_mlp_pca"
    per_split_path = output_dir / f"{prefix}_cv_per_split.csv"
    summary_path = output_dir / f"{prefix}_cv_summary.csv"
    predictions_path = output_dir / f"{prefix}_cv_patient_predictions.csv"
    per_split.to_csv(per_split_path, index=False)
    summary.to_csv(summary_path, index=False)
    patient_predictions.to_csv(predictions_path, index=False)

    print("\n--- Summary across splits ---")
    for _, row in summary.iterrows():
        print(
            f"{row['metric']} 95%: {row['mean']:.4f} "
            f"({row['ci_95_lower']:.4f}-{row['ci_95_upper']:.4f}); "
            f"mean ± SD {row['mean']:.4f} ± {row['std']:.4f}"
        )
    print(f"Wrote {per_split_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {predictions_path}")


if __name__ == "__main__":
    main()
