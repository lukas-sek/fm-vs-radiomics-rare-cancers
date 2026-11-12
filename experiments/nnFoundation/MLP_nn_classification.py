"""Classify WORC outcomes from frozen nnFoundation lesion embeddings.

Training samples one cached embedding per unique lesion each epoch from the
augmented feature CSV (K-crop caching). Evaluation uses only the corresponding
held-out patients from the original feature CSV and aggregates lesion
probabilities per patient.
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from timm.layers import ClassifierHead as TimmClassifierHead
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, TensorDataset

DATASETS = ("Lipo", "Desmoid", "Liver", "GIST", "CRLM", "Melanoma")
BACKBONES = ("CNN", "ViT")
FEATURE_DIMS = {"CNN": 320, "ViT": 1056}
HIDDEN_LAYERS = ("none", "full", "half")
DECISION_THRESHOLD = 0.5
SUMMARY_METRICS = (
    ("AUC", "auc"),
    ("Accuracy", "accuracy"),
    ("BCA", "bca"),
    ("F1-score", "f1"),
    ("NPV", "npv"),
    ("Precision", "precision"),
    ("Sensitivity", "sensitivity"),
    ("Specificity", "specificity"),
    ("AP", "ap"),
)
# Backwards-compatible alias used by older PCA helpers/tests.
EXPECTED_FEATURES = FEATURE_DIMS["CNN"]
CONFIG_DIR = Path(__file__).resolve().parent


def default_config_path(backbone: str) -> Path:
    """Return the backbone-specific MLP training YAML."""
    if backbone == "CNN":
        return CONFIG_DIR / "mlp_CNN.yaml"
    if backbone == "ViT":
        return CONFIG_DIR / "mlp_ViT.yaml"
    raise ValueError(f"Unknown backbone {backbone!r}; choose from {BACKBONES}")


# Backwards-compatible alias (CNN defaults).
DEFAULT_CONFIG = CONFIG_DIR / "mlp_CNN.yaml"


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


def output_prefix(settings: dict[str, Any], model_name: str = "mlp") -> str:
    """<DATASET>_<stem>[_<tag>]_<model_name>, used for all CV output files."""
    parts = [str(settings["dataset"]), feature_stem(str(settings["backbone"]))]
    tag = str(settings.get("output_tag") or "")
    if tag:
        parts.append(tag)
    parts.append(model_name)
    return "_".join(parts)


def default_feature_paths(dataset: str, backbone: str) -> tuple[Path, Path]:
    stem = feature_stem(backbone)
    root = REPO_ROOT / "Results" / "nnFoundation" / "Features"
    return (
        root / f"{dataset}_{stem}_features.csv",
        root / f"{dataset}_{stem}_augmented_features.csv",
    )


class ClassificationHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 1,
        dropout: float = 0.0,
        init_std: float | None = None,
    ):
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        self.input_dim = input_dim
        self.fc = TimmClassifierHead(
            input_dim, num_classes, pool_type="", drop_rate=dropout
        )
        if init_std is not None:
            nn.init.normal_(self.fc.fc.weight, mean=0.0, std=init_std)
            nn.init.zeros_(self.fc.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x)
        if logits.ndim > 1 and logits.shape[-1] == 1:
            return logits.squeeze(-1)
        return logits


class MLP(ClassificationHead):
    """Binary classification head over frozen embeddings.

    With ``hidden_dim`` set, ``num_hidden_layers`` hidden layers of that width
    (each Linear -> ReLU -> Dropout) precede the linear classifier.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 1,
        dropout: float = 0.0,
        init_std: float | None = None,
        hidden_dim: int | None = None,
        num_hidden_layers: int = 1,
    ):
        if hidden_dim is not None and hidden_dim < 1:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_hidden_layers < 1:
            raise ValueError(
                f"num_hidden_layers must be positive, got {num_hidden_layers}"
            )
        super().__init__(
            input_dim=in_features if hidden_dim is None else hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            init_std=init_std,
        )
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = 0 if hidden_dim is None else num_hidden_layers
        layers: list[nn.Module] = []
        width = in_features
        for _ in range(self.num_hidden_layers):
            layers += [nn.Linear(width, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden_dim
        self.hidden = nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(self.hidden(x))


def hidden_dim_for(in_features: int, hidden_layer: str) -> int | None:
    """Map a ``HIDDEN_LAYERS`` mode to the hidden width for ``in_features``."""
    if hidden_layer == "none":
        return None
    if hidden_layer == "full":
        return in_features
    if hidden_layer == "half":
        return max(1, in_features // 2)
    raise ValueError(
        f"Unknown hidden_layer {hidden_layer!r}; choose from {HIDDEN_LAYERS}"
    )


def output_suffix(hidden_layer: str, num_hidden_layers: int = 1) -> str:
    if hidden_layer == "none":
        return "mlp"
    return f"mlp_h{num_hidden_layers}{hidden_layer}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dataset-generic MLP classification of nnFoundationCNN or "
            "nnFoundationViT embeddings over the original WORC CV splits."
        )
    )
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument(
        "--backbone",
        required=True,
        choices=BACKBONES,
        help="Feature family: CNN -> *_nncnn_*.csv, ViT -> *_nnvit_*.csv.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "YAML training config. Default depends on --backbone: "
            "CNN -> mlp_CNN.yaml, ViT -> mlp_ViT.yaml."
        ),
    )
    parser.add_argument("--orig-features", type=Path, default=None)
    parser.add_argument("--aug-features", type=Path, default=None)
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--pinfo", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--output-tag",
        default=None,
        help=(
            "Optional tag inserted into output names, e.g. b29 -> "
            "<DATASET>_nnvit_b29_mlp_cv_*.csv (default: none)."
        ),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    add_hidden_layer_argument(parser)
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


def add_hidden_layer_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--hidden-layer",
        choices=HIDDEN_LAYERS,
        default=None,
        help=(
            "Hidden layer before the classifier: none (linear head), "
            "full (width = input dim) or half (width = input dim / 2). "
            "Overrides training.hidden_layer in the config (default: none)."
        ),
    )
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        choices=(1, 2),
        default=None,
        help=(
            "Number of hidden layers of the --hidden-layer width. "
            "Overrides training.num_hidden_layers in the config (default: 1)."
        ),
    )


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (REPO_ROOT / candidate).resolve()
    )


def load_yaml_config(path: Path) -> dict[str, Any]:
    path = resolve_path(path)
    if not path.is_file():
        raise SystemExit(f"Config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise SystemExit(f"Config must contain a YAML mapping: {path}")
    return config


def build_settings(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset
    backbone = args.backbone
    config_path = (
        resolve_path(args.config)
        if getattr(args, "config", None) is not None
        else default_config_path(backbone)
    )
    config = load_yaml_config(config_path)
    training = config.get("training") or {}
    if not isinstance(training, dict):
        raise SystemExit("The config 'training' entry must be a mapping")

    def training_value(cli_value: Any, key: str, default: Any) -> Any:
        return (
            cli_value
            if cli_value is not None
            else training.get(key, default)
        )

    orig_default, aug_default = default_feature_paths(dataset, backbone)
    feature_root = REPO_ROOT / "Results" / "nnFoundation"
    settings = {
        "dataset": dataset,
        "backbone": backbone,
        "config": config_path,
        "feature_dim": FEATURE_DIMS[backbone],
        "orig_features": resolve_path(args.orig_features or orig_default),
        "aug_features": resolve_path(args.aug_features or aug_default),
        "splits": resolve_path(
            args.splits
            or REPO_ROOT
            / "crossvalidationsplits"
            / f"crossvalidationsplits_{dataset}.csv"
        ),
        "pinfo": resolve_path(
            args.pinfo or REPO_ROOT / "metadata" / f"pinfo_{dataset}.csv"
        ),
        "output_dir": resolve_path(args.output_dir or feature_root),
        "output_tag": getattr(args, "output_tag", None) or "",
        "epochs": int(training_value(args.epochs, "epochs", 100)),
        "batch_size": int(training_value(args.batch_size, "batch_size", 32)),
        "lr": float(training_value(args.lr, "lr", 0.0125)),
        "scheduler_step_size": int(training.get("scheduler_step_size", 30)),
        "scheduler_gamma": float(training.get("scheduler_gamma", 0.1)),
        "weight_decay": float(training.get("weight_decay", 0.0)),
        "dropout": float(training.get("dropout", 0.0)),
        "hidden_layer": str(
            training_value(
                getattr(args, "hidden_layer", None), "hidden_layer", "none"
            )
        ),
        "num_hidden_layers": int(
            training_value(
                getattr(args, "num_hidden_layers", None), "num_hidden_layers", 1
            )
        ),
        "seed": int(training_value(args.seed, "seed", 42)),
        "device": args.device,
        "max_splits": (
            None if args.max_splits is None else int(args.max_splits)
        ),
    }
    if settings["max_splits"] is not None and settings["max_splits"] < 1:
        raise SystemExit("--max-splits must be at least 1")
    if settings["epochs"] < 1:
        raise SystemExit("epochs must be at least 1")
    if settings["batch_size"] < 1:
        raise SystemExit("batch_size must be at least 1")
    if settings["lr"] <= 0:
        raise SystemExit("lr must be positive")
    if not 0.0 <= settings["dropout"] < 1.0:
        raise SystemExit("dropout must be in [0, 1)")
    if settings["hidden_layer"] not in HIDDEN_LAYERS:
        raise SystemExit(f"hidden_layer must be one of {HIDDEN_LAYERS}")
    if settings["num_hidden_layers"] not in (1, 2):
        raise SystemExit("num_hidden_layers must be 1 or 2")
    return settings


def load_file(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Feature file not found: {path}")
    return (
        pd.read_parquet(path)
        if path.suffix.lower() == ".parquet"
        else pd.read_csv(path)
    )


def normalize_pid(value: object) -> str:
    patient_id = str(value).strip()
    return patient_id[:-2] if patient_id.endswith(".0") else patient_id


def get_feature_cols(
    frame: pd.DataFrame, expected_features: int | None = None
) -> list[str]:
    feature_columns = [
        column
        for column in frame.columns
        if re.fullmatch(r"feature_\d+", str(column))
    ]
    feature_columns.sort(key=lambda column: int(column.split("_", 1)[1]))
    expected_count = (
        EXPECTED_FEATURES if expected_features is None else expected_features
    )
    expected = [f"feature_{index}" for index in range(expected_count)]
    if feature_columns != expected:
        raise ValueError(
            f"Expected exactly feature_0 through feature_{expected_count - 1}; "
            f"found {len(feature_columns)} feature columns"
        )
    return feature_columns


def load_pinfo(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise SystemExit(f"pinfo file not found: {path}")
    pinfo = pd.read_csv(path)
    required = {"Patient", "Diagnosis"}
    if not required.issubset(pinfo.columns):
        raise SystemExit(f"pinfo must contain {sorted(required)}: {path}")
    labels = {
        normalize_pid(row["Patient"]): int(row["Diagnosis"])
        for _, row in pinfo.iterrows()
    }
    if not set(labels.values()).issubset({0, 1}):
        raise SystemExit("MLP classification requires binary pinfo labels")
    return labels


def attach_pinfo_labels(
    frame: pd.DataFrame, pinfo: dict[str, int]
) -> pd.DataFrame:
    if "patient_id" not in frame.columns:
        raise SystemExit("Feature CSV must contain patient_id")
    output = frame.copy()
    output["patient_id"] = output["patient_id"].map(normalize_pid)
    missing = sorted(set(output["patient_id"]) - set(pinfo))
    if missing:
        raise SystemExit(
            f"Missing pinfo labels for {len(missing)} patients "
            f"(for example {missing[:5]})"
        )
    output["label"] = output["patient_id"].map(pinfo).astype(np.int64)
    return output


def validate_feature_frames(
    original: pd.DataFrame,
    augmented: pd.DataFrame,
    expected_features: int | None = None,
) -> list[str]:
    original_features = get_feature_cols(original, expected_features)
    augmented_features = get_feature_cols(augmented, expected_features)
    if original_features != augmented_features:
        raise ValueError(
            "Original and augmented CSVs have different feature columns"
        )
    for name, frame in (("original", original), ("augmented", augmented)):
        values = frame[original_features].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} feature CSV contains non-finite values")
    if "augmentation_id" in original.columns:
        augmentation_ids = pd.to_numeric(
            original["augmentation_id"], errors="coerce"
        )
        if augmentation_ids.isna().any() or (augmentation_ids != 0).any():
            raise ValueError(
                "Original feature CSV must contain only augmentation_id == 0"
            )
    return original_features


def load_worc_splits(path: Path) -> list[tuple[set[str], set[str]]]:
    if not path.is_file():
        raise SystemExit(f"Split file not found: {path}")
    split_frame = pd.read_csv(path)
    fold_ids = sorted(
        {
            int(match.group(1))
            for column in split_frame.columns
            if (match := re.fullmatch(r"(\d+)_train", str(column)))
        }
    )
    if not fold_ids:
        raise SystemExit(f"No '{{i}}_train' columns found in {path}")

    folds: list[tuple[set[str], set[str]]] = []
    for fold_id in fold_ids:
        test_column = f"{fold_id}_test"
        if test_column not in split_frame.columns:
            raise SystemExit(f"Missing {test_column} in {path}")
        train_patients = {
            normalize_pid(value)
            for value in split_frame[f"{fold_id}_train"].dropna()
            if str(value).strip()
        }
        test_patients = {
            normalize_pid(value)
            for value in split_frame[test_column].dropna()
            if str(value).strip()
        }
        overlap = train_patients & test_patients
        if overlap:
            raise ValueError(
                f"Fold {fold_id} has train/test overlap: {sorted(overlap)[:5]}"
            )
        folds.append((train_patients, test_patients))
    return folds


def select_folds(
    folds: list[tuple[set[str], set[str]]],
    max_splits: int | None,
) -> list[tuple[set[str], set[str]]]:
    """Return the first ``max_splits`` folds, or all folds when unset."""
    if max_splits is None:
        return folds
    if max_splits < 1:
        raise SystemExit("--max-splits must be at least 1")
    return folds[:max_splits]


def lesion_keys(frame: pd.DataFrame) -> np.ndarray:
    """Return a stable identity for each row's lesion / input."""
    if "case_id" in frame.columns:
        return frame["case_id"].map(lambda value: str(value).strip()).to_numpy()
    if "lesion_id" in frame.columns:
        return np.array(
            [
                f"{patient}\0{lesion}"
                for patient, lesion in zip(
                    frame["patient_id"].astype(str),
                    frame["lesion_id"].astype(str),
                )
            ],
            dtype=object,
        )
    return frame["patient_id"].astype(str).to_numpy()


def group_cached_crops(
    features: np.ndarray,
    frame: pd.DataFrame,
) -> list[tuple[np.ndarray, float, str]]:
    """Group K cached embeddings, label, and patient id per unique lesion."""
    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape {features.shape}")
    if features.shape[0] != len(frame):
        raise ValueError("features and frame must have the same number of rows")
    if "patient_id" not in frame.columns or "label" not in frame.columns:
        raise ValueError("frame must contain patient_id and label")
    if len(frame) < 1:
        raise ValueError("Cached crop grouping requires at least one row")

    keys = lesion_keys(frame)
    labels = frame["label"].to_numpy()
    patients = frame["patient_id"].to_numpy(dtype=str)
    _, first_index, inverse = np.unique(keys, return_index=True, return_inverse=True)
    groups: list[tuple[np.ndarray, float, str]] = []
    for group_id in np.argsort(first_index):
        indices = np.flatnonzero(inverse == group_id)
        if indices.size < 1:
            raise ValueError("Cached crop group is empty")
        group_labels = labels[indices]
        group_patients = patients[indices]
        if np.unique(group_labels).size != 1:
            raise ValueError(
                f"Inconsistent labels for lesion {keys[indices[0]]!r}"
            )
        if np.unique(group_patients).size != 1:
            raise ValueError(
                f"Inconsistent patient_id for lesion {keys[indices[0]]!r}"
            )
        groups.append(
            (
                np.ascontiguousarray(features[indices], dtype=np.float32),
                float(group_labels[0]),
                str(group_patients[0]),
            )
        )
    return groups


class CachedCropDataset(Dataset):
    """One training example per lesion; each access samples one of K crops."""

    def __init__(
        self,
        groups: list[tuple[np.ndarray, float, str]],
        generator: torch.Generator | None = None,
    ):
        if not groups:
            raise ValueError("CachedCropDataset requires at least one lesion")
        self.crops = [
            torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
            for features, _, _ in groups
        ]
        self.labels = torch.tensor(
            [label for _, label, _ in groups], dtype=torch.float32
        )
        patients = np.array([patient for _, _, patient in groups], dtype=str)
        self.weights = torch.from_numpy(patient_balanced_weights(patients))
        self.generator = generator

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        crops = self.crops[index]
        if crops.shape[0] == 1:
            crop_index = 0
        else:
            crop_index = int(
                torch.randint(
                    crops.shape[0],
                    (1,),
                    generator=self.generator,
                ).item()
            )
        return crops[crop_index], self.labels[index], self.weights[index]


def make_cached_crop_loader(
    groups: list[tuple[np.ndarray, float, str]],
    batch_size: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        CachedCropDataset(groups, generator=generator),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )


def patient_balanced_weights(patient_ids: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(
        patient_ids, return_inverse=True, return_counts=True
    )
    weights = 1.0 / counts[inverse]
    return (weights / weights.mean()).astype(np.float32)


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator) / float(denominator)


def binary_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, float]:
    """Patient-level ranking and 0.5-threshold metrics used in WORC reporting."""
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    predicted = (scores >= threshold).astype(np.int64)
    true_negatives, false_positives, false_negatives, true_positives = (
        confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    )
    sensitivity = _safe_rate(true_positives, true_positives + false_negatives)
    specificity = _safe_rate(true_negatives, true_negatives + false_positives)
    precision = _safe_rate(true_positives, true_positives + false_positives)
    npv = _safe_rate(true_negatives, true_negatives + false_negatives)
    if np.isnan(precision) or np.isnan(sensitivity):
        f1 = float("nan")
    else:
        f1 = _safe_rate(2.0 * precision * sensitivity, precision + sensitivity)
    if len(np.unique(labels)) < 2:
        auc = float("nan")
        ap = float("nan")
    else:
        auc = float(roc_auc_score(labels, scores))
        ap = float(average_precision_score(labels, scores))
    return {
        "auc": auc,
        "auroc": auc,
        "ap": ap,
        "accuracy": _safe_rate(
            true_positives + true_negatives, labels.size
        ),
        "bca": (
            float("nan")
            if np.isnan(sensitivity) or np.isnan(specificity)
            else 0.5 * (sensitivity + specificity)
        ),
        "f1": f1,
        "npv": npv,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def metrics_from_patient_predictions(predictions: pd.DataFrame) -> dict[str, float]:
    return binary_classification_metrics(
        predictions["label"].to_numpy(dtype=np.int64),
        predictions["probability"].to_numpy(dtype=np.float64),
    )


def aggregate_patient_predictions(
    patient_ids: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    predictions = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "label": labels.astype(np.int64),
            "probability": probabilities.astype(np.float64),
        }
    )
    inconsistent = predictions.groupby("patient_id")["label"].nunique()
    if (inconsistent > 1).any():
        patients = inconsistent[inconsistent > 1].index.tolist()
        raise ValueError(f"Inconsistent labels for patients {patients[:5]}")
    return (
        predictions.groupby("patient_id", as_index=False)
        .agg(label=("label", "first"), probability=("probability", "mean"))
        .sort_values("patient_id")
        .reset_index(drop=True)
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    weighted_loss = 0.0
    total_weight = 0.0
    for features, labels, weights in loader:
        features = features.to(device)
        labels = labels.to(device)
        weights = weights.to(device)
        optimizer.zero_grad()
        losses = criterion(model(features), labels)
        loss = (losses * weights).sum() / weights.sum()
        loss.backward()
        optimizer.step()
        weighted_loss += float((losses.detach() * weights).sum().item())
        total_weight += float(weights.sum().item())
    return weighted_loss / total_weight


@torch.inference_mode()
def predict_probabilities(
    model: nn.Module,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)),
        batch_size=batch_size,
        shuffle=False,
    )
    probabilities: list[np.ndarray] = []
    for (batch,) in loader:
        logits = model(batch.to(device))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities).astype(np.float64, copy=False)


def run_split(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    settings: dict[str, Any],
    fold_seed: int,
) -> tuple[float, float, pd.DataFrame, dict[str, float]]:
    seed_everything(fold_seed)
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(
        train_frame[feature_columns].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    test_features = scaler.transform(
        test_frame[feature_columns].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    test_labels = test_frame["label"].to_numpy(dtype=np.int64)
    test_patients = test_frame["patient_id"].to_numpy(dtype=str)
    loader = make_cached_crop_loader(
        group_cached_crops(scaled_train, train_frame),
        batch_size=int(settings["batch_size"]),
        seed=fold_seed,
    )
    device = torch.device(settings["device"])
    in_features = int(settings["feature_dim"])
    model = MLP(
        in_features=in_features,
        dropout=float(settings.get("dropout", 0.0)),
        hidden_dim=hidden_dim_for(
            in_features, settings.get("hidden_layer", "none")
        ),
        num_hidden_layers=int(settings.get("num_hidden_layers", 1)),
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
        train_epoch(model, loader, optimizer, criterion, device)
        scheduler.step()

    probabilities = predict_probabilities(
        model,
        test_features,
        settings["batch_size"],
        device,
    )
    patient_predictions = aggregate_patient_predictions(
        test_patients,
        test_labels,
        probabilities,
    )
    metrics = metrics_from_patient_predictions(patient_predictions)
    return metrics["auc"], metrics["ap"], patient_predictions, metrics


def metric_summary(per_split: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for display_name, column in SUMMARY_METRICS:
        if column not in per_split.columns:
            continue
        values = per_split[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            lower, upper = np.percentile(finite, [2.5, 97.5])
        else:
            lower, upper = float("nan"), float("nan")
        rows.append(
            {
                "metric": display_name,
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "median": float(np.nanmedian(values)),
                "ci_95_lower": float(lower),
                "ci_95_upper": float(upper),
                "n_splits": len(per_split),
            }
        )
    return pd.DataFrame(rows)


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
        f"weight_decay={settings['weight_decay']} "
        f"dropout={settings['dropout']} "
        f"hidden_layer={settings['hidden_layer']} "
        f"num_hidden_layers={settings['num_hidden_layers']} "
        f"seed={settings['seed']}"
    )

    pinfo = load_pinfo(settings["pinfo"])
    original = attach_pinfo_labels(
        load_file(settings["orig_features"]), pinfo
    )
    augmented = attach_pinfo_labels(
        load_file(settings["aug_features"]), pinfo
    )
    feature_columns = validate_feature_frames(
        original, augmented, expected_features=int(settings["feature_dim"])
    )
    folds = select_folds(load_worc_splits(settings["splits"]), settings["max_splits"])
    print(f"Running splits: {len(folds)}")

    per_split_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for split_index, (train_patients, test_patients) in enumerate(folds):
        train_frame = augmented[
            augmented["patient_id"].isin(train_patients)
        ].copy()
        test_frame = original[
            original["patient_id"].isin(test_patients)
        ].copy()
        if train_frame.empty or test_frame.empty:
            raise RuntimeError(
                f"Split {split_index:02d} has empty training or test data"
            )
        auc, ap, predictions, metrics = run_split(
            train_frame,
            test_frame,
            feature_columns,
            settings,
            fold_seed=settings["seed"] + split_index,
        )
        predictions.insert(0, "split", split_index)
        prediction_frames.append(predictions)
        per_split_rows.append(
            {
                "split": split_index,
                "n_train_rows": len(train_frame),
                "n_test_rows": len(test_frame),
                "n_train_patients": train_frame["patient_id"].nunique(),
                "n_test_patients": test_frame["patient_id"].nunique(),
                "auroc": auc,
                **metrics,
            }
        )
        print(
            f"Split {split_index:02d}: AUROC={auc:.4f} AP={ap:.4f} "
            f"BCA={metrics['bca']:.4f} "
            f"({train_frame['patient_id'].nunique()} train / "
            f"{test_frame['patient_id'].nunique()} test patients)"
        )

    per_split = pd.DataFrame(per_split_rows)
    summary = metric_summary(per_split)
    patient_predictions = pd.concat(prediction_frames, ignore_index=True)
    output_dir = settings["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix(
        settings,
        output_suffix(settings["hidden_layer"], settings["num_hidden_layers"]),
    )
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
