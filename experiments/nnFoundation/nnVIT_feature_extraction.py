"""Extract mean-pooled nnFoundationViT features from WORC lesion patches."""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.primus import Primus
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("Lipo", "Desmoid", "Liver", "GIST", "CRLM", "Melanoma")
PATCH_MODES = ("original", "augmented")
PATCH_SIZE = (192, 192, 192)
REQUIRED_COLUMNS = ("dataset", "patient_id", "lesion_id", "label", "image_path")
METADATA_COLUMNS = (
    "dataset",
    "patient_id",
    "lesion_id",
    "label",
    "image_path",
    "augmentation_id",
    "augmentation_type",
)
DEFAULT_REPO_ID = "MIC-DKFZ/nnFoundationViT"
DEFAULT_FILENAME = "checkpoint_final.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one mean-pooled nnFoundationViT embedding per 192^3 WORC "
            "lesion patch."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS,
        help="WORC dataset to process.",
    )
    parser.add_argument(
        "--patches",
        required=True,
        choices=PATCH_MODES,
        help="Use original patches or augmented variants (augmentation_id > 0).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Override the default metadata manifest.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Override the default feature CSV under ~/experiments/Results.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda when available, otherwise cpu).",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--checkpoint-filename", default=DEFAULT_FILENAME)
    return parser.parse_args()


def resolve_cli_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def default_manifest(dataset: str, patches: str) -> Path:
    infix = "_augmented" if patches == "augmented" else ""
    return REPO_ROOT / "metadata" / f"worc_lesion_patches{infix}_{dataset}.csv"


def default_output_csv(dataset: str, patches: str) -> Path:
    return (
        Path.home()
        / "experiments"
        / "Results"
        / "nnFoundation"
        / f"{dataset}_nnvit_{patches}_features.csv"
    )


def load_manifest(path: Path, dataset: str, patches: str) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Manifest not found: {path}")

    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise SystemExit(f"Manifest is missing required columns {missing}: {path}")
    if frame.empty:
        raise SystemExit(f"Manifest has no rows: {path}")

    frame = frame.loc[frame["dataset"].astype(str) == dataset].copy()
    if frame.empty:
        raise SystemExit(f"Manifest has no rows for dataset {dataset}: {path}")

    if patches == "augmented":
        if "augmentation_id" not in frame.columns:
            raise SystemExit(
                "Augmented manifest must contain an 'augmentation_id' column"
            )
        augmentation_ids = pd.to_numeric(frame["augmentation_id"], errors="coerce")
        if augmentation_ids.isna().any():
            raise SystemExit("Manifest contains invalid augmentation_id values")
        frame = frame.loc[augmentation_ids > 0].copy()
        if frame.empty:
            raise SystemExit(
                f"Manifest has no augmented rows (augmentation_id > 0): {path}"
            )

    return frame.reset_index(drop=True)


def resolve_image_path(raw_path: object) -> Path:
    path = Path(str(raw_path))
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _architecture_kwargs(checkpoint: dict[str, Any]) -> dict[str, Any]:
    try:
        return checkpoint["nnssl_adaptation_plan"]["architecture_plans"][
            "arch_kwargs"
        ]
    except KeyError as error:
        raise RuntimeError(
            "Checkpoint does not contain nnssl_adaptation_plan."
            "architecture_plans.arch_kwargs"
        ) from error


def _matching_encoder_weights(
    model: nn.Module, state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    model_state = model.state_dict()
    prefixes = ("down_projection.", "eva.")
    matching = {
        key: value
        for key, value in state_dict.items()
        if key.startswith(prefixes)
        and key in model_state
        and model_state[key].shape == value.shape
    }
    for prefix in prefixes:
        if not any(key.startswith(prefix) for key in matching):
            raise RuntimeError(
                f"No compatible checkpoint weights found for '{prefix[:-1]}'"
            )
    return matching


def load_nnfoundation_vit(
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_FILENAME,
    device: str | torch.device = "cuda",
) -> Primus:
    checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    arch_kwargs = _architecture_kwargs(checkpoint)

    model = Primus(
        input_channels=int(arch_kwargs["input_channels"]),
        embed_dim=int(arch_kwargs["embed_dim"]),
        patch_embed_size=tuple(arch_kwargs["patch_embed_size"]),
        num_classes=int(arch_kwargs["output_channels"]),
        eva_depth=int(arch_kwargs["encoder_eva_depth"]),
        eva_numheads=int(arch_kwargs["encoder_eva_numheads"]),
        input_shape=tuple(arch_kwargs["input_shape"]),
        drop_path_rate=0.0,
        patch_drop_rate=0.0,
        attn_drop_rate=float(arch_kwargs.get("attn_drop_rate", 0.0)),
        init_values=arch_kwargs.get("init_values"),
        scale_attn_inner=bool(arch_kwargs.get("scale_attn_inner", False)),
    )

    compatible_weights = _matching_encoder_weights(
        model, checkpoint["network_weights"]
    )
    load_result = model.load_state_dict(compatible_weights, strict=False)
    loaded_eva = sum(key.startswith("eva.") for key in compatible_weights)
    loaded_stem = sum(
        key.startswith("down_projection.") for key in compatible_weights
    )
    print(
        f"Loaded {loaded_eva} EVA and {loaded_stem} patch-embedding tensors "
        f"from {checkpoint_path}"
    )
    if load_result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected compatible checkpoint keys: {load_result.unexpected_keys[:5]}"
        )

    del checkpoint, compatible_weights
    gc.collect()

    model.requires_grad_(False)
    model.eval()
    return model.to(device)


def load_patch(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Patch not found: {path}")
    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3 or image.GetSize() != PATCH_SIZE:
        raise ValueError(
            f"{path}: expected a 3D {PATCH_SIZE} patch, got {image.GetSize()}"
        )
    values = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: patch contains non-finite values")

    # Patch construction already performed the required per-volume Z-score
    # normalization. Only batch and channel dimensions are added here.
    return torch.from_numpy(values.copy()).unsqueeze(0).unsqueeze(0)


@torch.inference_mode()
def extract_features(model: Primus, volume: torch.Tensor) -> torch.Tensor:
    embedded = model.down_projection(volume)
    tokens = embedded.flatten(2).transpose(1, 2)
    tokens, _ = model.eva(tokens)
    if tokens.ndim != 3:
        raise RuntimeError(f"Expected (B, tokens, features), got {tokens.shape}")
    return tokens.mean(dim=1)


def metadata_for_row(row: pd.Series) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for column in METADATA_COLUMNS:
        if column == "image_path":
            metadata[column] = str(row["image_path"])
        elif column in row.index and not pd.isna(row[column]):
            metadata[column] = row[column]
    return metadata


def extract_manifest_features(
    model: Primus,
    frame: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    feature_count: int | None = None

    for row_number, (_, row) in enumerate(frame.iterrows(), start=1):
        image_path = resolve_image_path(row["image_path"])
        volume = load_patch(image_path).to(device, non_blocking=True)
        embedding = extract_features(model, volume).squeeze(0).cpu().numpy()
        if embedding.ndim != 1:
            raise RuntimeError(f"Expected a 1D embedding, got {embedding.shape}")
        if feature_count is None:
            feature_count = len(embedding)
            if feature_count != model.embed_dim:
                raise RuntimeError(
                    f"Expected {model.embed_dim} features, got {feature_count}"
                )
        elif len(embedding) != feature_count:
            raise RuntimeError("Feature dimensionality changed between patches")

        output_row = metadata_for_row(row)
        output_row.update(
            {f"feature_{index}": float(value) for index, value in enumerate(embedding)}
        )
        rows.append(output_row)
        print(f"\rExtracted {row_number}/{len(frame)} patches", end="", flush=True)

    print()
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    manifest_path = (
        resolve_cli_path(args.manifest)
        if args.manifest is not None
        else default_manifest(args.dataset, args.patches)
    )
    output_csv = (
        resolve_cli_path(args.output_csv)
        if args.output_csv is not None
        else default_output_csv(args.dataset, args.patches)
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")

    frame = load_manifest(manifest_path, args.dataset, args.patches)
    print(f"Reading {len(frame)} patches from {manifest_path}")
    model = load_nnfoundation_vit(
        repo_id=args.repo_id,
        filename=args.checkpoint_filename,
        device=device,
    )
    features = extract_manifest_features(model, frame, device)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_csv, index=False)
    print(f"Wrote {len(features)} feature rows -> {output_csv}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
