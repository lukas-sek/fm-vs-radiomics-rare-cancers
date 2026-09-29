"""Offline 50^3 FMCIB patch extraction with dedicated single-slot augmentations.

Extracts a 1.0 mm isotropic 50x50x50 patch centered on the lesion centroid, then
applies exactly one isolated transform per variant (flip / rotate / zoom).

Inference: get_features(csv_path, precropped=True)

Input CSV required columns:
  patient_id, lesion_id, image_path, coordX, coordY, coordZ, label
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_SIZE = (50, 50, 50)
PATCH_SPACING = (1.0, 1.0, 1.0)
IDENTITY_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

REQUIRED_COLUMNS = (
    "patient_id",
    "lesion_id",
    "image_path",
    "coordX",
    "coordY",
    "coordZ",
    "label",
)

AUG_SPECS = (
    (0, "original", None),
    (1, "flip", "flip"),
    (2, "rotate", "rotate"),
    (3, "zoom", "zoom"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract FMCIB 50^3 patches and apply dedicated 3D augmentations."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. Melanoma")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default= REPO_ROOT /'data' / 'fmcib_CSVs' / f'fmcib_original_<dataset>.csv',
        help="Input CSV (default: fmcib_CSVs/fmcib_original_<dataset>.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "fmcib_patches",
        help="Root directory for patch NIfTIs (default: data/fmcib_patches)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing patch files; otherwise skip existing paths",
    )
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    import torch
    from monai.utils import set_determinism

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


def validate_input_csv(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Input CSV missing required columns: {missing}. "
            f"Required: {list(REQUIRED_COLUMNS)}"
        )
    if df.empty:
        raise SystemExit("Input CSV has no rows")
    null_mask = df[list(REQUIRED_COLUMNS)].isna().any(axis=1)
    if null_mask.any():
        idxs = df.index[null_mask].tolist()[:5]
        raise SystemExit(f"Input CSV has nulls in required columns at rows: {idxs}")


def resolve_image_path(image_path: str | Path) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def image_min_value(image: sitk.Image) -> float:
    stats = sitk.MinimumMaximumImageFilter()
    stats.Execute(image)
    return float(stats.GetMinimum())


def extract_base_patch(
    image: sitk.Image,
    center_phys: tuple[float, float, float],
    pad_value: float,
) -> sitk.Image:
    """Resample a 50 mm isotropic patch centered at center_phys (identity LPS axes)."""
    size = PATCH_SIZE
    spacing = np.array(PATCH_SPACING, dtype=float)
    direction = np.eye(3, dtype=float)
    center_idx = np.array([(s - 1) / 2.0 for s in size], dtype=float)
    origin = tuple(np.array(center_phys, dtype=float) - direction @ (center_idx * spacing))

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(PATCH_SPACING)
    resampler.SetSize(size)
    resampler.SetOutputDirection(IDENTITY_DIRECTION)
    resampler.SetOutputOrigin(origin)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(pad_value)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    return resampler.Execute(image)


def sitk_to_monai(patch: sitk.Image) -> Any:
    import torch

    arr = sitk.GetArrayFromImage(patch).astype(np.float32)  # z, y, x
    # MONAI spatial axes: (C, X, Y, Z) matching LPS index order used by FMCIB crops
    vol = np.transpose(arr, (2, 1, 0))
    return torch.from_numpy(vol).unsqueeze(0)


def monai_to_sitk(tensor: Any, reference: sitk.Image) -> sitk.Image:
    vol = tensor.detach().cpu().numpy()
    if vol.ndim == 4:
        vol = vol[0]
    arr = np.transpose(vol, (2, 1, 0)).astype(np.float32)  # back to z, y, x
    out = sitk.GetImageFromArray(arr)
    out.SetSpacing(reference.GetSpacing())
    out.SetOrigin(reference.GetOrigin())
    out.SetDirection(IDENTITY_DIRECTION)
    return out


def apply_flip(patch_t: Any) -> Any:
    from monai.transforms import Flip

    return Flip(spatial_axis=0)(patch_t)


def apply_rotate(patch_t: Any) -> Any:
    from monai.transforms import RandAffine

    # Torch backend (avoids scipy/torch padding_mode mismatch of RandRotate/RandZoom).
    return RandAffine(
        rotate_range=(0.26, 0.26, 0.26),
        prob=1.0,
        mode="bilinear",
        padding_mode="border",
    )(patch_t)


def apply_zoom(patch_t: Any) -> Any:
    from monai.transforms import Affine

    # Isotropic zoom in [0.95, 1.05] via torch Affine (border padding).
    zoom = float(np.random.uniform(0.95, 1.05))
    return Affine(
        scale_params=(zoom, zoom, zoom),
        mode="bilinear",
        padding_mode="border",
        image_only=True,
    )(patch_t)


def apply_augmentation(patch: sitk.Image, aug_key: str | None) -> sitk.Image:
    if aug_key is None:
        return patch
    patch_t = sitk_to_monai(patch)
    if aug_key == "flip":
        out_t = apply_flip(patch_t)
    elif aug_key == "rotate":
        out_t = apply_rotate(patch_t)
    elif aug_key == "zoom":
        out_t = apply_zoom(patch_t)
    else:
        raise ValueError(f"Unknown augmentation key: {aug_key}")
    return monai_to_sitk(out_t, patch)


def patch_filename(patient_id: str, lesion_id: str, aug_id: int) -> str:
    return f"{patient_id}_{lesion_id}_aug{aug_id}.nii.gz"


def assert_fmcib_patch(patch: sitk.Image) -> None:
    size = patch.GetSize()
    spacing = patch.GetSpacing()
    direction = tuple(float(x) for x in patch.GetDirection())
    if size != PATCH_SIZE:
        raise ValueError(f"Expected size {PATCH_SIZE}, got {size}")
    if not np.allclose(spacing, PATCH_SPACING, atol=1e-6):
        raise ValueError(f"Expected spacing {PATCH_SPACING}, got {spacing}")
    if not np.allclose(direction, IDENTITY_DIRECTION, atol=1e-6):
        raise ValueError(f"Expected identity direction, got {direction}")


def process_row(
    row: pd.Series,
    dataset: str,
    out_dir: Path,
    overwrite: bool,
) -> list[dict]:
    patient_id = str(row["patient_id"])
    lesion_id = str(row["lesion_id"])
    label = int(row["label"])
    center = (float(row["coordX"]), float(row["coordY"]), float(row["coordZ"]))
    image_path = resolve_image_path(row["image_path"])
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    image = sitk.ReadImage(str(image_path))
    pad_value = image_min_value(image)
    base_patch = extract_base_patch(image, center, pad_value)
    assert_fmcib_patch(base_patch)
    del image  # free full volume ASAP

    meta_rows: list[dict] = []
    for aug_id, aug_type, aug_key in AUG_SPECS:
        out_name = patch_filename(patient_id, lesion_id, aug_id)
        out_path = out_dir / out_name
        rel_path = out_path.resolve().relative_to(REPO_ROOT).as_posix()

        if out_path.exists() and not overwrite:
            pass
        else:
            variant = apply_augmentation(base_patch, aug_key)
            assert_fmcib_patch(variant)
            out_dir.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(variant, str(out_path))

        meta_rows.append(
            {
                "dataset": dataset,
                "patient_id": patient_id,
                "lesion_id": lesion_id,
                "aug_id": aug_id,
                "aug_type": aug_type,
                "image_path": rel_path,
                "label": label,
            }
        )
    return meta_rows


def main() -> None:
    args = parse_args()

    input_csv = args.input_csv
    if input_csv is None:
        input_csv = REPO_ROOT / "fmcib_CSVs" / f"fmcib_original_{args.dataset}.csv"
    elif not input_csv.is_absolute():
        input_csv = (REPO_ROOT / input_csv).resolve()

    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    validate_input_csv(df)
    set_seeds(args.seed)

    out_root = args.output_dir
    if not out_root.is_absolute():
        out_root = (REPO_ROOT / out_root).resolve()
    dataset_out = out_root / args.dataset
    dataset_out.mkdir(parents=True, exist_ok=True)

    meta_rows: list[dict] = []
    n = len(df)
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        meta_rows.extend(process_row(row, args.dataset, dataset_out, args.overwrite))
        if i % 25 == 0 or i == n:
            print(f"  {args.dataset}: {i}/{n} lesions")

    meta_df = pd.DataFrame(meta_rows)
    out_csv = REPO_ROOT / 'experiments' / "fmcib_CSVs" / f"fmcib_patches_{args.dataset}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    meta_df.to_csv(out_csv, index=False)
    print(f"Wrote {len(meta_df)} rows -> {out_csv.relative_to(REPO_ROOT).as_posix()}")
    print(f"Patches under {dataset_out.relative_to(REPO_ROOT).as_posix()}")
    print("Use get_features(..., precropped=True) on the patches CSV.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
