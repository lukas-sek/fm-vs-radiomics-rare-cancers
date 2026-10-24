"""Create offline nnU-Net-style augmentations of normalized 192^3 patches.

The transforms are applied in the order documented by nnU-Net. One combined
affine interpolation handles rotation and scaling. This script targets the
single-modality patches produced by build_nnFoundation_lesion_patches.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("Lipo", "Desmoid", "Liver", "GIST", "CRLM", "Melanoma")
PATCH_SIZE = (192, 192, 192)
PATCH_SPACING = (1.0, 1.0, 1.0)
REQUIRED_COLUMNS = ("patient_id", "lesion_id", "image_path")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "worc_lesion_patches_augmented"
EPSILON = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate three reproducible nnU-Net-style augmented variants for "
            "each normalized 192^3 lesion patch."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS,
        help="WORC dataset whose patch manifest should be augmented.",
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=None,
        help=(
            "Source patch CSV (default: "
            "metadata/worc_lesion_patches_<dataset>.csv)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Root for augmented patches "
            "(default: data/worc_lesion_patches_augmented)."
        ),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=None,
        help=(
            "Combined original+augmented CSV (default: "
            "metadata/worc_lesion_patches_augmented_<dataset>.csv)."
        ),
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=3,
        help="Stochastic variants per original patch (default: 3).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing variants; otherwise verify and reuse them.",
    )
    return parser.parse_args()


def resolve_cli_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def resolve_image_path(raw_path: object) -> Path:
    path = Path(str(raw_path))
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def validate_manifest(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Input manifest is missing columns {missing}; "
            f"required columns are {list(REQUIRED_COLUMNS)}"
        )
    if frame.empty:
        raise ValueError("Input manifest has no rows")
    null_rows = frame[list(REQUIRED_COLUMNS)].isna().any(axis=1)
    if null_rows.any():
        examples = frame.index[null_rows].tolist()[:5]
        raise ValueError(f"Required manifest values are null at rows {examples}")


def validate_patch(image: sitk.Image, path: Path | None = None) -> None:
    prefix = f"{path}: " if path is not None else ""
    if image.GetDimension() != 3:
        raise ValueError(f"{prefix}expected a 3D patch")
    if image.GetSize() != PATCH_SIZE:
        raise ValueError(
            f"{prefix}expected size {PATCH_SIZE}, got {image.GetSize()}"
        )
    if not np.allclose(image.GetSpacing(), PATCH_SPACING, atol=1e-6):
        raise ValueError(
            f"{prefix}expected spacing {PATCH_SPACING}, got {image.GetSpacing()}"
        )
    values = sitk.GetArrayViewFromImage(image)
    if not np.isfinite(values).all():
        raise ValueError(f"{prefix}patch contains non-finite values")


def variant_rng(
    global_seed: int,
    patient_id: str,
    lesion_id: str,
    augmentation_id: int,
) -> tuple[np.random.Generator, str]:
    identity = (
        f"{global_seed}\0{patient_id}\0{lesion_id}\0{augmentation_id}"
    ).encode("utf-8")
    digest = hashlib.blake2b(identity, digest_size=16).digest()
    words = np.frombuffer(digest, dtype="<u4").astype(np.uint32)
    seed_sequence = np.random.SeedSequence(words.tolist())
    return np.random.default_rng(seed_sequence), digest.hex()


def rotation_matrix_xyz(angles_degrees: tuple[float, float, float]) -> np.ndarray:
    ax, ay, az = np.deg2rad(np.asarray(angles_degrees, dtype=np.float64))
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(ax), -np.sin(ax)], [0.0, np.sin(ax), np.cos(ax)]]
    )
    ry = np.array(
        [[np.cos(ay), 0.0, np.sin(ay)], [0.0, 1.0, 0.0], [-np.sin(ay), 0.0, np.cos(ay)]]
    )
    rz = np.array(
        [[np.cos(az), -np.sin(az), 0.0], [np.sin(az), np.cos(az), 0.0], [0.0, 0.0, 1.0]]
    )
    return rz @ ry @ rx


def apply_combined_affine(
    values: np.ndarray,
    angles_degrees: tuple[float, float, float],
    scale: float,
) -> np.ndarray:
    """Rotate and scale around the patch center with one cubic interpolation."""
    if scale <= 0.0:
        raise ValueError(f"Scale must be positive, got {scale}")

    matrix_xyz = rotation_matrix_xyz(angles_degrees) / scale
    xyz_from_zyx = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    )
    matrix_zyx = xyz_from_zyx.T @ matrix_xyz @ xyz_from_zyx
    center = (np.asarray(values.shape, dtype=np.float64) - 1.0) / 2.0
    offset = center - matrix_zyx @ center
    return ndimage.affine_transform(
        values,
        matrix_zyx,
        offset=offset,
        output_shape=values.shape,
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=True,
    ).astype(np.float32, copy=False)


def apply_gaussian_noise(
    values: np.ndarray,
    variance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if variance < 0.0:
        raise ValueError(f"Noise variance must be non-negative, got {variance}")
    noise = rng.normal(0.0, np.sqrt(variance), size=values.shape)
    return (values + noise).astype(np.float32)


def apply_gaussian_blur(values: np.ndarray, sigma: float) -> np.ndarray:
    return ndimage.gaussian_filter(
        values,
        sigma=sigma,
        order=0,
        mode="nearest",
    ).astype(np.float32, copy=False)


def apply_contrast(values: np.ndarray, factor: float) -> np.ndarray:
    old_min = float(np.min(values))
    old_max = float(np.max(values))
    mean = float(np.mean(values, dtype=np.float64))
    contrasted = (values - mean) * factor + mean
    return np.clip(contrasted, old_min, old_max).astype(np.float32, copy=False)


def restore_shape(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    result = np.zeros(shape, dtype=np.float32)
    source_slices = []
    target_slices = []
    for current, target in zip(values.shape, shape):
        copied = min(current, target)
        source_start = (current - copied) // 2
        target_start = (target - copied) // 2
        source_slices.append(slice(source_start, source_start + copied))
        target_slices.append(slice(target_start, target_start + copied))
    result[tuple(target_slices)] = values[tuple(source_slices)]
    return result


def apply_low_resolution(values: np.ndarray, factor: float) -> np.ndarray:
    if factor < 1.0:
        raise ValueError(f"Low-resolution factor must be >= 1, got {factor}")
    original_shape = tuple(int(size) for size in values.shape)
    low_shape = tuple(
        max(1, int(round(size / factor))) for size in original_shape
    )
    downsampled = ndimage.zoom(
        values,
        zoom=tuple(low / high for low, high in zip(low_shape, original_shape)),
        order=0,
        mode="nearest",
        prefilter=False,
        grid_mode=False,
    )
    upsampled = ndimage.zoom(
        downsampled,
        zoom=tuple(
            high / low for high, low in zip(original_shape, downsampled.shape)
        ),
        order=3,
        mode="nearest",
        prefilter=True,
        grid_mode=False,
    )
    return restore_shape(upsampled, original_shape)


def apply_gamma(values: np.ndarray, gamma: float, inverted: bool) -> np.ndarray:
    old_min = float(np.min(values))
    old_max = float(np.max(values))
    value_range = old_max - old_min
    if value_range <= EPSILON:
        return values.astype(np.float32, copy=True)

    normalized = np.clip((values - old_min) / value_range, 0.0, 1.0)
    if inverted:
        transformed = 1.0 - np.power(1.0 - normalized, gamma)
    else:
        transformed = np.power(normalized, gamma)
    return (transformed * value_range + old_min).astype(np.float32)


def apply_mirroring(
    values: np.ndarray,
    mirror_x: bool,
    mirror_y: bool,
    mirror_z: bool,
) -> np.ndarray:
    axes = []
    if mirror_x:
        axes.append(2)
    if mirror_y:
        axes.append(1)
    if mirror_z:
        axes.append(0)
    return np.ascontiguousarray(np.flip(values, axis=tuple(axes))) if axes else values


def augment_patch(
    values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    if values.ndim != 3 or tuple(values.shape) != PATCH_SIZE:
        raise ValueError(
            f"Expected array shape {PATCH_SIZE}, got {tuple(values.shape)}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Cannot augment an array containing non-finite values")
    output = values.astype(np.float32, copy=True)
    parameters: dict[str, Any] = {}

    rotate = bool(rng.random() < 0.2)
    scale_image = bool(rng.random() < 0.2)
    angles = (
        tuple(float(value) for value in rng.uniform(-30.0, 30.0, size=3))
        if rotate
        else (0.0, 0.0, 0.0)
    )
    scale = float(rng.uniform(0.7, 1.4)) if scale_image else 1.0
    parameters.update(
        {
            "rotation_applied": rotate,
            "rotation_degrees_xyz": list(angles),
            "scaling_applied": scale_image,
            "scale_factor": scale,
        }
    )
    if rotate or scale_image:
        output = apply_combined_affine(output, angles, scale)

    noise_applied = bool(rng.random() < 0.15)
    noise_variance = float(rng.uniform(0.0, 0.1)) if noise_applied else None
    parameters["gaussian_noise_applied"] = noise_applied
    parameters["gaussian_noise_variance"] = noise_variance
    if noise_applied and noise_variance is not None:
        output = apply_gaussian_noise(output, noise_variance, rng)

    blur_sample = bool(rng.random() < 0.2)
    blur_applied = blur_sample and bool(rng.random() < 0.5)
    blur_sigma = float(rng.uniform(0.5, 1.5)) if blur_applied else None
    parameters["gaussian_blur_applied"] = blur_applied
    parameters["gaussian_blur_sigma"] = blur_sigma
    if blur_applied and blur_sigma is not None:
        output = apply_gaussian_blur(output, blur_sigma)

    brightness_applied = bool(rng.random() < 0.15)
    brightness_factor = (
        float(rng.uniform(0.7, 1.3)) if brightness_applied else None
    )
    parameters["brightness_applied"] = brightness_applied
    parameters["brightness_factor"] = brightness_factor
    if brightness_applied and brightness_factor is not None:
        output = (output * brightness_factor).astype(np.float32)

    contrast_applied = bool(rng.random() < 0.15)
    contrast_factor = (
        float(rng.uniform(0.65, 1.5)) if contrast_applied else None
    )
    parameters["contrast_applied"] = contrast_applied
    parameters["contrast_factor"] = contrast_factor
    if contrast_applied and contrast_factor is not None:
        output = apply_contrast(output, contrast_factor)

    lowres_sample = bool(rng.random() < 0.25)
    lowres_applied = lowres_sample and bool(rng.random() < 0.5)
    lowres_factor = float(rng.uniform(1.0, 2.0)) if lowres_applied else None
    parameters["low_resolution_applied"] = lowres_applied
    parameters["low_resolution_factor"] = lowres_factor
    if lowres_applied and lowres_factor is not None:
        output = apply_low_resolution(output, lowres_factor)

    inverted_gamma_applied = bool(rng.random() < 0.15)
    inverted_gamma = (
        float(rng.uniform(0.7, 1.5)) if inverted_gamma_applied else None
    )
    parameters["inverted_gamma_applied"] = inverted_gamma_applied
    parameters["inverted_gamma"] = inverted_gamma
    if inverted_gamma_applied and inverted_gamma is not None:
        output = apply_gamma(output, inverted_gamma, inverted=True)

    gamma_applied = bool(rng.random() < 0.15)
    gamma = float(rng.uniform(0.7, 1.5)) if gamma_applied else None
    parameters["gamma_applied"] = gamma_applied
    parameters["gamma"] = gamma
    if gamma_applied and gamma is not None:
        output = apply_gamma(output, gamma, inverted=False)

    mirror_x = bool(rng.random() < 0.5)
    mirror_y = bool(rng.random() < 0.5)
    mirror_z = bool(rng.random() < 0.5)
    parameters["mirror_x"] = mirror_x
    parameters["mirror_y"] = mirror_y
    parameters["mirror_z"] = mirror_z
    output = apply_mirroring(output, mirror_x, mirror_y, mirror_z)

    output = np.ascontiguousarray(output, dtype=np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Augmentation produced non-finite values")
    return output, parameters


def array_to_image(values: np.ndarray, reference: sitk.Image) -> sitk.Image:
    output = sitk.GetImageFromArray(values.astype(np.float32, copy=False))
    output.CopyInformation(reference)
    return output


def write_or_validate(
    image: sitk.Image,
    output_path: Path,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        existing = sitk.ReadImage(str(output_path))
        validate_patch(existing, output_path)
        geometry_matches = (
            np.allclose(existing.GetOrigin(), image.GetOrigin(), atol=1e-6)
            and np.allclose(existing.GetDirection(), image.GetDirection(), atol=1e-7)
        )
        values_match = np.allclose(
            sitk.GetArrayViewFromImage(existing),
            sitk.GetArrayViewFromImage(image),
            rtol=1e-5,
            atol=1e-6,
        )
        if not geometry_matches or not values_match:
            raise FileExistsError(
                f"Existing variant differs from deterministic output: {output_path}. "
                "Use --overwrite to replace it."
            )
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path), useCompression=True)


def nifti_stem(path: Path) -> str:
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


def process_manifest(
    frame: pd.DataFrame,
    dataset: str,
    output_root: Path,
    num_augmentations: int,
    seed: int,
    overwrite: bool,
) -> pd.DataFrame:
    if num_augmentations < 1:
        raise ValueError("--num-augmentations must be at least 1")
    validate_manifest(frame)
    output_dir = output_root / dataset
    rows: list[dict[str, Any]] = []

    for row_number, (_, row) in enumerate(frame.iterrows(), start=1):
        patient_id = str(row["patient_id"])
        lesion_id = str(row["lesion_id"])
        source_path = resolve_image_path(row["image_path"])
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source patch: {source_path}")

        source_image = sitk.ReadImage(str(source_path))
        validate_patch(source_image, source_path)
        source_values = sitk.GetArrayFromImage(source_image).astype(
            np.float32, copy=False
        )

        original_row = row.to_dict()
        original_row.update(
            {
                "image_path": display_path(source_path),
                "augmentation_id": 0,
                "augmentation_type": "original",
                "augmentation_seed": "",
                "augmentation_parameters": "{}",
            }
        )
        rows.append(original_row)

        for augmentation_id in range(1, num_augmentations + 1):
            rng, variant_seed = variant_rng(
                seed, patient_id, lesion_id, augmentation_id
            )
            augmented_values, parameters = augment_patch(source_values, rng)
            augmented_image = array_to_image(augmented_values, source_image)
            validate_patch(augmented_image)

            output_path = (
                output_dir
                / f"{nifti_stem(source_path)}__aug{augmentation_id}.nii.gz"
            )
            write_or_validate(augmented_image, output_path, overwrite)

            augmented_row = row.to_dict()
            augmented_row.update(
                {
                    "image_path": display_path(output_path),
                    "augmentation_id": augmentation_id,
                    "augmentation_type": "nnunet",
                    "augmentation_seed": variant_seed,
                    "augmentation_parameters": json.dumps(
                        parameters, sort_keys=True, separators=(",", ":")
                    ),
                }
            )
            rows.append(augmented_row)

        if row_number % 25 == 0 or row_number == len(frame):
            print(f"  {dataset}: {row_number}/{len(frame)} source patches")

    output_columns = list(frame.columns)
    for column in (
        "augmentation_id",
        "augmentation_type",
        "augmentation_seed",
        "augmentation_parameters",
    ):
        if column not in output_columns:
            output_columns.append(column)
    return pd.DataFrame(rows, columns=output_columns)


def main() -> None:
    args = parse_args()
    input_manifest = (
        resolve_cli_path(args.input_manifest)
        if args.input_manifest is not None
        else REPO_ROOT
        / "metadata"
        / f"worc_lesion_patches_{args.dataset}.csv"
    )
    output_root = resolve_cli_path(args.output_dir)
    output_manifest = (
        resolve_cli_path(args.output_manifest)
        if args.output_manifest is not None
        else REPO_ROOT
        / "metadata"
        / f"worc_lesion_patches_augmented_{args.dataset}.csv"
    )
    if not input_manifest.is_file():
        raise SystemExit(f"Input manifest not found: {input_manifest}")

    frame = pd.read_csv(input_manifest)
    combined = process_manifest(
        frame,
        args.dataset,
        output_root,
        args.num_augmentations,
        args.seed,
        args.overwrite,
    )
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_manifest, index=False)
    print(
        f"Wrote {len(combined)} manifest rows "
        f"({args.num_augmentations + 1} per source) -> "
        f"{display_path(output_manifest)}"
    )
    print(f"Augmented patches under {display_path(output_root / args.dataset)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
