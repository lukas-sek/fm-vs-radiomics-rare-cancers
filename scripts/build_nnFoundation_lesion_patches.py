"""Build lesion-centered 192 mm patches from an original WORC dataset.

Each output is a float32 NIfTI with 192x192x192 voxels at 1 mm isotropic
spacing. Intensities are z-score normalized per patch using only voxels covered
by the source image; out-of-image padding is set to zero after normalization.
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
WORC_ROOT = REPO_ROOT / "data" / "worc"
PINFO_ROOT = REPO_ROOT / "metadata"
DATASETS = ("Lipo", "Desmoid", "Liver", "GIST", "CRLM", "Melanoma")
DATASET_MODALITY = {
    "Lipo": "MR",
    "Desmoid": "MR",
    "Liver": "MR",
    "GIST": "CT",
    "CRLM": "CT",
    "Melanoma": "CT",
}
CRLM_SOURCE_PRIORITY = ("RAD", "STUD1", "STUD2", "PhD", "CNN")
CT_AIR_HU = -1000.0
PATCH_SIZE = (192, 192, 192)
PATCH_SPACING = (1.0, 1.0, 1.0)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "worc_lesion_patches"
STD_EPSILON = 1e-8

MANIFEST_COLUMNS = (
    "dataset",
    "patient_id",
    "lesion_id",
    "image_path",
    "source_image_path",
    "source_mask_path",
    "label",
    "mask_source",
    "coordX",
    "coordY",
    "coordZ",
    "valid_voxel_fraction",
    "normalization_mean",
    "normalization_std",
)


def get_foreground_label(stats: sitk.LabelShapeStatisticsImageFilter) -> int:
    labels = list(stats.GetLabels())
    if not labels:
        raise ValueError("Mask has no foreground labels")
    if 255 in labels:
        return 255
    if 1 in labels:
        return 1
    return int(labels[0])


def get_clean_lesion_mask(mask: sitk.Image) -> tuple[sitk.Image, int]:
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    label = get_foreground_label(stats)
    connected = sitk.ConnectedComponent(sitk.Equal(mask, label))
    relabeled = sitk.RelabelComponent(connected)
    return sitk.Cast(sitk.Equal(relabeled, 1), sitk.sitkUInt8), 1


def get_centroid(mask: sitk.Image) -> tuple[float, float, float]:
    cleaned, label = get_clean_lesion_mask(mask)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cleaned)
    return tuple(float(value) for value in stats.GetCentroid(label))


def get_background_value(image: sitk.Image, modality: str) -> float:
    if modality == "CT":
        return CT_AIR_HU
    stats = sitk.MinimumMaximumImageFilter()
    stats.Execute(image)
    return float(stats.GetMinimum())


def load_pinfo(dataset: str) -> dict[str, int]:
    path = PINFO_ROOT / f"pinfo_{dataset}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing patient labels: {path}")
    frame = pd.read_csv(path)
    return {
        str(row["Patient"]): int(row["Diagnosis"])
        for _, row in frame.iterrows()
    }


def resolve_label(
    dataset: str,
    patient_id: str,
    lesion_id: str,
    pinfo: dict[str, int],
) -> int:
    if dataset == "Lipo" and lesion_id == "WDLPS":
        return 1
    if dataset == "Lipo" and lesion_id == "Lipoma":
        return 0
    if patient_id not in pinfo:
        raise KeyError(f"No pinfo label for {patient_id}")
    return pinfo[patient_id]


def discover_lesion_masks(
    dataset: str,
    patient_dir: Path,
) -> list[tuple[str, Path, str | None]]:
    if dataset == "Lipo":
        wdlps = patient_dir / "segmentation_WDLPS.nii.gz"
        lipoma = patient_dir / "segmentation_Lipoma.nii.gz"
        if wdlps.exists() and lipoma.exists():
            return [("WDLPS", wdlps, None), ("Lipoma", lipoma, None)]

    if dataset == "CRLM":
        by_lesion: dict[str, dict[str, Path]] = {}
        pattern = re.compile(
            r"segmentation_(lesion\d+)_(RAD|STUD1|STUD2|PhD|CNN)\.nii\.gz$"
        )
        for path in sorted(patient_dir.glob("segmentation_lesion*_*.nii.gz")):
            match = pattern.match(path.name)
            if match:
                lesion_id, source = match.groups()
                by_lesion.setdefault(lesion_id, {})[source] = path

        rows: list[tuple[str, Path, str | None]] = []
        for lesion_id, sources in sorted(by_lesion.items()):
            source = next(
                (name for name in CRLM_SOURCE_PRIORITY if name in sources),
                None,
            )
            if source is None:
                raise FileNotFoundError(
                    f"No usable segmentation for {patient_dir.name} {lesion_id}"
                )
            if source not in ("RAD", "STUD1"):
                warnings.warn(
                    f"{patient_dir.name} {lesion_id}: using '{source}' mask",
                    stacklevel=2,
                )
            rows.append((lesion_id, sources[source], source))
        if rows:
            return rows

    melanoma_pattern = re.compile(r"segmentation_(lesion\d+)\.nii\.gz$")
    melanoma_masks = [
        (match.group(1), path, None)
        for path in sorted(patient_dir.glob("segmentation_lesion*.nii.gz"))
        if (match := melanoma_pattern.match(path.name))
    ]
    if melanoma_masks:
        return melanoma_masks

    gist_pattern = re.compile(r"segmentation_(lesion_\d+)\.nii\.gz$")
    gist_masks = [
        (match.group(1), path, None)
        for path in sorted(patient_dir.glob("segmentation_lesion_*.nii.gz"))
        if (match := gist_pattern.match(path.name))
    ]
    if gist_masks:
        return gist_masks

    default = patient_dir / "segmentation.nii.gz"
    if default.exists():
        return [("lesion0", default, None)]
    raise FileNotFoundError(f"No segmentation found in {patient_dir}")


def resolve_image_path(patient_dir: Path, lesion_id: str) -> Path:
    shared = patient_dir / "image.nii.gz"
    if shared.exists():
        return shared
    per_lesion = patient_dir / f"image_{lesion_id}.nii.gz"
    if per_lesion.exists():
        return per_lesion
    raise FileNotFoundError(
        f"Missing image for {patient_dir.name} {lesion_id}: "
        f"tried {shared.name} and {per_lesion.name}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create lesion-centered, z-score normalized 192^3 patches at "
            "1 mm isotropic spacing from one WORC dataset."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS,
        help="WORC dataset to process.",
    )
    parser.add_argument(
        "--worc-root",
        type=Path,
        default=WORC_ROOT,
        help="WORC data root (default: data/worc).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Patch output root (default: data/worc_lesion_patches).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Output CSV path "
            "(default: metadata/worc_lesion_patches_<dataset>.csv)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing patches. Without this flag, matching files are reused.",
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


def patch_origin(
    center_phys: tuple[float, float, float],
    direction: tuple[float, ...],
) -> tuple[float, float, float]:
    direction_matrix = np.asarray(direction, dtype=float).reshape(3, 3)
    center_index = (np.asarray(PATCH_SIZE, dtype=float) - 1.0) / 2.0
    offset = direction_matrix @ (center_index * np.asarray(PATCH_SPACING))
    return tuple(np.asarray(center_phys, dtype=float) - offset)


def configure_resampler(
    center_phys: tuple[float, float, float],
    direction: tuple[float, ...],
    interpolator: int,
    default_value: float,
    output_pixel_type: int,
) -> sitk.ResampleImageFilter:
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(PATCH_SIZE)
    resampler.SetOutputSpacing(PATCH_SPACING)
    resampler.SetOutputDirection(direction)
    resampler.SetOutputOrigin(patch_origin(center_phys, direction))
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetOutputPixelType(output_pixel_type)
    return resampler


def make_source_support(image: sitk.Image) -> sitk.Image:
    support = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    support.CopyInformation(image)
    support = support + 1
    return support


def assert_center_inside_image(
    image: sitk.Image,
    center_phys: tuple[float, float, float],
) -> None:
    center_index = image.TransformPhysicalPointToContinuousIndex(center_phys)
    size = image.GetSize()
    if not all(-0.5 <= center_index[i] <= size[i] - 0.5 for i in range(3)):
        raise ValueError(
            f"Lesion centroid {center_phys} maps outside the source image "
            f"(continuous index {center_index}, size {size})"
        )


def normalize_valid_voxels(
    patch: sitk.Image,
    support: sitk.Image,
) -> tuple[sitk.Image, float, float, float]:
    values = sitk.GetArrayFromImage(patch).astype(np.float32, copy=False)
    valid = sitk.GetArrayViewFromImage(support) > 0
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        raise ValueError("Resampled patch has no overlap with the source image")

    valid_values = values[valid].astype(np.float64, copy=False)
    if not np.isfinite(valid_values).all():
        raise ValueError("Source-covered patch voxels contain non-finite intensities")

    mean = float(np.mean(valid_values))
    std = float(np.std(valid_values))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= STD_EPSILON:
        raise ValueError(
            f"Cannot z-score patch with zero or invalid variance "
            f"(mean={mean}, std={std})"
        )

    normalized = np.zeros(values.shape, dtype=np.float32)
    normalized[valid] = ((valid_values - mean) / std).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("Normalized patch contains non-finite intensities")

    output = sitk.GetImageFromArray(normalized)
    output.CopyInformation(patch)
    valid_fraction = valid_count / float(np.prod(PATCH_SIZE))
    return output, mean, std, valid_fraction


def create_normalized_patch(
    image: sitk.Image,
    center_phys: tuple[float, float, float],
    background_value: float,
) -> tuple[sitk.Image, float, float, float]:
    assert_center_inside_image(image, center_phys)
    direction = tuple(float(value) for value in image.GetDirection())

    image_resampler = configure_resampler(
        center_phys,
        direction,
        sitk.sitkLinear,
        background_value,
        sitk.sitkFloat32,
    )
    raw_patch = image_resampler.Execute(image)

    support_resampler = configure_resampler(
        center_phys,
        direction,
        sitk.sitkNearestNeighbor,
        0.0,
        sitk.sitkUInt8,
    )
    support_patch = support_resampler.Execute(make_source_support(image))
    return normalize_valid_voxels(raw_patch, support_patch)


def assert_patch_geometry(patch: sitk.Image) -> None:
    if patch.GetSize() != PATCH_SIZE:
        raise ValueError(f"Expected patch size {PATCH_SIZE}, got {patch.GetSize()}")
    if not np.allclose(patch.GetSpacing(), PATCH_SPACING, atol=1e-7):
        raise ValueError(
            f"Expected patch spacing {PATCH_SPACING}, got {patch.GetSpacing()}"
        )
    if patch.GetPixelID() != sitk.sitkFloat32:
        raise ValueError(
            f"Expected float32 patch, got {patch.GetPixelIDTypeAsString()}"
        )


def write_or_validate_patch(
    patch: sitk.Image,
    output_path: Path,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        existing = sitk.ReadImage(str(output_path))
        assert_patch_geometry(existing)
        geometry_matches = (
            np.allclose(existing.GetOrigin(), patch.GetOrigin(), atol=1e-6)
            and np.allclose(existing.GetDirection(), patch.GetDirection(), atol=1e-7)
        )
        values_match = np.allclose(
            sitk.GetArrayViewFromImage(existing),
            sitk.GetArrayViewFromImage(patch),
            rtol=1e-5,
            atol=1e-6,
        )
        if not geometry_matches or not values_match:
            raise FileExistsError(
                f"Existing patch differs from current source: {output_path}. "
                "Use --overwrite to replace it."
            )
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(patch, str(output_path), useCompression=True)


def process_dataset(
    dataset: str,
    worc_root: Path,
    output_root: Path,
    overwrite: bool,
) -> pd.DataFrame:
    dataset_root = worc_root / dataset
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Missing WORC dataset directory: {dataset_root}")

    patient_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    if not patient_dirs:
        raise ValueError(f"No patient directories found under {dataset_root}")

    pinfo = load_pinfo(dataset)
    modality = DATASET_MODALITY[dataset]
    dataset_output = output_root / dataset
    rows: list[dict[str, object]] = []

    for patient_number, patient_dir in enumerate(patient_dirs, start=1):
        patient_id = patient_dir.name
        image_cache: dict[Path, sitk.Image] = {}
        background_cache: dict[Path, float] = {}

        for lesion_id, mask_path, mask_source in discover_lesion_masks(
            dataset, patient_dir
        ):
            source_image_path = resolve_image_path(patient_dir, lesion_id).resolve()
            if source_image_path not in image_cache:
                image = sitk.ReadImage(str(source_image_path))
                if image.GetDimension() != 3:
                    raise ValueError(
                        f"Expected a 3D image, got {image.GetDimension()}D: "
                        f"{source_image_path}"
                    )
                image_cache[source_image_path] = image
                background_cache[source_image_path] = get_background_value(
                    image, modality
                )

            image = image_cache[source_image_path]
            mask = sitk.ReadImage(str(mask_path))
            center = get_centroid(mask)
            patch, mean, std, valid_fraction = create_normalized_patch(
                image,
                center,
                background_cache[source_image_path],
            )
            assert_patch_geometry(patch)

            output_path = dataset_output / f"{patient_id}__{lesion_id}.nii.gz"
            write_or_validate_patch(patch, output_path, overwrite)
            rows.append(
                {
                    "dataset": dataset,
                    "patient_id": patient_id,
                    "lesion_id": lesion_id,
                    "image_path": display_path(output_path),
                    "source_image_path": display_path(source_image_path),
                    "source_mask_path": display_path(mask_path),
                    "label": int(
                        resolve_label(dataset, patient_id, lesion_id, pinfo)
                    ),
                    "mask_source": mask_source or "",
                    "coordX": center[0],
                    "coordY": center[1],
                    "coordZ": center[2],
                    "valid_voxel_fraction": valid_fraction,
                    "normalization_mean": mean,
                    "normalization_std": std,
                }
            )

        if patient_number % 25 == 0 or patient_number == len(patient_dirs):
            print(f"  {dataset}: {patient_number}/{len(patient_dirs)} patients")

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def main() -> None:
    args = parse_args()
    worc_root = resolve_cli_path(args.worc_root)
    output_root = resolve_cli_path(args.output_dir)
    manifest_path = (
        resolve_cli_path(args.manifest)
        if args.manifest is not None
        else REPO_ROOT / "metadata" / f"worc_lesion_patches_{args.dataset}.csv"
    )

    manifest = process_dataset(
        args.dataset,
        worc_root,
        output_root,
        args.overwrite,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    print(f"Wrote {len(manifest)} patches under {display_path(output_root / args.dataset)}")
    print(f"Wrote manifest -> {display_path(manifest_path)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
