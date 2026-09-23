"""Build full-lesion 50^3 crops and FMCIB CSVs for WORC datasets.

Lesions are isolated and resampled into a 50x50x50 voxel cube. Lesions smaller
than 50 mm use a 50 mm window at 1.0 mm spacing; larger lesions expand the window
and scale spacing so the full boundary fits into the 50^3 tensor.

Inference must use: get_features(csv_path, precropped=True)
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
WORC_ROOT = REPO_ROOT / "data" / "worc"
OUT_CROP_ROOT = REPO_ROOT / "data" / "fmcib_full_lesion"
OUT_CSV_ROOT = REPO_ROOT / "metadata"
PINFO_ROOT = REPO_ROOT / "metadata"

DATASETS = ("Lipo", "Desmoid", "Liver", "GIST", "CRLM", "Melanoma")
EXPECTED_LESIONS = {
    "Lipo": 116,
    "Desmoid": 203,
    "Liver": 186,
    "GIST": 247,
    "CRLM": 93,
    "Melanoma": 169,
}
OUT_SIZE = (50, 50, 50)
MARGIN_MM = 4.0
MIN_SIDE_MM = 50.0

DATASET_MODALITY = {
    "Lipo": "MR",
    "Desmoid": "MR",
    "Liver": "MR",
    "GIST": "CT",
    "CRLM": "CT",
    "Melanoma": "CT",
}
CT_AIR_HU = -1000.0
CRLM_SOURCE_PRIORITY = ("RAD", "STUD1", "STUD2", "PhD", "CNN")


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

    binary = sitk.Equal(mask, label)
    cc = sitk.ConnectedComponent(binary)
    relabeled = sitk.RelabelComponent(cc)
    cleaned = sitk.Cast(sitk.Equal(relabeled, 1), sitk.sitkUInt8)
    return cleaned, 1


def get_centroid(mask: sitk.Image) -> tuple[float, float, float]:
    cleaned, label = get_clean_lesion_mask(mask)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cleaned)
    return tuple(float(x) for x in stats.GetCentroid(label))


def bbox_center_and_extents_mm(mask: sitk.Image) -> tuple[tuple[float, float, float], np.ndarray]:
    cleaned, label = get_clean_lesion_mask(mask)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cleaned)
    x, y, z, sx, sy, sz = stats.GetBoundingBox(label)
    spacing = np.array(mask.GetSpacing(), dtype=float)
    extents = np.array([sx, sy, sz], dtype=float) * spacing
    center_idx = (x + sx / 2.0, y + sy / 2.0, z + sz / 2.0)
    center_phys = mask.TransformContinuousIndexToPhysicalPoint(center_idx)
    return tuple(float(c) for c in center_phys), extents


def direction_obliqueness(image: sitk.Image) -> float:
    abs_d = np.abs(np.array(image.GetDirection(), dtype=float).reshape(3, 3))
    return float(np.max(abs_d.sum(axis=0) - abs_d.max(axis=0)))


def get_background_value(image: sitk.Image, modality: str) -> float:
    if modality == "CT":
        return CT_AIR_HU
    stats = sitk.MinimumMaximumImageFilter()
    stats.Execute(image)
    return float(stats.GetMinimum())


def apply_per_axis_antialias(
    image: sitk.Image,
    orig_spacing: tuple[float, float, float] | np.ndarray,
    new_spacing: tuple[float, float, float] | np.ndarray,
) -> sitk.Image:
    smoothed = sitk.Cast(image, sitk.sitkFloat32)
    applied = False
    for axis in range(3):
        ratio = new_spacing[axis] / orig_spacing[axis]
        if ratio > 1.05:
            sigma_mm = float(0.5 * orig_spacing[axis] * (ratio - 1.0))
            if sigma_mm > 0.05:
                smoother = sitk.RecursiveGaussianImageFilter()
                smoother.SetDirection(axis)
                smoother.SetSigma(sigma_mm)
                smoother.SetOrder(sitk.RecursiveGaussianImageFilter.ZeroOrder)
                smoother.SetNormalizeAcrossScale(False)
                smoothed = smoother.Execute(smoothed)
                applied = True
    return smoothed if applied else image


def resample_full_lesion_cube(
    image: sitk.Image,
    mask: sitk.Image,
    background_value: float,
    out_size: tuple[int, int, int] = OUT_SIZE,
    margin_mm: float = MARGIN_MM,
) -> tuple[sitk.Image, tuple[float, float, float], float, float, tuple[float, float, float]]:
    center_phys, extents = bbox_center_and_extents_mm(mask)
    com_phys = get_centroid(mask)
    side_mm = float(max(np.max(extents) + margin_mm, MIN_SIDE_MM))
    voxel_spacing = side_mm / out_size[0]

    orig_spacing = np.array(image.GetSpacing(), dtype=float)
    new_spacing = np.array([voxel_spacing, voxel_spacing, voxel_spacing], dtype=float)

    smoothed = apply_per_axis_antialias(image, orig_spacing, new_spacing)

    direction = np.array(image.GetDirection(), dtype=float).reshape(3, 3)
    center_cont = np.array([(s - 1) / 2.0 for s in out_size], dtype=float)
    new_origin = tuple(np.array(center_phys, dtype=float) - direction @ (center_cont * new_spacing))

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(float(s) for s in new_spacing))
    resampler.SetSize(out_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(new_origin)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(background_value)
    resampler.SetOutputPixelType(sitk.sitkFloat32)

    crop = resampler.Execute(smoothed)
    return crop, center_phys, side_mm, voxel_spacing, com_phys


def load_pinfo(dataset: str) -> dict[str, int]:
    path = PINFO_ROOT / f"pinfo_{dataset}.csv"
    df = pd.read_csv(path)
    return {str(row["Patient"]): int(row["Diagnosis"]) for _, row in df.iterrows()}


def resolve_label(dataset: str, patient_id: str, lesion_id: str, pinfo: dict[str, int]) -> int:
    if dataset == "Lipo" and lesion_id == "WDLPS":
        return 1
    if dataset == "Lipo" and lesion_id == "Lipoma":
        return 0
    if patient_id not in pinfo:
        raise KeyError(f"No pinfo label for {patient_id}")
    return pinfo[patient_id]


def discover_lesion_masks(dataset: str, patient_dir: Path) -> list[tuple[str, Path, str | None]]:
    if dataset == "Lipo":
        wdlps = patient_dir / "segmentation_WDLPS.nii.gz"
        lipoma = patient_dir / "segmentation_Lipoma.nii.gz"
        if wdlps.exists() and lipoma.exists():
            return [("WDLPS", wdlps, None), ("Lipoma", lipoma, None)]

    if dataset == "CRLM":
        by_lesion: dict[str, dict[str, Path]] = {}
        for path in sorted(patient_dir.glob("segmentation_lesion*_*.nii.gz")):
            m = re.match(r"segmentation_(lesion\d+)_(RAD|STUD1|STUD2|PhD|CNN)\.nii\.gz$", path.name)
            if not m:
                continue
            lesion_id, source = m.group(1), m.group(2)
            by_lesion.setdefault(lesion_id, {})[source] = path
        rows = []
        for lesion_id, sources in sorted(by_lesion.items()):
            chosen_source = next((s for s in CRLM_SOURCE_PRIORITY if s in sources), None)
            if chosen_source is None:
                raise FileNotFoundError(
                    f"No usable mask (checked {CRLM_SOURCE_PRIORITY}) for "
                    f"{patient_dir.name} {lesion_id}; found sources: {sorted(sources)}"
                )
            if chosen_source not in ("RAD", "STUD1"):
                warnings.warn(
                    f"{patient_dir.name} {lesion_id}: falling back to '{chosen_source}'-quality "
                    f"mask (no RAD/STUD1 available; had {sorted(sources)})."
                )
            rows.append((lesion_id, sources[chosen_source], chosen_source))
        return rows

    melanoma_masks = sorted(patient_dir.glob("segmentation_lesion*.nii.gz"))
    melanoma_masks = [p for p in melanoma_masks if re.match(r"segmentation_lesion\d+\.nii\.gz$", p.name)]
    if melanoma_masks:
        return [
            (re.match(r"segmentation_(lesion\d+)\.nii\.gz$", p.name).group(1), p, None)
            for p in melanoma_masks
        ]

    gist_masks = sorted(patient_dir.glob("segmentation_lesion_*.nii.gz"))
    if gist_masks:
        return [
            (re.match(r"segmentation_(lesion_\d+)\.nii\.gz$", p.name).group(1), p, None)
            for p in gist_masks
        ]

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
        f"Missing image for {patient_dir.name} lesion {lesion_id}: "
        f"tried {shared.name} and {per_lesion.name}"
    )


def process_dataset(dataset: str) -> pd.DataFrame:
    pinfo = load_pinfo(dataset)
    modality = DATASET_MODALITY.get(dataset, "CT")
    patient_dirs = sorted(p for p in (WORC_ROOT / dataset).iterdir() if p.is_dir())
    out_dir = OUT_CROP_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    qc_rows = []
    missing_rad = 0
    n_oblique = 0
    max_obliqueness = 0.0

    for i, patient_dir in enumerate(patient_dirs, start=1):
        patient_id = patient_dir.name
        lesions = discover_lesion_masks(dataset, patient_dir)
        image_cache: dict[Path, sitk.Image] = {}
        background_cache: dict[Path, float] = {}

        for lesion_id, mask_path, seg_source in lesions:
            if dataset == "CRLM" and seg_source == "STUD1":
                missing_rad += 1

            image_path = resolve_image_path(patient_dir, lesion_id)
            if image_path not in image_cache:
                image = sitk.ReadImage(str(image_path))
                image_cache[image_path] = image
                obl = direction_obliqueness(image)
                max_obliqueness = max(max_obliqueness, obl)
                if obl > 1e-2:
                    n_oblique += 1
                background_cache[image_path] = get_background_value(image, modality)

            image = image_cache[image_path]
            background_value = background_cache[image_path]
            mask = sitk.ReadImage(str(mask_path))

            crop, center_phys, side_mm, voxel_spacing, com_phys = resample_full_lesion_cube(
                image, mask, background_value
            )
            _, extents = bbox_center_and_extents_mm(mask)
            centroid_bbox_dist = float(np.linalg.norm(np.array(com_phys) - np.array(center_phys)))

            out_path = out_dir / f"{patient_id}__{lesion_id}.nii.gz"
            sitk.WriteImage(crop, str(out_path))

            label = resolve_label(dataset, patient_id, lesion_id, pinfo)
            rows.append(
                {
                    "image_path": out_path.relative_to(REPO_ROOT).as_posix(),
                    "coordX": center_phys[0],
                    "coordY": center_phys[1],
                    "coordZ": center_phys[2],
                    "label": int(label),
                    "mask_source": seg_source if seg_source is not None else "",
                }
            )
            qc_rows.append(
                {
                    "crop_side_mm": side_mm,
                    "voxel_spacing_mm": voxel_spacing,
                    "bbox_max_extent_mm": float(np.max(extents)),
                    "exceeds_50mm_published_crop": bool(np.max(extents) > 50.0),
                    "background_value": background_value,
                    "centroid_to_bbox_center_mm": centroid_bbox_dist,
                }
            )

        if i % 50 == 0 or i == len(patient_dirs):
            print(f"  {dataset}: {i}/{len(patient_dirs)} patients")

    df = pd.DataFrame(rows)
    qc = pd.DataFrame(qc_rows)
    csv_path = OUT_CSV_ROOT / f"fmcib_full_lesion_{dataset}.csv"
    df.to_csv(csv_path, index=False)

    expected = EXPECTED_LESIONS[dataset]
    n_over_50 = int(qc["exceeds_50mm_published_crop"].sum())
    print(f"=== {dataset} ({modality}) ===")
    print(f"Wrote {len(df)} rows -> {csv_path}")
    print(f"Expected lesions: {expected} | delta: {len(df) - expected}")
    if n_oblique:
        print(f"Oblique direction matrices: {n_oblique} images (max residual {max_obliqueness:.3f})")
    if dataset == "CRLM":
        print(f"CRLM lesions using STUD1 fallback (no RAD): {missing_rad}")
    print(
        f"Crop side mm min/median/max: "
        f"{qc['crop_side_mm'].min():.1f} / {qc['crop_side_mm'].median():.1f} / {qc['crop_side_mm'].max():.1f}"
    )
    print(
        f"Voxel spacing mm min/median/max: "
        f"{qc['voxel_spacing_mm'].min():.3f} / {qc['voxel_spacing_mm'].median():.3f} / {qc['voxel_spacing_mm'].max():.3f}"
    )
    print(f"Lesions exceeding 50 mm (scaled down to fit 50^3): {n_over_50}/{len(df)}")
    return df


def main(datasets: tuple[str, ...] | None = None) -> None:
    if not WORC_ROOT.exists():
        raise SystemExit(f"Missing WORC data root: {WORC_ROOT}")
    for ds in datasets or DATASETS:
        process_dataset(ds)
    print("\nDone. Use get_features(..., precropped=True) on these CSVs.")


if __name__ == "__main__":
    import sys

    selected = tuple(sys.argv[1:]) if len(sys.argv) > 1 else None
    main(selected)