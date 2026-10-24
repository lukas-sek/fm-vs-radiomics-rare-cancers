"""Convert one WORC cohort into a lesion-per-case nnU-Net raw dataset."""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import SimpleITK as sitk

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_nnFoundation_lesion_patches import (  # noqa: E402
    DATASET_MODALITY,
    DATASETS,
    discover_lesion_masks,
    get_clean_lesion_mask,
    load_pinfo,
    resolve_image_path,
    resolve_label,
)

DEFAULT_DATASET_IDS = {
    "Melanoma": 901,
    "GIST": 902,
    "CRLM": 903,
    "Liver": 904,
    "Desmoid": 905,
    "Lipo": 906,
}


def nnunet_dataset_name(cohort: str, dataset_id: int | None = None) -> str:
    if cohort not in DATASETS:
        raise ValueError(f"Unknown WORC cohort {cohort!r}; choose from {DATASETS}")
    resolved_id = DEFAULT_DATASET_IDS[cohort] if dataset_id is None else dataset_id
    if not 1 <= resolved_id <= 999:
        raise ValueError("nnU-Net dataset ID must be between 1 and 999")
    return f"Dataset{resolved_id:03d}_WORC_{cohort}"


def safe_identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-")
    if not cleaned:
        raise ValueError(f"Cannot create a case identifier from {value!r}")
    return cleaned


def geometry_matches(image: sitk.Image, mask: sitk.Image) -> bool:
    return (
        image.GetDimension() == mask.GetDimension() == 3
        and image.GetSize() == mask.GetSize()
        and all(
            abs(a - b) <= 1e-6
            for a, b in zip(image.GetSpacing(), mask.GetSpacing())
        )
        and all(
            abs(a - b) <= 1e-5
            for a, b in zip(image.GetOrigin(), mask.GetOrigin())
        )
        and all(
            abs(a - b) <= 1e-6
            for a, b in zip(image.GetDirection(), mask.GetDirection())
        )
    )


def same_voxel_grid(image: sitk.Image, mask: sitk.Image) -> bool:
    """True when image and mask share the same array size (header may differ)."""
    return (
        image.GetDimension() == mask.GetDimension() == 3
        and image.GetSize() == mask.GetSize()
    )


def geometry_summary(image: sitk.Image) -> str:
    return (
        f"size={image.GetSize()} spacing={image.GetSpacing()} "
        f"origin={image.GetOrigin()} direction={image.GetDirection()}"
    )


def copy_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_binary_mask(
    mask: sitk.Image,
    destination: Path,
    reference_image: sitk.Image | None = None,
) -> None:
    cleaned, label = get_clean_lesion_mask(mask)
    binary = sitk.Cast(sitk.Equal(cleaned, label), sitk.sitkUInt8)
    if reference_image is not None and not geometry_matches(reference_image, binary):
        if not same_voxel_grid(reference_image, binary):
            raise ValueError(
                "Cannot align mask: image and mask have different sizes.\n"
                f"  image: {geometry_summary(reference_image)}\n"
                f"  mask:  {geometry_summary(binary)}"
            )
        # Copy array onto the image header so nnU-Net gets exact matching geometry.
        # Same size + nearest-neighbor identity transform keeps labels unchanged
        # for WORC cases where only NIfTI headers differ by floating-point noise.
        binary = sitk.Resample(
            binary,
            reference_image,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(binary, str(destination), useCompression=True)


def existing_preparation(
    dataset_dir: Path,
    metadata_path: Path,
) -> pd.DataFrame | None:
    dataset_json = dataset_dir / "dataset.json"
    if not dataset_json.is_file() or not metadata_path.is_file():
        return None
    metadata = pd.read_csv(metadata_path)
    if metadata.empty:
        return None
    expected = {
        dataset_dir / "imagesTr" / f"{case_id}_0000.nii.gz"
        for case_id in metadata["case_id"].astype(str)
    }
    expected.update(
        dataset_dir / "labelsTr" / f"{case_id}.nii.gz"
        for case_id in metadata["case_id"].astype(str)
    )
    return metadata if all(path.is_file() for path in expected) else None


def prepare_worc_nnunet_dataset(
    cohort: str,
    worc_root: Path,
    nnunet_raw_root: Path,
    nnunet_preprocessed_root: Path,
    dataset_id: int | None = None,
    overwrite: bool = False,
) -> tuple[str, pd.DataFrame]:
    """Create imagesTr, labelsTr, dataset.json and persistent case metadata."""
    dataset_name = nnunet_dataset_name(cohort, dataset_id)
    source_root = worc_root / cohort
    if not source_root.is_dir():
        raise FileNotFoundError(f"WORC cohort directory not found: {source_root}")

    dataset_dir = nnunet_raw_root / dataset_name
    preprocessed_dir = nnunet_preprocessed_root / dataset_name
    metadata_path = preprocessed_dir / "worc_case_metadata.csv"
    if not overwrite:
        existing = existing_preparation(dataset_dir, metadata_path)
        if existing is not None:
            return dataset_name, existing

    if overwrite and dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    if overwrite and preprocessed_dir.exists():
        shutil.rmtree(preprocessed_dir)
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    pinfo = load_pinfo(cohort)
    rows: list[dict[str, Any]] = []
    patient_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        for lesion_number, (lesion_id, mask_path, mask_source) in enumerate(
            discover_lesion_masks(cohort, patient_dir),
            start=1,
        ):
            image_path = resolve_image_path(patient_dir, lesion_id).resolve()
            image = sitk.ReadImage(str(image_path))
            mask = sitk.ReadImage(str(mask_path))
            if not same_voxel_grid(image, mask):
                raise ValueError(
                    f"Image/mask size mismatch for {patient_id} {lesion_id}\n"
                    f"  image: {geometry_summary(image)}\n"
                    f"  mask:  {geometry_summary(mask)}"
                )

            case_id = (
                f"{safe_identifier(patient_id)}"
                f"--{safe_identifier(lesion_id)}--{lesion_number}"
            )
            copy_image(image_path, images_dir / f"{case_id}_0000.nii.gz")
            write_binary_mask(
                mask,
                labels_dir / f"{case_id}.nii.gz",
                reference_image=image,
            )
            rows.append(
                {
                    "case_id": case_id,
                    "dataset": cohort,
                    "patient_id": patient_id,
                    "lesion_id": lesion_id,
                    "label": int(
                        resolve_label(cohort, patient_id, lesion_id, pinfo)
                    ),
                    "mask_source": mask_source or "",
                    "source_image_path": str(image_path),
                    "source_mask_path": str(mask_path.resolve()),
                }
            )

    if not rows:
        raise RuntimeError(f"No lesion cases found under {source_root}")

    dataset_json = {
        "channel_names": {"0": DATASET_MODALITY[cohort]},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(rows),
        "file_ending": ".nii.gz",
    }
    with (dataset_dir / "dataset.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_json, handle, indent=2)
        handle.write("\n")

    metadata = pd.DataFrame(rows)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_path, index=False)
    return dataset_name, metadata
