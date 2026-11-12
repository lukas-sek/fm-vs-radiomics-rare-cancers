"""Build a glovita generic_3d_dataset from the MLP's nnU-Net-preprocessed crops.

Every sample is exactly the input the frozen-feature MLP pipeline feeds to
nnFoundationViT (``nnVIT_nnunet_feature_extraction.py`` without augmentation):
the ``nnUNetv2_preprocess_like_nnssl`` volume cropped to the plan patch size
around each connected lesion component, with zero padding at the border.

Does **not** require ``nnunetv2`` — it reads the plans JSON and ``.b2nd``
files directly, so it runs in the GloViTa env as well as the extraction env.

Example:

    python scripts/prepare_Glovita_nnunet.py --dataset Liver
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

import blosc2
import numpy as np
import pandas as pd
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prepare_Glovita import build_splits  # noqa: E402

DATASETS = ("Lipo", "Desmoid", "Liver", "GIST", "CRLM", "Melanoma")
DEFAULT_DATASET_IDS = {
    "Melanoma": 901,
    "GIST": 902,
    "CRLM": 903,
    "Liver": 904,
    "Desmoid": 905,
    "Lipo": 906,
}
VIT_PLANS_GLOB = "ptPlans__nnFoundationViT____*.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write the MLP pipeline's nnU-Net-preprocessed lesion crops as a "
            "glovita generic_3d_dataset (no nnunetv2 required)."
        )
    )
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=None,
        help="Override nnUNet_preprocessed (default: $nnUNet_preprocessed or data/nnUNet_preprocessed).",
    )
    parser.add_argument("--dataset-id", type=int, default=None)
    parser.add_argument("-c", "--configuration", default="3d_fullres")
    parser.add_argument("--splits-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-component-voxels", type=int, default=1)
    parser.add_argument(
        "--empty-mask",
        choices=("error", "skip", "center"),
        default="error",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def nnunet_dataset_name(cohort: str, dataset_id: int | None = None) -> str:
    resolved_id = DEFAULT_DATASET_IDS[cohort] if dataset_id is None else dataset_id
    return f"Dataset{resolved_id:03d}_WORC_{cohort}"


def resolve_preprocessed_root(override: Path | None) -> Path:
    if override is not None:
        root = override.expanduser().resolve()
    else:
        configured = os.environ.get("nnUNet_preprocessed")
        root = (
            Path(configured).expanduser().resolve()
            if configured
            else (REPO_ROOT / "data" / "nnUNet_preprocessed").resolve()
        )
    if not root.is_dir():
        raise SystemExit(f"nnUNet_preprocessed directory not found: {root}")
    return root


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected a JSON object in {path}")
    return payload


def find_vit_plans(dataset_root: Path) -> Path:
    candidates = sorted(dataset_root.glob(VIT_PLANS_GLOB))
    if len(candidates) != 1:
        raise SystemExit(
            "Expected exactly one nnFoundationViT plan under "
            f"{dataset_root}, found {[path.name for path in candidates]}"
        )
    return candidates[0]


def plan_geometry(plans: dict[str, Any], configuration: str) -> tuple[str, tuple[int, int, int], tuple[float, float, float]]:
    configs = plans.get("configurations") or {}
    if configuration not in configs:
        raise SystemExit(
            f"Configuration {configuration!r} missing from plans; "
            f"available: {sorted(configs)}"
        )
    cfg = configs[configuration]
    data_identifier = str(cfg["data_identifier"])
    patch_size = tuple(int(size) for size in cfg["patch_size"])
    spacing = tuple(float(value) for value in cfg["spacing"])
    if len(patch_size) != 3 or len(spacing) != 3:
        raise SystemExit(f"Expected 3D patch_size/spacing, got {patch_size}/{spacing}")
    return data_identifier, patch_size, spacing


def load_b2nd(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    array = np.asarray(blosc2.open(str(path))[:])
    return array


def list_case_ids(data_folder: Path) -> list[str]:
    """Case IDs that have both image and segmentation ``.b2nd`` files."""
    images = {path.name.removesuffix(".b2nd") for path in data_folder.glob("*.b2nd") if not path.name.endswith("_seg.b2nd")}
    segs = {
        path.name.removesuffix("_seg.b2nd")
        for path in data_folder.glob("*_seg.b2nd")
    }
    case_ids = sorted(images & segs)
    if not case_ids:
        raise SystemExit(f"No image+seg .b2nd pairs found in {data_folder}")
    return case_ids


def component_centers(
    segmentation: np.ndarray,
    min_component_voxels: int = 1,
) -> Iterator[tuple[int, int, tuple[int, int, int], int]]:
    seg = np.asarray(segmentation)
    if seg.ndim == 4:
        if seg.shape[0] != 1:
            raise ValueError(f"Expected one segmentation channel, got shape {seg.shape}")
        seg = seg[0]
    if seg.ndim != 3:
        raise ValueError(f"Expected a 3D segmentation, got shape {seg.shape}")
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    for raw_label in np.unique(seg[seg > 0]):
        label_value = int(raw_label)
        components, count = ndimage.label(seg == raw_label, structure=structure)
        for component_id in range(1, count + 1):
            coordinates = np.argwhere(components == component_id)
            voxel_count = int(coordinates.shape[0])
            if voxel_count < min_component_voxels:
                continue
            center = tuple(int(np.rint(value)) for value in coordinates.mean(axis=0))
            yield label_value, component_id, center, voxel_count


def crop_or_pad(
    data: np.ndarray,
    center: Sequence[int],
    patch_size: Sequence[int],
    padding_value: float | int = 0,
    dtype: np.dtype[Any] | type[np.generic] = np.float32,
) -> np.ndarray:
    array = np.asarray(data)
    if array.ndim != 4:
        raise ValueError(f"Expected data shaped (C, X, Y, Z), got {array.shape}")
    center = tuple(int(value) for value in center)
    patch_size = tuple(int(value) for value in patch_size)
    output = np.full((array.shape[0], *patch_size), fill_value=padding_value, dtype=dtype)
    source_slices: list[slice] = [slice(None)]
    target_slices: list[slice] = [slice(None)]
    for axis_center, requested, available in zip(center, patch_size, array.shape[1:]):
        start = axis_center - requested // 2
        stop = start + requested
        source_start = max(start, 0)
        source_stop = min(stop, available)
        if source_start >= source_stop:
            raise ValueError(f"Center {center} lies outside data shape {array.shape[1:]}")
        target_start = source_start - start
        target_stop = target_start + (source_stop - source_start)
        source_slices.append(slice(source_start, source_stop))
        target_slices.append(slice(target_start, target_stop))
    output[tuple(target_slices)] = array[tuple(source_slices)]
    if not np.isfinite(output).all():
        raise ValueError("Input patch contains non-finite values")
    return output


def ensure_channels_first(array: np.ndarray) -> np.ndarray:
    """nnU-Net stores (C, X, Y, Z); accept a bare 3D volume as one channel."""
    if array.ndim == 3:
        return array[np.newaxis, ...]
    if array.ndim == 4:
        return array
    raise ValueError(f"Expected 3D or 4D array, got shape {array.shape}")


def glovita_case_id(
    patient_id: str, lesion_id: str, component_index: int, n_components: int
) -> str:
    base = f"{patient_id}__{lesion_id}"
    return base if n_components == 1 else f"{base}__c{component_index}"


def main() -> None:
    args = parse_args()
    if args.min_component_voxels < 1:
        raise SystemExit("--min-component-voxels must be at least 1")

    dataset_name = nnunet_dataset_name(args.dataset, args.dataset_id)
    dataset_root = resolve_preprocessed_root(args.preprocessed_root) / dataset_name
    metadata_path = dataset_root / "worc_case_metadata.csv"
    if not metadata_path.is_file():
        raise SystemExit(
            f"{metadata_path} not found; run nnVIT_nnunet_feature_extraction.py "
            f"--dataset {args.dataset} first."
        )

    splits_csv = (
        args.splits_csv.resolve()
        if args.splits_csv is not None
        else REPO_ROOT / "crossvalidationsplits" / f"crossvalidationsplits_{args.dataset}.csv"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else REPO_ROOT / "data" / "glovita_worc_nnunet" / args.dataset
    )
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    plans_path = find_vit_plans(dataset_root)
    data_identifier, patch_size, spacing = plan_geometry(
        load_json(plans_path), args.configuration
    )
    data_folder = dataset_root / data_identifier
    if not data_folder.is_dir():
        raise SystemExit(f"Preprocessed configuration directory not found: {data_folder}")

    metadata = pd.read_csv(metadata_path, dtype={"patient_id": str, "lesion_id": str})
    metadata = metadata.set_index("case_id")
    case_ids = list_case_ids(data_folder)

    print(f"Dataset:      {args.dataset} ({dataset_name})")
    print(f"Plans:        {plans_path}")
    print(f"Data:         {data_folder}")
    print(f"Patch size:   {patch_size} (plan spacing {spacing})")
    print(f"Splits CSV:   {splits_csv}")
    print(f"Output:       {output_dir}")
    print(f"Cases:        {len(case_ids)}")

    rows = []
    for number, nnunet_case_id in enumerate(case_ids, start=1):
        if nnunet_case_id not in metadata.index:
            raise SystemExit(f"{nnunet_case_id} missing from {metadata_path}")
        case = metadata.loc[nnunet_case_id]
        data = ensure_channels_first(
            load_b2nd(data_folder / f"{nnunet_case_id}.b2nd")
        ).astype(np.float32, copy=False)
        segmentation = ensure_channels_first(
            load_b2nd(data_folder / f"{nnunet_case_id}_seg.b2nd")
        )
        components = list(component_centers(segmentation, args.min_component_voxels))
        if not components:
            if args.empty_mask == "error":
                raise SystemExit(f"{nnunet_case_id}: segmentation has no foreground")
            if args.empty_mask == "skip":
                print(f"\nSkipping {nnunet_case_id}: segmentation has no foreground")
                continue
            components = [(0, 0, tuple(int(size // 2) for size in data.shape[1:]), 0)]

        for index, (_, component_id, center, voxel_count) in enumerate(components, start=1):
            case_id = glovita_case_id(
                str(case["patient_id"]), str(case["lesion_id"]), index, len(components)
            )
            out_path = images_dir / f"{case_id}.b2nd"
            if args.overwrite or not out_path.exists():
                if out_path.exists():
                    out_path.unlink()
                blosc2.asarray(
                    np.ascontiguousarray(crop_or_pad(data, center, patch_size)),
                    urlpath=str(out_path),
                    mode="w",
                )
            rows.append(
                {
                    "case_id": case_id,
                    "patient_id": str(case["patient_id"]),
                    "lesion_id": str(case["lesion_id"]),
                    "label": int(case["label"]),
                    "nnunet_case_id": nnunet_case_id,
                    "component_id": component_id,
                    "component_voxels": voxel_count,
                    "center_x": center[0],
                    "center_y": center[1],
                    "center_z": center[2],
                }
            )
        print(f"\rWrote {number}/{len(case_ids)} cases", end="", flush=True)
    print()

    cases = pd.DataFrame(rows)
    if cases["case_id"].duplicated().any():
        duplicates = cases.loc[cases["case_id"].duplicated(), "case_id"].tolist()
        raise SystemExit(f"Duplicate glovita case ids: {duplicates[:5]}")
    cases.to_csv(output_dir / "cases.csv", index=False)
    (output_dir / "dataset.json").write_text(
        json.dumps(
            {"num_classes": int(cases["label"].nunique()), "subtask": "multiclass"},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "labels.json").write_text(
        json.dumps(dict(zip(cases["case_id"], cases["label"].astype(int))), indent=2)
        + "\n",
        encoding="utf-8",
    )
    splits = build_splits(splits_csv, cases)
    (output_dir / "splits.json").write_text(
        json.dumps(splits, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "preprocessing.json").write_text(
        json.dumps(
            {
                "source": "nnUNetv2_preprocess_like_nnssl",
                "plans": str(plans_path),
                "configuration": args.configuration,
                "data_identifier": data_identifier,
                "spacing": list(spacing),
                "patch_size": list(patch_size),
                "crop": "centered on connected lesion component, zero padded",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Prepared {len(cases)} crops of {cases['patient_id'].nunique()} patients "
        f"with {len(splits)} folds in {output_dir}"
    )


if __name__ == "__main__":
    main()
