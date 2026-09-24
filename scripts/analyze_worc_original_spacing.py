"""Analyze native voxel spacing of original WORC NIfTI images.

Reads SimpleITK GetSpacing()/GetSize() from unique image.nii.gz files under
data/worc (not FMCIB resampled crops). Writes metadata/worc_original_spacing.csv
and prints per-dataset min/median/max summaries.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import SimpleITK as sitk

# Allow importing discovery helpers from the sibling build script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_fmcib_full_lesion_csvs import (  # noqa: E402
    DATASETS,
    WORC_ROOT,
    discover_lesion_masks,
    resolve_image_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = REPO_ROOT / "metadata" / "worc_original_spacing.csv"


def collect_dataset_rows(dataset: str) -> list[dict]:
    ds_root = WORC_ROOT / dataset
    if not ds_root.exists():
        raise FileNotFoundError(f"Missing dataset directory: {ds_root}")

    patient_dirs = sorted(p for p in ds_root.iterdir() if p.is_dir())
    seen_images: set[Path] = set()
    rows: list[dict] = []

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        lesions = discover_lesion_masks(dataset, patient_dir)

        for lesion_id, _mask_path, _seg_source in lesions:
            image_path = resolve_image_path(patient_dir, lesion_id).resolve()
            if image_path in seen_images:
                continue
            seen_images.add(image_path)

            image = sitk.ReadImage(str(image_path))
            sx, sy, sz = image.GetSpacing()
            size_x, size_y, size_z = image.GetSize()
            try:
                rel_path = image_path.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                rel_path = image_path.as_posix()

            rows.append(
                {
                    "dataset": dataset,
                    "patient_id": patient_id,
                    "image_path": rel_path,
                    "spacing_x_mm": float(sx),
                    "spacing_y_mm": float(sy),
                    "spacing_z_mm": float(sz),
                    "size_x": int(size_x),
                    "size_y": int(size_y),
                    "size_z": int(size_z),
                    "voxel_volume_mm3": float(sx * sy * sz),
                }
            )

    return rows


def _fmt_axis(series: pd.Series) -> str:
    return f"{series.min():.3f} / {series.median():.3f} / {series.max():.3f}"


def print_summary(df: pd.DataFrame) -> None:
    for dataset, g in df.groupby("dataset", sort=False):
        print(f"=== {dataset} ===")
        print(f"n images: {len(g)}")
        print(f"spacing_x_mm min/median/max: {_fmt_axis(g['spacing_x_mm'])}")
        print(f"spacing_y_mm min/median/max: {_fmt_axis(g['spacing_y_mm'])}")
        print(f"spacing_z_mm min/median/max: {_fmt_axis(g['spacing_z_mm'])}")
        in_plane = pd.concat([g["spacing_x_mm"], g["spacing_y_mm"]], ignore_index=True)
        print(f"in-plane (x+y) min/median/max: {_fmt_axis(in_plane)}")
        print(f"slice thickness (z) min/median/max: {_fmt_axis(g['spacing_z_mm'])}")
        print()


def main(datasets: tuple[str, ...] | None = None) -> None:
    if not WORC_ROOT.exists():
        raise SystemExit(f"Missing WORC data root: {WORC_ROOT}")

    selected = datasets or DATASETS
    all_rows: list[dict] = []
    for ds in selected:
        print(f"Scanning {ds}...")
        all_rows.extend(collect_dataset_rows(ds))

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise SystemExit("No images found.")

    # Stable dataset order matching DATASETS when present.
    order = {name: i for i, name in enumerate(DATASETS)}
    df = df.sort_values(
        by=["dataset", "patient_id"],
        key=lambda col: col.map(order) if col.name == "dataset" else col,
    ).reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows -> {OUT_CSV}\n")
    print_summary(df)


if __name__ == "__main__":
    selected = tuple(sys.argv[1:]) if len(sys.argv) > 1 else None
    main(selected)
