"""Build Melanoma FMCIB CSV with original WORC image paths and CoM seeds.

Seeds are physical-space centers of mass from LabelShapeStatisticsImageFilter
(GetCentroid), matching the FMCIB notebook — not bbox centers used in the
full-lesion crop CSVs.

Columns include patient_id and lesion_id (required by extract_fmcib_patches.py).

Inference on native volumes: get_features(csv_path, precropped=False)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_fmcib_full_lesion_csvs import (  # noqa: E402
    EXPECTED_LESIONS,
    WORC_ROOT,
    discover_lesion_masks,
    get_centroid,
    load_pinfo,
    resolve_image_path,
    resolve_label,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = "Melanoma"
OUT_CSV = REPO_ROOT / "fmcib_CSVs" / "fmcib_original_Melanoma.csv"


def build_rows() -> list[dict]:
    ds_root = WORC_ROOT / DATASET
    if not ds_root.exists():
        raise FileNotFoundError(f"Missing dataset directory: {ds_root}")

    pinfo = load_pinfo(DATASET)
    patient_dirs = sorted(p for p in ds_root.iterdir() if p.is_dir())
    rows: list[dict] = []

    for i, patient_dir in enumerate(patient_dirs, start=1):
        patient_id = patient_dir.name
        lesions = discover_lesion_masks(DATASET, patient_dir)

        for lesion_id, mask_path, seg_source in lesions:
            image_path = resolve_image_path(patient_dir, lesion_id)
            mask = sitk.ReadImage(str(mask_path))
            cx, cy, cz = get_centroid(mask)
            label = resolve_label(DATASET, patient_id, lesion_id, pinfo)

            try:
                rel_path = image_path.resolve().relative_to(REPO_ROOT).as_posix()
            except ValueError:
                rel_path = image_path.as_posix()

            if not image_path.exists():
                raise FileNotFoundError(f"Missing image: {image_path}")

            rows.append(
                {
                    "patient_id": patient_id,
                    "lesion_id": lesion_id,
                    "image_path": rel_path,
                    "coordX": cx,
                    "coordY": cy,
                    "coordZ": cz,
                    "label": int(label),
                    "mask_source": seg_source if seg_source is not None else "",
                }
            )

        if i % 25 == 0 or i == len(patient_dirs):
            print(f"  {DATASET}: {i}/{len(patient_dirs)} patients")

    return rows


def main() -> None:
    print(f"Building {DATASET} original-image CoM CSV...")
    rows = build_rows()
    df = pd.DataFrame(rows)

    expected = EXPECTED_LESIONS[DATASET]
    if len(df) != expected:
        raise SystemExit(f"Expected {expected} lesions, got {len(df)}")

    missing = [p for p in df["image_path"] if not (REPO_ROOT / p).exists()]
    if missing:
        raise SystemExit(f"Missing {len(missing)} image paths (e.g. {missing[0]})")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)


if __name__ == "__main__":
    main()
