"""Convert WORC lesion patches into a glovita generic_3d_dataset.

Defaults (relative to the repo root):

- patches: ``metadata/worc_lesion_patches_<dataset>.csv``
- splits: ``crossvalidationsplits/crossvalidationsplits_<dataset>.csv``
- output: ``data/glovita/<dataset>``
- project root: repo root (prefix for relative ``image_path`` entries)

Example:

    python scripts/prepare_Glovita.py --dataset Melanoma
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import blosc2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import StratifiedGroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("Lipo", "Desmoid", "Liver", "GIST", "CRLM", "Melanoma")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert WORC patch metadata and official splits into glovita "
            "generic_3d_dataset format."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS,
        help="WORC cohort name used to resolve default input/output paths.",
    )
    parser.add_argument(
        "--patches-csv",
        type=Path,
        default=None,
        help=(
            "Lesion patch metadata CSV "
            "(default: metadata/worc_lesion_patches_<dataset>.csv)."
        ),
    )
    parser.add_argument(
        "--splits-csv",
        type=Path,
        default=None,
        help=(
            "Official WORC splits CSV "
            "(default: crossvalidationsplits/crossvalidationsplits_<dataset>.csv)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Root directory for images/, dataset.json, labels.json, splits.json "
            "(default: data/glovita/<dataset>)."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "Prefix for relative image_path entries in the patches CSV "
            "(default: repository root)."
        ),
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    dataset = args.dataset
    patches_csv = (
        args.patches_csv.resolve()
        if args.patches_csv is not None
        else REPO_ROOT / "metadata" / f"worc_lesion_patches_{dataset}.csv"
    )
    splits_csv = (
        args.splits_csv.resolve()
        if args.splits_csv is not None
        else (
            REPO_ROOT
            / "crossvalidationsplits"
            / f"crossvalidationsplits_{dataset}.csv"
        )
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else REPO_ROOT / "data" / "glovita" / dataset
    )
    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else REPO_ROOT
    )
    if not patches_csv.is_file():
        raise SystemExit(f"Patches CSV not found: {patches_csv}")
    if not splits_csv.is_file():
        raise SystemExit(f"Splits CSV not found: {splits_csv}")
    return {
        "patches_csv": patches_csv,
        "splits_csv": splits_csv,
        "output_dir": output_dir,
        "project_root": project_root,
    }


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    patches_csv = paths["patches_csv"]
    splits_csv = paths["splits_csv"]
    output_dir = paths["output_dir"]
    project_root = paths["project_root"]

    images_out_dir = output_dir / "images"
    images_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset:      {args.dataset}")
    print(f"Patches CSV:  {patches_csv}")
    print(f"Splits CSV:   {splits_csv}")
    print(f"Output:       {output_dir}")
    print(f"Project root: {project_root}")

    print(f"Loading patch metadata from {patches_csv}...")
    df = pd.read_csv(patches_csv)
    required = {"patient_id", "lesion_id", "image_path", "label"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"Patches CSV missing columns: {sorted(missing)}"
        )
    df["case_id"] = (
        df["patient_id"].astype(str) + "__" + df["lesion_id"].astype(str)
    )

    print(f"Converting NIfTI patches to blosc2 (.b2nd) in {images_out_dir}...")
    for _, row in df.iterrows():
        case_id = row["case_id"]
        nifti_path = Path(row["image_path"])
        if not nifti_path.is_absolute():
            nifti_path = project_root / nifti_path
        out_b2nd = images_out_dir / f"{case_id}.b2nd"

        if out_b2nd.exists():
            continue

        if not nifti_path.is_file():
            raise SystemExit(f"Missing patch image: {nifti_path}")

        img = sitk.ReadImage(str(nifti_path))
        vol = sitk.GetArrayFromImage(img).astype(np.float32)

        # Normalize using CSV stats if the patch is still in raw intensity units
        if abs(float(vol.mean())) > 5.0 or float(vol.std()) > 5.0:
            mean = float(row["normalization_mean"])
            std = float(row["normalization_std"])
            vol = (vol - mean) / max(std, 1e-8)

        if vol.ndim == 3:
            vol = vol[np.newaxis, ...]

        blosc2.asarray(
            np.ascontiguousarray(vol, dtype=np.float32),
            urlpath=str(out_b2nd),
            mode="w",
        )

    dataset_json = {
        "num_classes": int(df["label"].nunique()),
        "subtask": "multiclass",
    }
    (output_dir / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8"
    )

    labels_dict = dict(zip(df["case_id"], df["label"].astype(int)))
    (output_dir / "labels.json").write_text(
        json.dumps(labels_dict, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Generating splits from {splits_csv}...")
    splits_df = pd.read_csv(splits_csv, index_col=0)
    patient_to_cases = df.groupby("patient_id")["case_id"].apply(list).to_dict()
    patient_to_label = df.groupby("patient_id")["label"].first().to_dict()

    num_splits = len(
        [col for col in splits_df.columns if col.endswith("_train")]
    )
    splits = []

    for i in range(num_splits):
        train_patients = splits_df[f"{i}_train"].dropna().astype(str).tolist()
        test_patients = splits_df[f"{i}_test"].dropna().astype(str).tolist()

        train_patients = [p for p in train_patients if p in patient_to_cases]
        test_patients = [p for p in test_patients if p in patient_to_cases]

        train_labels = [patient_to_label[p] for p in train_patients]
        sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42 + i)
        tr_idx, val_idx = next(
            sgkf.split(train_patients, train_labels, train_patients)
        )

        actual_train_patients = [train_patients[idx] for idx in tr_idx]
        val_patients = [train_patients[idx] for idx in val_idx]

        train_cases = [
            c for p in actual_train_patients for c in patient_to_cases[p]
        ]
        val_cases = [c for p in val_patients for c in patient_to_cases[p]]
        test_cases = [c for p in test_patients for c in patient_to_cases[p]]

        splits.append(
            {
                "train": train_cases,
                "val": val_cases,
                "test": test_cases,
            }
        )

    (output_dir / "splits.json").write_text(
        json.dumps(splits, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Successfully prepared dataset in {output_dir} "
        f"with {len(splits)} folds."
    )


if __name__ == "__main__":
    main()
