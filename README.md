# Foundation Models vs. Radiomics for Rare Cancers

This repository contains experiments comparing foundation models with radiomics methods on the six public WORC datasets.

## Setup

Requires Python `>=3.10,<3.12` (nnU-Net requires 3.10+ and FMCIB does not
support 3.12+):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data and metadata

- `docs/datasets.md` documents the cohorts, targets, modalities, and known count discrepancies.
- `metadata/datasets.yaml` contains machine-readable dataset-level definitions.
- `metadata/pinfo_*.csv` stores per-dataset patient labels.
- `metadata/fmcib_full_lesion_*.csv` and `metadata/fmcib_full_image_*.csv` are FMCIB feature-extraction CSVs for full-lesion and full-image crops (`precropped=True`).
- `metadata/fmcib_original_Melanoma.csv` points at original WORC Melanoma images with CoM seeds (`get_features(..., precropped=False)`).
- `crossvalidationsplits/` contains the original WORC patient-level splits and should remain unchanged.

Imaging data is stored locally under `data/` (for example `data/worc/{Dataset}/{PatientID}/`) and is excluded from Git. Paths recorded in manifests should be relative to the repository root.

### Original spacing & matrix dimensions

Native voxel spacing and matrix sizes of the original WORC images (not FMCIB crops):

| Artifact | Description |
| --- | --- |
| `metadata/worc_original_spacing.csv` | Per-image spacing (`x/y/z` mm), size, and voxel volume |
| `metadata/worc_spacing_dimension_summary.csv` | Per-dataset median / min–max summary table |
| `metadata/worc_spacing_distribution.png` | Boxplot of original spacing per axis |
| `metadata/worc_dimensions_distribution.png` | Boxplot of matrix dimensions per axis |

Rebuild with either:

```powershell
.\.venv\Scripts\python.exe scripts/analyze_worc_original_spacing.py
# optional subset: ... CRLM Melanoma
```

or by running `Voxel_spacing_analysis_worc.ipynb` (writes the same CSV/PNG outputs under `metadata/`).

## Scripts

- `scripts/build_fmcib_full_lesion_csvs.py` — builds full-lesion `50³` crops under `data/fmcib_full_lesion/` and corresponding FMCIB CSVs.
- `scripts/build_fmcib_original_melanoma_csv.py` — Melanoma CSV with original image paths and CoM seeds (`metadata/fmcib_original_Melanoma.csv`).
- `scripts/analyze_worc_original_spacing.py` — scans original WORC NIfTIs and writes `metadata/worc_original_spacing.csv` with console summaries.
- `experiments/nnFoundation/nnCNN_feature_extraction.py` — prepares a named
  WORC cohort as a lesion-per-case nnU-Net dataset, runs nnssl preprocessing
  when needed, and extracts one pooled nnFoundationCNN embedding per lesion:

```powershell
python experiments/nnFoundation/nnCNN_feature_extraction.py `
  --dataset Melanoma `
  --num-augmentations 3
```

The source is read from `data/worc/Melanoma`. Generated nnU-Net raw and
preprocessed data default to `data/nnUNet_raw` and
`data/nnUNet_preprocessed`; configured nnU-Net environment paths take
precedence. Existing `DatasetXXX_Name` inputs remain supported but require
`--plans`.

`--num-augmentations 3` writes the original embedding plus three reproducible
variants generated with nnU-Net's default training transforms. Use augmented
rows only for training patients; do not include them in validation or test
folds. Omit the option for unaugmented extraction. The default output for this
example is `Results/nnFoundation/Melanoma_nncnn_features.csv`; pass
`--output-csv` only to override it.

### nnFoundation MLP classification

Train the MLP over the original WORC patient splits for CNN (`320 → 128 → 1`)
or ViT (`1056 → 128 → 1`) embeddings:

```powershell
python experiments/nnFoundation/MLP_nn_classification.py `
  --dataset Melanoma `
  --backbone CNN
```

Use `--backbone ViT` to read `*_nnvit_*.csv` instead of `*_nncnn_*.csv`.
By default, training reads the augmented feature CSV, while held-out
evaluation reads only the original feature CSV. Paths and training parameters
can be overridden through CLI options or
`experiments/nnFoundation/mlp_CNN.yaml` (or `mlp_ViT.yaml` for ViT).

The classifier writes these files under `Results/nnFoundation` (CNN example):

- `Melanoma_nncnn_mlp_cv_per_split.csv`
- `Melanoma_nncnn_mlp_cv_summary.csv`
- `Melanoma_nncnn_mlp_cv_patient_predictions.csv`

The same command supports `Lipo`, `Desmoid`, `Liver`, `GIST`, and `CRLM`.
Feature scaling is fitted within each training fold, augmented rows are
patient-balanced during optimization, and lesion predictions are averaged per
patient before AUROC and average precision are calculated.

## Notebooks

- `Voxel_spacing_analysis_worc.ipynb` — voxel spacing and matrix-dimension analysis with summary table and plots.
- `worc_lesion_volume_axis_analysis.ipynb` — lesion volume density and axis-extent analysis (including >50 mm counts).
- `get_seed_from_mask.ipynb` — prototype for seed extraction / full-lesion crop logic.


