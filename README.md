# Foundation Models vs. Radiomics for Rare Cancers

This repository contains experiments comparing foundation models with radiomics methods on the six public WORC datasets.

## Setup

Requires Python `>=3.9,<3.12` (FMCIB does not support 3.12+):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data and metadata

- `docs/datasets.md` documents the cohorts, targets, modalities, and known count discrepancies.
- `metadata/datasets.yaml` contains machine-readable dataset-level definitions.
- `metadata/pinfo_*.csv` stores per-dataset patient labels.
- `metadata/fmcib_full_lesion_*.csv` and `metadata/fmcib_full_image_*.csv` are FMCIB feature-extraction CSVs for full-lesion and full-image crops.
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
- `scripts/analyze_worc_original_spacing.py` — scans original WORC NIfTIs and writes `metadata/worc_original_spacing.csv` with console summaries.

## Notebooks

- `Voxel_spacing_analysis_worc.ipynb` — voxel spacing and matrix-dimension analysis with summary table and plots.
- `worc_lesion_volume_axis_analysis.ipynb` — lesion volume density and axis-extent analysis (including >50 mm counts).
- `get_seed_from_mask.ipynb` — prototype for seed extraction / full-lesion crop logic.


