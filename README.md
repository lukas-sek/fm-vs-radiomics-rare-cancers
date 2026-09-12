# Foundation Models vs. Radiomics for Rare Cancers

This repository contains experiments comparing foundation models with radiomics methods on the six public WORC datasets.

## Data and metadata

- `docs/datasets.md` documents the cohorts, targets, modalities, and known count discrepancies.
- `metadata/datasets.yaml` contains machine-readable dataset-level definitions.
- `metadata/subjects.csv`, `metadata/lesions.csv`, and `metadata/segmentations.csv` are normalized patient-level manifest templates.
- `metadata/data_dictionary.md` defines the manifest columns and validation rules.
- `crossvalidationsplits/` contains the original WORC patient-level splits and should remain unchanged.

Imaging data is stored locally under `data/` and is excluded from Git. Paths recorded in manifests should be relative to that data root.

Before using or publishing results from the data, consult the WORC data usage agreement. The XNAT project states that use is non-commercial, redistribution is prohibited, and the relevant WORC papers must be cited.