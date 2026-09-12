# WORC datasets

This project uses the six datasets in the [WORC Database](https://xnat.health-ri.nl/data/projects/worc), described in [The WORC database: MRI and CT scans, segmentations, and clinical labels for 930 patients from six radiomics studies](https://doi.org/10.1101/2021.08.19.21262238).

The machine-readable definitions are maintained in `metadata/datasets.yaml`. Patient-, lesion-, and segmentation-level observations belong in the corresponding CSV manifests in `metadata/`.

## Lipo

The Lipo dataset contains 115 patients and 116 lesions: 58 well-differentiated liposarcomas (WDLPS) and 58 lipomas. Each patient has a T1-weighted MRI scan. The binary target is MDM2 amplification status: WDLPS is encoded as 1 and lipoma as 0.

One patient has both a WDLPS and a lipoma. Its lesions have separate segmentations (`segmentation_WDLPS.nii.gz` and `segmentation_Lipoma.nii.gz`). Labels must therefore be retained at lesion level, while data splitting must remain at patient level.

## Desmoid

The Desmoid dataset contains 203 patients with T1-weighted MRI: 72 with desmoid-type fibromatosis (DTF) and 131 with extremity soft-tissue sarcoma (non-DTF). The non-DTF cases comprise 64 myxofibrosarcomas, 31 leiomyosarcomas, and 36 myxoid liposarcomas.

Histology established the differential diagnosis. DTF is encoded as 1 and non-DTF as 0.

## Liver

The Liver dataset reports 186 patients with T2-weighted MRI and a malignant-versus-benign primary solid liver tumor target. Malignant tumors are encoded as 1 and benign tumors as 0.

The reported malignant group contains 81 hepatocellular carcinomas and 13 intrahepatic cholangiocarcinomas. The reported benign group contains 48 hepatocellular adenomas and 44 focal nodular hyperplasias (FNH). Pathology established ground truth except for radiologically typical FNH.

The publication reports 94 malignant and 93 benign cases, which sum to 187 rather than the stated 186 patients. It also names 92 benign cases by subtype while reporting 93 benign cases. These discrepancies must be resolved against the downloaded XNAT metadata before analysis.

## GIST

The GIST dataset contains 246 patients and 247 lesions. One patient has two GIST lesions. Each patient has a contrast-enhanced venous-phase CT scan.

There are 125 GIST lesions and 122 non-GIST lesions. The non-GIST group comprises 22 schwannomas and 25 cases each of leiomyosarcoma, leiomyoma, esophageal or gastric junctional adenocarcinoma, and lymphoma. Histology established the target: GIST is encoded as 1 and non-GIST as 0.

## CRLM

The CRLM dataset contains 77 patients and 93 colorectal liver metastases on portal-venous-phase CT. The target is histopathological growth pattern (HGP): 47 lesions have 100% replacement HGP, encoded as 1, and 46 have 100% desmoplastic HGP, encoded as 0.

Ground truth is assigned at patient level because HGP is assumed to be identical for all lesions of a patient. Each lesion may have segmentations from STUD1, STUD2, PhD, RAD, and CNN. STUD2 is a repeat segmentation by the first observer. The CNN missed 8 of 93 lesions, so missing CNN segmentations must remain explicitly missing rather than being interpreted as negative labels.

## Melanoma

The Melanoma dataset contains 169 lung metastases from 103 patients with contrast-enhanced thoracic CT. Of these patients, 51 are BRAF mutated and 52 are BRAF wild type. BRAF mutated is encoded as 1 and wild type as 0.

Mutation status is assigned at patient level and shared by all lesions from that patient.

## Splitting and provenance

The original cross-validation files in `crossvalidationsplits/` must be preserved unchanged. Every lesion from the same patient must remain in the same partition to avoid leakage.

All file paths in manifests should be relative to the project data root. Record observed counts from XNAT separately from the publication's reported counts, and do not silently correct source discrepancies.

The XNAT project states that use is non-commercial, redistribution is prohibited, and the relevant WORC papers must be cited. Consult the complete data usage agreement before publishing or sharing derived artifacts.
