# Dataset Card: OASIS-2 Visit Pairs with MRI (ML-ready)

## Summary

An ML-ready longitudinal dataset for predicting **next-visit cognitive decline** in older adults. Each row pairs one clinical/MRI visit with the same patient's next visit. Features describe the patient and their MRI scan at the earlier visit; the binary target `declined` is 1 when the Clinical Dementia Rating (CDR) increases at the next visit. Every row links to the raw structural MRI scans of both visits through `mri_id`.

| Property | Value |
|---|---|
| Version | 2.0 (Checkpoint 2: clinical metadata + MRI scans) |
| Instances | 223 visit pairs from 150 patients |
| Features | 20 model inputs: 19 clinical metadata features + 1 MRI scan measure (see Fields) |
| MRI quality columns | 4 audit columns (scan count, SNR, repeat-scan agreement, QC flag); not model inputs |
| Target | `declined` (0 = stable, 1 = declined); 31 positives (13.9%) |
| Splits | Patient-level train (178 pairs, 119 patients) / test (45 pairs, 31 patients) + 5 grouped CV folds in train |
| Linked MRI | 1,367 raw T1 scans in 373 sessions; all 223 pairs have scans at both visits |
| Files | `data/processed/oasis2_visit_pairs.csv`, `oasis2_mri_sessions.csv`, `oasis2_mri_scans.csv` (+ metadata JSON for each build) |
| Curators | Christina Dames, Cinthia Ochoa Torre, Rayna Alexandre-Petrie (DSCI 4350, Group 5) |
| Shared at | GitHub: `data/processed/`; Google Drive: <ADD SHARED LINK> |

## Motivation and Intended Use

- **Purpose:** compare how well MRI scans, patient metadata, and their combination predict whether a patient's CDR will increase by the next visit.
- **Intended users:** the project team and course instructors, for academic model development and evaluation.
- **Out-of-scope uses:** clinical diagnosis, treatment decisions, or any individual-level medical use. The dataset is small, single-site, and not representative of the general population.

## Sources and Lineage

| Stage | Description |
|---|---|
| Original collection | OASIS-2: Open Access Series of Imaging Studies, longitudinal MRI data in nondemented and demented older adults (Marcus et al., 2010). Washington University Knight Alzheimer Disease Research Center. |
| Clinical source | Kaggle dataset "MRI and Alzheimers" (`oasis_longitudinal.csv`): 373 visits x 15 columns, 150 patients, 2 to 5 visits each. Used unmodified. |
| MRI source | OASIS-2 raw scan archives from oasis-brains.org: `OAS2_RAW_PART1.tar.gz` (10.9 GB, 209 sessions) and `OAS2_RAW_PART2.tar.gz` (8.4 GB, 164 sessions). |
| MRI build | `src/data/make_mri_dataset.py`: streams the archives and writes per-scan and per-session tables. |
| Dataset build | `src/data/make_dataset.py`: cleans the CSV, pairs visits, attaches the MRI session measures, splits, and imputes (deterministic, `random_state = 42`). |
| Versioning | Raw-file, archive, and output SHA-256 hashes are stored in `oasis2_visit_pairs_metadata.json` and `oasis2_mri_metadata.json`. |

## Composition

- **Instance:** one visit pair (visit *t* to visit *t+1*) for one patient. Patients with *k* visits contribute *k - 1* pairs.
- **Patients:** 150 (41% male), ages 60 to 96 at the earlier visit (mean 76.1), all right-handed.
- **Baseline CDR of pairs:** 133 at CDR 0, 70 at CDR 0.5, 20 at CDR 1.
- **Target balance:** 192 stable, 31 declined. No patient declined more than once, so the 31 positives come from 31 different patients.

### CDR transitions (rows = CDR at visit t, columns = CDR at visit t+1)

| CDR at t | 0 | 0.5 | 1 | 2 |
|---|---|---|---|---|
| 0 | 118 | 15 | 0 | 0 |
| 0.5 | 3 | 54 | 13 | 0 |
| 1 | 0 | 2 | 15 | 3 |

### MRI scans

| Property | Value |
|---|---|
| Sessions / scans | 373 sessions (one per clinical visit), 1,367 scans; 2 to 4 repeated scans per session |
| Sequence | T1-weighted MPRAGE, sagittal acquisition, 1.5 T |
| Format and grid | Analyze 7.5 (.img + .hdr), big-endian float32; 256 x 256 x 128 voxels at 1 x 1 x 1.25 mm for every scan |
| Size | 19.3 GB compressed, about 45 GB extracted (not stored in the repository) |
| Quality | SNR 6.7 to 36.4 (mean 21.4); 19 sessions flagged for motion (low repeat-scan agreement) |

## Fields (`oasis2_visit_pairs.csv`)

| Column | Group | Description |
|---|---|---|
| `subject_id` | ID | Patient identifier (grouping key; not a feature) |
| `mri_id` | ID | MRI session of the earlier visit; joins to the MRI tables |
| `visit` | ID | Visit number of the earlier visit |
| `age` | Demographics | Age in years at the earlier visit |
| `sex_male` | Demographics | 1 = male, 0 = female |
| `educ` | Demographics | Years of education |
| `ses` | Demographics | Socioeconomic status, 1 (highest) to 5 (lowest) |
| `ses_imputed` | Demographics | 1 if SES was missing and imputed |
| `mmse` | Clinical | Mini-Mental State Examination, 0 to 30 (lower = worse) |
| `cdr` | Clinical | Clinical Dementia Rating at the earlier visit (0, 0.5, 1) |
| `etiv` | MRI (CSV) | Estimated total intracranial volume (cm^3) |
| `nwbv` | MRI (CSV) | Normalized whole-brain volume (fraction of eTIV) |
| `asf` | MRI (CSV) | Atlas scaling factor (about 1/eTIV; r = -0.99 with etiv) |
| `days_since_baseline` | Visit timing | Days since the patient's first visit |
| `days_to_next` | Visit timing | Days until the next visit (prediction horizon) |
| `has_prev_visit` | Visit timing | 1 if the earlier visit has its own previous visit |
| `days_since_prev` | Visit timing | Days since the previous visit (0 if none) |
| `mmse_change_prev` | Change | MMSE change since the previous visit (0 if none) |
| `nwbv_change_prev` | Change | nWBV change since the previous visit (0 if none) |
| `cdr_change_prev` | Change | CDR change since the previous visit (0 if none) |
| `mmse_rate_prev` | Change | MMSE change per year since the previous visit (0 if none) |
| `nwbv_rate_prev` | Change | nWBV change per year since the previous visit (0 if none) |
| `mri_csf_proxy` | MRI scan | Share of dark (CSF-like) voxels in the central 40% box of the scan, averaged over repeats. Exploratory atrophy proxy; r = -0.54 with nWBV across the 373 sessions |
| `mri_n_scans` | MRI quality | Repeated scans in the session (2 to 4). Audit only |
| `mri_snr` | MRI quality | Signal-to-noise ratio (mean foreground / background SD). Audit only |
| `mri_repeat_r` | MRI quality | Pearson r between the first scan and each repeat; low = motion. Audit only |
| `mri_qc_flag` | MRI quality | 1 if SNR or repeat-scan agreement is more than 3 IQR below Q1. Audit only |
| `mmse_imputed` | Audit | 1 if MMSE was filled from the previous visit |
| `cdr_next` | Target source | CDR at the next visit. **Defines the target; never use as a feature.** |
| `declined` | Target | **Target.** 1 if `cdr_next > cdr` |
| `split` | Split | `train` or `test` (patient-level) |
| `cv_fold` | Split | CV fold 0 to 4 for training rows; -1 for test rows |

`oasis2_mri_sessions.csv` (373 rows) and `oasis2_mri_scans.csv` (1,367 rows) hold the same MRI measures per session and per scan, plus header fields and raw intensity percentiles.

## Preprocessing

### Clinical data (`make_dataset.py`)

1. **Validate** the raw schema and check that `MRI ID` is unique.
2. **MMSE (2 missing visits, 1 patient):** filled with the same patient's previous MMSE. Only 1 of the resulting pairs is affected.
3. **Change features:** differences and yearly rates between each visit and the same patient's previous visit. They look backward only, so they contain no future information.
4. **Pairing:** each visit is paired with the next; the last visit of each patient is dropped (373 visits to 223 pairs). `declined = 1` if CDR increases.
5. **MRI join:** the MRI session measures of visit t are attached by `mri_id` (all 223 pairs match).
6. **Patient-level split:** `StratifiedGroupKFold` holds out 1 of 5 patient groups as the test set (about 20%), stratified on the target; the training set gets 5 grouped, stratified CV folds.
7. **SES (8 patients, 11 pairs):** imputed with the mode of training patients (SES = 2); `ses_imputed` flags these rows.
8. **Excluded raw columns:** `Group` (assigned from the full visit history, so it leaks the outcome) and `Hand` (constant).
9. **Encoding:** `M/F` to `sex_male`; column names converted to snake_case.

### MRI scans (`make_mri_dataset.py`)

1. **Stream** each archive once and decode every scan in memory with nibabel; nothing is extracted to disk. Archive SHA-256 hashes are computed during the same read.
2. **Per scan:** header fields, intensity percentiles above an Otsu threshold, SNR, and the CSF proxy.
3. **Per session:** average the scan metrics, measure repeat-scan agreement on 4x-downsampled volumes, and add the QC flags.
4. **Link** every session to the clinical visits and visit pairs and verify coverage.
5. **Keep** half-resolution central slices of the first scan per session (`data/interim/`, not versioned) for figures.

The scans are not yet skull-stripped, registered, or intensity-normalized; these steps belong to the CNN preprocessing in the modeling sprint.

## Known Limitations and Biases

- **Small sample:** 223 pairs and 31 positives. The test set has only 6 positives, so test metrics will have wide uncertainty. Use the grouped CV folds for model selection.
- **Class imbalance:** 13.9% positives. Accuracy is misleading; report recall, F1-score, and ROC-AUC.
- **Unequal follow-up:** visit gaps range from 182 to 1,707 days. `days_to_next` is the horizon the prediction refers to and must be known (or planned) at prediction time.
- **Coarse target:** CDR moves in steps of 0.5. Five pairs show a CDR decrease and are labeled stable. Only 3 pairs reach CDR 2.
- **Zero-filled change features:** for the 150 first-visit pairs the change features are 0 by construction; `has_prev_visit` lets models tell these apart from true "no change".
- **Redundant features:** `etiv` and `asf` are almost perfectly correlated; linear models should keep only one.
- **Raw MRI:** scans include skull, neck, and face, are not registered, and have scanner-dependent intensity scales. The CSF proxy is a coarse measure, not a validated segmentation. Flagged sessions (`mri_qc_flag = 1`) are kept for a sensitivity analysis.
- **Representativeness:** one research site, right-handed volunteers aged 60+, no race or ethnicity information. Results may not generalize to other populations.

## Ethical and Privacy Considerations

- OASIS data are de-identified. Do not attempt to re-identify participants or link them to other sources.
- Use must follow the OASIS Data Use Agreement and acknowledgment requirements published at https://www.oasis-brains.org/. Raw scans are not redistributed in the repository.
- Predictions from models trained on this data are research outputs, not medical advice.

## Maintenance and Versioning

Rebuild in this order (outputs are deterministic for the same inputs):

1. `python -m src.data.make_mri_dataset` (about 2 to 5 minutes; needs both archives in `data/raw/`)
2. `python -m src.data.make_dataset`
3. `python -m src.visualization.eda_report`

Any change to the hashes, row counts, imputation values, or split statistics in the metadata JSON files requires a new dataset version.

## Citation

Marcus, D. S., Fotenos, A. F., Csernansky, J. G., Morris, J. C., & Buckner, R. L. (2010). Open Access Series of Imaging Studies: Longitudinal MRI data in nondemented and demented older adults. *Journal of Cognitive Neuroscience, 22*(12), 2677-2684.
