"""Project-wide paths and constants shared by the data and reporting scripts."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw OASIS-2 longitudinal CSV (Kaggle-hosted copy, unmodified).
RAW_CSV = PROJECT_ROOT / "oasis_longitudinal.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ML_READY_CSV = PROCESSED_DIR / "oasis2_visit_pairs.csv"
ML_READY_METADATA = PROCESSED_DIR / "oasis2_visit_pairs_metadata.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DATASET_CARD = REPORTS_DIR / "dataset_card.md"
EDA_REPORT_PDF = REPORTS_DIR / "eda_report.pdf"

# Raw OASIS-2 structural MRI archives from oasis-brains.org (kept compressed;
# they are about 45 GB once extracted, so they are streamed instead).
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MRI_ARCHIVES = [RAW_DIR / "OAS2_RAW_PART1.tar.gz", RAW_DIR / "OAS2_RAW_PART2.tar.gz"]

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
MRI_SCANS_CSV = PROCESSED_DIR / "oasis2_mri_scans.csv"
MRI_SESSIONS_CSV = PROCESSED_DIR / "oasis2_mri_sessions.csv"
MRI_METADATA = PROCESSED_DIR / "oasis2_mri_metadata.json"
MRI_SLICES = INTERIM_DIR / "oasis2_mri_slices.npz"

RANDOM_STATE = 42
N_TEST_FOLDS = 5  # 1 of 5 patient groups held out -> ~20% test set
N_CV_FOLDS = 5  # patient-grouped cross-validation folds inside the training set

RAW_COLUMNS = [
    "Subject ID", "MRI ID", "Group", "Visit", "MR Delay", "M/F", "Hand",
    "Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF",
]

ID_COLUMNS = ["subject_id", "mri_id", "visit"]

# Model inputs, grouped by source. Every value is known at the earlier visit
# (or is the planned follow-up horizon), so none of them leaks the outcome.
FEATURE_GROUPS = {
    "Demographics": ["age", "sex_male", "educ", "ses", "ses_imputed"],
    "Clinical / cognitive": ["mmse", "cdr"],
    "MRI measures (CSV)": ["etiv", "nwbv", "asf"],
    "Visit timing": ["days_since_baseline", "days_to_next", "has_prev_visit",
                     "days_since_prev"],
    "Change since previous visit": ["mmse_change_prev", "nwbv_change_prev",
                                    "cdr_change_prev", "mmse_rate_prev",
                                    "nwbv_rate_prev"],
    "MRI scan measure": ["mri_csf_proxy"],
}

# Inputs of the metadata-only model (everything in the Kaggle CSV).
METADATA_FEATURES = [c for group, columns in FEATURE_GROUPS.items()
                     if group != "MRI scan measure" for c in columns]
# Computed from the raw scans by src/data/make_mri_dataset.py.
MRI_SCAN_FEATURES = FEATURE_GROUPS["MRI scan measure"]
FEATURE_COLUMNS = METADATA_FEATURES + MRI_SCAN_FEATURES

# Scan quality of the earlier visit. Kept for audits and sensitivity checks,
# not as model inputs.
MRI_QC_COLUMNS = ["mri_n_scans", "mri_snr", "mri_repeat_r", "mri_qc_flag"]

TARGET_COLUMN = "declined"

# Kept for analysis only. CDR at the next visit defines the target, so it
# must never be used as a model input.
TARGET_DERIVED_COLUMNS = ["cdr_next"]

SPLIT_COLUMNS = ["split", "cv_fold"]
