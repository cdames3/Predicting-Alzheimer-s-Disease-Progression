"""Build the ML-ready OASIS-2 visit-pair dataset from the raw longitudinal CSV.

Each output row pairs one visit (time t) with the same patient's next visit
(time t+1). Features describe the patient at visit t (clinical metadata and
measures from that visit's MRI scan); the target ``declined`` is 1 when the
Clinical Dementia Rating (CDR) at visit t+1 is higher than at visit t.

Pipeline:
    1. Load and validate the raw CSV.
    2. Fill missing MMSE values with the patient's previous visit.
    3. Add backward-looking change features (current visit vs. previous visit).
    4. Pair each visit with the next one and create the target.
    5. Attach the MRI scan measures of visit t (from make_mri_dataset).
    6. Assign a patient-level train/test split and grouped CV folds.
    7. Impute SES using statistics from training patients only.
    8. Encode, select, and validate the final columns.

Usage:
    python -m src.data.make_mri_dataset   # MRI session table (run first)
    python -m src.data.make_dataset
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src import config

DAYS_PER_YEAR = 365.25

# Session-table columns joined to each pair, renamed with an ``mri_`` prefix.
MRI_SESSION_COLUMNS = {
    "csf_fraction_proxy": "mri_csf_proxy",
    "n_scans": "mri_n_scans",
    "snr": "mri_snr",
    "repeat_scan_r": "mri_repeat_r",
    "qc_flag": "mri_qc_flag",
}


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hash of a file, used to track dataset versions."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw(path: Path = config.RAW_CSV) -> pd.DataFrame:
    """Load the raw OASIS-2 CSV and check that its schema is as expected."""
    df = pd.read_csv(path)
    missing = set(config.RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Raw CSV is missing columns: {sorted(missing)}")
    if df["MRI ID"].duplicated().any():
        raise ValueError("Raw CSV contains duplicated MRI IDs.")
    return df.sort_values(["Subject ID", "Visit"]).reset_index(drop=True)


def fill_mmse_from_previous_visit(df: pd.DataFrame) -> pd.DataFrame:
    """Carry each patient's last observed MMSE forward to later visits.

    Only earlier visits of the same patient are used, so no future or
    cross-patient information enters the value.
    """
    df = df.copy()
    df["mmse_imputed"] = df["MMSE"].isna().astype(int)
    df["MMSE"] = df.groupby("Subject ID")["MMSE"].ffill()
    return df


def add_previous_visit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add changes since the previous visit (rate of decline features).

    Rates are expressed per year. First visits have no previous visit, so
    their change features are set to 0 and ``has_prev_visit`` is 0.
    """
    df = df.copy()
    grouped = df.groupby("Subject ID")

    df["has_prev_visit"] = grouped.cumcount().gt(0).astype(int)
    df["days_since_prev"] = grouped["MR Delay"].diff()
    df["mmse_change_prev"] = grouped["MMSE"].diff()
    df["nwbv_change_prev"] = grouped["nWBV"].diff()
    df["cdr_change_prev"] = grouped["CDR"].diff()

    years_since_prev = df["days_since_prev"] / DAYS_PER_YEAR
    df["mmse_rate_prev"] = df["mmse_change_prev"] / years_since_prev
    df["nwbv_rate_prev"] = df["nwbv_change_prev"] / years_since_prev

    change_columns = [
        "days_since_prev", "mmse_change_prev", "nwbv_change_prev",
        "cdr_change_prev", "mmse_rate_prev", "nwbv_rate_prev",
    ]
    df[change_columns] = df[change_columns].fillna(0.0)
    return df


def build_visit_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Pair each visit with the next visit and create the binary target.

    The last visit of each patient has no next visit and is dropped.
    """
    df = df.copy()
    grouped = df.groupby("Subject ID")
    df["cdr_next"] = grouped["CDR"].shift(-1)
    df["days_to_next"] = grouped["MR Delay"].shift(-1) - df["MR Delay"]

    pairs = df.dropna(subset=["cdr_next"]).copy()
    pairs["declined"] = (pairs["cdr_next"] > pairs["CDR"]).astype(int)
    return pairs.reset_index(drop=True)


def attach_mri_measures(pairs: pd.DataFrame,
                        path: Path = config.MRI_SESSIONS_CSV) -> pd.DataFrame:
    """Join the MRI session measures of visit t to each pair by MRI ID.

    The measures come from the scan taken at the earlier visit, so they are
    known at prediction time.
    """
    if not path.exists():
        raise FileNotFoundError("Run `python -m src.data.make_mri_dataset` first.")
    sessions = (pd.read_csv(path, usecols=["mri_id", *MRI_SESSION_COLUMNS])
                .rename(columns=MRI_SESSION_COLUMNS))
    merged = (pairs.merge(sessions, left_on="MRI ID", right_on="mri_id",
                          how="left", validate="one_to_one")
              .drop(columns="mri_id"))
    if merged[list(MRI_SESSION_COLUMNS.values())].isna().any().any():
        raise ValueError("Some visit pairs have no matching MRI session.")
    return merged


def assign_splits(pairs: pd.DataFrame) -> pd.DataFrame:
    """Assign a patient-level train/test split and grouped CV folds.

    Splits are stratified on the target and grouped by patient, so no
    patient appears in more than one split or fold.
    """
    pairs = pairs.copy()
    groups = pairs["Subject ID"]
    y = pairs["declined"]

    outer = StratifiedGroupKFold(
        n_splits=config.N_TEST_FOLDS, shuffle=True,
        random_state=config.RANDOM_STATE,
    )
    train_idx, test_idx = next(outer.split(pairs, y, groups))
    pairs["split"] = "train"
    pairs.loc[test_idx, "split"] = "test"

    pairs["cv_fold"] = -1
    train = pairs.loc[train_idx]
    inner = StratifiedGroupKFold(
        n_splits=config.N_CV_FOLDS, shuffle=True,
        random_state=config.RANDOM_STATE,
    )
    for fold, (_, val_idx) in enumerate(
        inner.split(train, train["declined"], train["Subject ID"])
    ):
        pairs.loc[train.index[val_idx], "cv_fold"] = fold
    return pairs


def impute_with_train_statistics(pairs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Impute SES (and any remaining MMSE) using training patients only.

    SES is constant within a patient, so its mode is computed over one value
    per training patient rather than per visit.
    """
    pairs = pairs.copy()
    train = pairs[pairs["split"] == "train"]

    ses_mode = train.groupby("Subject ID")["SES"].first().mode().iloc[0]
    mmse_median = train["MMSE"].median()

    pairs["ses_imputed"] = pairs["SES"].isna().astype(int)
    pairs["SES"] = pairs["SES"].fillna(ses_mode)
    pairs["mmse_imputed"] = (pairs["mmse_imputed"] | pairs["MMSE"].isna()).astype(int)
    pairs["MMSE"] = pairs["MMSE"].fillna(mmse_median)

    return pairs, {"ses_mode": float(ses_mode), "mmse_median": float(mmse_median)}


def finalize_columns(pairs: pd.DataFrame) -> pd.DataFrame:
    """Rename to snake_case, encode sex, and keep only documented columns.

    ``Group`` is dropped because it is assigned from the full visit history
    (target leakage), and ``Hand`` is dropped because it is constant.
    """
    out = pairs.rename(columns={
        "Subject ID": "subject_id",
        "MRI ID": "mri_id",
        "Visit": "visit",
        "MR Delay": "days_since_baseline",
        "Age": "age",
        "EDUC": "educ",
        "SES": "ses",
        "MMSE": "mmse",
        "CDR": "cdr",
        "eTIV": "etiv",
        "nWBV": "nwbv",
        "ASF": "asf",
    })
    out["sex_male"] = (out["M/F"] == "M").astype(int)

    columns = (
        config.ID_COLUMNS + config.FEATURE_COLUMNS + config.MRI_QC_COLUMNS
        + ["mmse_imputed"]
        + config.TARGET_DERIVED_COLUMNS + [config.TARGET_COLUMN]
        + config.SPLIT_COLUMNS
    )
    out = out[columns]

    int_columns = ["days_since_baseline", "days_to_next", "days_since_prev", "educ", "ses"]
    out[int_columns] = out[int_columns].astype(int)
    return out


def validate(df: pd.DataFrame) -> None:
    """Raise an error if the ML-ready dataset breaks any project guarantee."""
    if df[config.FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Features contain missing values after imputation.")
    if not np.isfinite(df[config.FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Features contain infinite values.")
    if not set(df[config.TARGET_COLUMN].unique()) <= {0, 1}:
        raise ValueError("Target must be binary.")
    if (df["days_to_next"] <= 0).any():
        raise ValueError("Each pair must move forward in time.")

    train_ids = set(df.loc[df["split"] == "train", "subject_id"])
    test_ids = set(df.loc[df["split"] == "test", "subject_id"])
    if train_ids & test_ids:
        raise ValueError("Patients overlap between train and test splits.")
    if (df.groupby("subject_id")["cv_fold"].nunique() > 1).any():
        raise ValueError("A patient appears in more than one CV fold.")


def split_summary(df: pd.DataFrame) -> dict:
    """Summarize patients, pairs, and class balance for each split."""
    summary = {}
    for name, part in df.groupby("split"):
        summary[name] = {
            "patients": int(part["subject_id"].nunique()),
            "visit_pairs": int(len(part)),
            "declined": int(part["declined"].sum()),
            "decline_rate": round(float(part["declined"].mean()), 4),
        }
    return summary


def main() -> None:
    """Run the full pipeline and write the dataset and its metadata."""
    raw = load_raw()
    lineage = [{"step": "Load raw OASIS-2 CSV", "rows": len(raw),
                "patients": int(raw["Subject ID"].nunique())}]

    visits = fill_mmse_from_previous_visit(raw)
    visits = add_previous_visit_features(visits)
    lineage.append({"step": "Fill MMSE from previous visit; add change features",
                    "rows": len(visits),
                    "patients": int(visits["Subject ID"].nunique())})

    pairs = build_visit_pairs(visits)
    lineage.append({"step": "Pair each visit with the next; drop last visits",
                    "rows": len(pairs),
                    "patients": int(pairs["Subject ID"].nunique())})

    pairs = attach_mri_measures(pairs)
    lineage.append({"step": "Attach MRI session measures of visit t (mri_id)",
                    "rows": len(pairs),
                    "patients": int(pairs["Subject ID"].nunique())})

    pairs = assign_splits(pairs)
    pairs, imputation = impute_with_train_statistics(pairs)
    dataset = finalize_columns(pairs)
    validate(dataset)
    lineage.append({"step": "Split by patient; impute SES; encode; select columns",
                    "rows": len(dataset),
                    "patients": int(dataset["subject_id"].nunique())})

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(config.ML_READY_CSV, index=False)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_file": config.RAW_CSV.name,
        "raw_sha256": file_sha256(config.RAW_CSV),
        "output_file": config.ML_READY_CSV.name,
        "output_sha256": file_sha256(config.ML_READY_CSV),
        "random_state": config.RANDOM_STATE,
        "lineage": lineage,
        "imputation": imputation,
        "splits": split_summary(dataset),
        "mri_sessions_file": config.MRI_SESSIONS_CSV.name,
        "mri_sessions_sha256": file_sha256(config.MRI_SESSIONS_CSV),
        "feature_columns": config.FEATURE_COLUMNS,
        "metadata_features": config.METADATA_FEATURES,
        "mri_scan_features": config.MRI_SCAN_FEATURES,
        "mri_qc_columns": config.MRI_QC_COLUMNS,
        "target_column": config.TARGET_COLUMN,
        "excluded_raw_columns": {"Group": "target leakage", "Hand": "constant"},
    }
    config.ML_READY_METADATA.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote {len(dataset)} visit pairs to {config.ML_READY_CSV}")
    print(json.dumps(metadata["splits"], indent=2))


if __name__ == "__main__":
    main()
