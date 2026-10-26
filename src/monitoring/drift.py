"""Dataset and model monitoring checks for the visit-pair models.

Three kinds of checks, each with an alert threshold:

* Data quality: schema, missing values, value ranges, and the share of MRI
  sessions that fail quality control.
* Data drift: Population Stability Index (PSI) and the two-sample
  Kolmogorov-Smirnov test for every model input, comparing a new batch of
  visit pairs with the training data the models were fit on. PSI is biased
  upward for small batches, so a feature only drifts when its PSI also
  exceeds what random splits of the same data produce (permutation test).
* Model performance: ROC-AUC, PR-AUC, recall, Brier score, and calibration,
  overall and per subgroup (sex, age band), once outcomes are known.

Running this module compares the held-out test patients with the training
patients as a stand-in for a future batch of data. It uses the test
features only, never the test labels, so the test set stays untouched for
the final evaluation. The performance checks run on the CNN out-of-fold
predictions of the training patients (written by ``src.models.train_cnn``).

Usage:
    python -m src.monitoring.drift
"""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ks_2samp
from sklearn.metrics import (average_precision_score, brier_score_loss, recall_score,
                             roc_auc_score)

from src import config

# Common PSI rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major.
PSI_WARN, PSI_ALERT = 0.1, 0.25
PSI_PERMUTATIONS = 200
PSI_NULL_QUANTILE = 0.95
KS_ALPHA = 0.01
# 5.1% of all OASIS-2 sessions are QC-flagged; alert when a batch doubles that.
MAX_QC_FAIL_RATE = 0.10
MIN_ROC_AUC = 0.60  # the best baseline reached 0.62; below 0.60 means retraining
MAX_SUBGROUP_AUC_GAP = 0.10

# Plausible ranges used by the data quality checks.
VALID_RANGES = {"age": (18, 110), "mmse": (0, 30), "cdr": (0, 3), "nwbv": (0.5, 0.9),
                "etiv": (1000, 2200), "educ": (0, 30), "ses": (1, 5)}


def population_stability_index(reference: np.ndarray, current: np.ndarray,
                               bins: int = 10) -> float:
    """PSI between two samples, binned on the reference deciles.

    Features with few distinct values (for example CDR or sex) are binned on
    those values instead, so each category is its own bin.
    """
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    values = np.unique(reference)
    if len(values) <= bins:
        ref_counts = np.array([(reference == v).sum() for v in values], float)
        cur_counts = np.array([(current == v).sum() for v in values], float)
        cur_counts = np.append(cur_counts, len(current) - cur_counts.sum())  # unseen values
        ref_counts = np.append(ref_counts, 0.0)
    else:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ref_counts = np.histogram(reference, edges)[0].astype(float)
        cur_counts = np.histogram(current, edges)[0].astype(float)

    # Half a count per bin avoids log(0) for empty bins without letting a
    # single empty bin of a small batch dominate the index.
    ref_share = (ref_counts + 0.5) / (ref_counts.sum() + 0.5 * len(ref_counts))
    cur_share = (cur_counts + 0.5) / (cur_counts.sum() + 0.5 * len(cur_counts))
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def psi_null_threshold(reference: np.ndarray, current: np.ndarray,
                       n_permutations: int = PSI_PERMUTATIONS,
                       quantile: float = PSI_NULL_QUANTILE) -> float:
    """PSI that random splits of the pooled data exceed with prob. 1 - quantile.

    Shuffling the pooled values and splitting them into groups of the
    original sizes gives the PSI expected from sampling noise alone.
    """
    rng = np.random.default_rng(config.RANDOM_STATE)
    pooled = np.concatenate([reference, current])
    null = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(pooled)
        null.append(population_stability_index(shuffled[:len(reference)],
                                                shuffled[len(reference):]))
    return float(np.quantile(null, quantile))


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame,
                  features: list[str]) -> pd.DataFrame:
    """PSI, KS test, and mean shift (in reference SDs) for each feature.

    A feature is flagged only when its PSI is above both the fixed threshold
    and the permutation threshold, or when the KS test rejects at
    ``KS_ALPHA``.
    """
    rows = []
    for feature in features:
        ref, cur = reference[feature].to_numpy(float), current[feature].to_numpy(float)
        psi = population_stability_index(ref, cur)
        psi_null = psi_null_threshold(ref, cur)
        ks = ks_2samp(ref, cur)
        std = ref.std() or 1.0
        beyond_noise = psi > psi_null
        rows.append({
            "feature": feature,
            "psi": round(psi, 4),
            "psi_null_p95": round(psi_null, 4),
            "ks_statistic": round(float(ks.statistic), 4),
            "ks_pvalue": round(float(ks.pvalue), 4),
            "mean_shift_sd": round(float((cur.mean() - ref.mean()) / std), 3),
            "status": ("alert" if (beyond_noise and psi > PSI_ALERT) or ks.pvalue < KS_ALPHA
                       else "warn" if beyond_noise and psi > PSI_WARN else "ok"),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def data_quality(batch: pd.DataFrame) -> dict:
    """Schema, missing values, out-of-range values, and MRI QC failures."""
    expected = (config.ID_COLUMNS + config.FEATURE_COLUMNS + config.MRI_QC_COLUMNS)
    missing_columns = sorted(set(expected) - set(batch.columns))
    present = [c for c in config.FEATURE_COLUMNS if c in batch.columns]
    out_of_range = {c: int((~batch[c].between(*VALID_RANGES[c])).sum())
                    for c in VALID_RANGES if c in batch.columns}
    qc_fail_rate = float(batch["mri_qc_flag"].mean()) if "mri_qc_flag" in batch else None
    report = {
        "rows": int(len(batch)),
        "patients": int(batch["subject_id"].nunique()),
        "missing_columns": missing_columns,
        "missing_values": {c: int(n) for c, n in batch[present].isna().sum().items() if n},
        "out_of_range": {c: n for c, n in out_of_range.items() if n},
        "mri_qc_fail_rate": qc_fail_rate,
    }
    report["status"] = ("alert" if missing_columns or report["missing_values"]
                        or report["out_of_range"]
                        or (qc_fail_rate or 0) > MAX_QC_FAIL_RATE else "ok")
    return report


def label_drift(y_batch: np.ndarray, reference_rate: float, alpha: float = 0.05) -> dict:
    """Test whether a labeled batch's decline rate differs from training.

    Uses an exact binomial test, so it is valid for the small batch sizes of
    this project.
    """
    y_batch = np.asarray(y_batch).astype(int)
    test = binomtest(int(y_batch.sum()), len(y_batch), reference_rate)
    return {"batch_rate": float(y_batch.mean()), "reference_rate": float(reference_rate),
            "pvalue": float(test.pvalue), "status": "alert" if test.pvalue < alpha else "ok"}


def calibration_table(y: np.ndarray, probs: np.ndarray, bins: int = 5) -> list[dict]:
    """Mean predicted risk vs. observed decline rate in equal-size risk bins."""
    order = np.argsort(probs)
    return [{"mean_predicted": round(float(probs[idx].mean()), 3),
             "observed_rate": round(float(y[idx].mean()), 3), "n": int(len(idx))}
            for idx in np.array_split(order, bins)]


def model_performance(y: np.ndarray, probs: np.ndarray, groups: pd.DataFrame | None = None,
                      threshold: float = 0.5) -> dict:
    """Overall and per-subgroup metrics for a batch with known outcomes.

    Args:
        y: Observed targets (1 = declined).
        probs: Predicted probabilities of decline.
        groups: Optional columns to break the metrics down by (one column
            per grouping, for example sex and age band).
        threshold: Decision threshold for recall.
    """
    y, probs = np.asarray(y), np.asarray(probs)
    report = {
        "roc_auc": float(roc_auc_score(y, probs)),
        "pr_auc": float(average_precision_score(y, probs)),
        "recall": float(recall_score(y, probs >= threshold, zero_division=0)),
        "brier": float(brier_score_loss(y, probs)),
        "predicted_positive_rate": float((probs >= threshold).mean()),
        "observed_positive_rate": float(y.mean()),
        "calibration": calibration_table(y, probs),
        "subgroups": {},
    }
    gaps = []
    for column in ([] if groups is None else groups.columns):
        aucs = {}
        for value, idx in groups.groupby(column).indices.items():
            if len(np.unique(y[idx])) == 2:
                aucs[str(value)] = round(float(roc_auc_score(y[idx], probs[idx])), 3)
        report["subgroups"][column] = aucs
        if len(aucs) > 1:
            gaps.append(max(aucs.values()) - min(aucs.values()))
    report["max_subgroup_auc_gap"] = max(gaps) if gaps else None
    report["status"] = ("alert" if report["roc_auc"] < MIN_ROC_AUC
                        or (gaps and max(gaps) > MAX_SUBGROUP_AUC_GAP) else "ok")
    return report


def age_band(age: pd.Series) -> pd.Series:
    """Age bands used for the subgroup checks."""
    return pd.cut(age, [0, 70, 80, 120], labels=["<70", "70-79", "80+"], right=False)


def main() -> None:
    """Check the held-out patients' features against the training patients."""
    pairs = pd.read_csv(config.ML_READY_CSV)
    reference = pairs[pairs["split"] == "train"]
    batch = pairs[pairs["split"] == "test"]

    drift = feature_drift(reference, batch, config.FEATURE_COLUMNS)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference": {"split": "train", "rows": int(len(reference))},
        "batch": {"split": "test (features only)", "rows": int(len(batch))},
        "thresholds": {"psi_warn": PSI_WARN, "psi_alert": PSI_ALERT,
                       "psi_null_quantile": PSI_NULL_QUANTILE, "ks_alpha": KS_ALPHA,
                       "max_qc_fail_rate": MAX_QC_FAIL_RATE, "min_roc_auc": MIN_ROC_AUC,
                       "max_subgroup_auc_gap": MAX_SUBGROUP_AUC_GAP},
        "data_quality": data_quality(batch),
        "feature_drift": drift.to_dict(orient="records"),
        "drift_summary": drift["status"].value_counts().to_dict(),
        "model_performance": {},
    }
    for model in ["mri", "multimodal"]:
        path = config.REPORTS_DIR / config.CNN_OOF_TEMPLATE.format(model=model)
        if path.exists():
            preds = pd.read_csv(path)
            groups = pd.DataFrame({"sex": preds["sex_male"].map({1: "male", 0: "female"}),
                                   "age_band": age_band(preds["age"])})
            report["model_performance"][f"cnn_{model} (out-of-fold, train)"] = \
                model_performance(preds[config.TARGET_COLUMN], preds["predicted_risk"], groups)
    config.MONITORING_REPORT.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Data quality: {report['data_quality']['status']}")
    print(f"Feature drift: {report['drift_summary']}")
    print(drift.head(8).to_string(index=False))
    for name, perf in report["model_performance"].items():
        print(f"{name}: ROC-AUC {perf['roc_auc']:.3f}, Brier {perf['brier']:.3f}, "
              f"subgroups {perf['subgroups']} -> {perf['status']}")
    print(f"Saved {config.MONITORING_REPORT}")


if __name__ == "__main__":
    main()
