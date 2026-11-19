"""Final, one-time evaluation of the candidate models on the held-out test patients.

Everything up to this step used only the 119 training patients. Here every
candidate model is refit on all 178 training pairs with the settings chosen
by cross-validation, and scored once on the 45 test pairs (31 patients, 6
declines) that no earlier step has looked at.

What is reported for each model:

* ROC-AUC (primary), PR-AUC, Brier score, and recall / precision / F1 /
  accuracy at two thresholds: the default 0.5, and a threshold chosen on
  the training out-of-fold predictions with Youden's J (the point of the
  training ROC curve that maximizes sensitivity + specificity - 1). With
  14% declines, 0.5 is rarely the right cut-off for a screening use. The
  test labels are never used to pick a threshold or a model.
* 95% confidence intervals from a patient-level bootstrap: test patients
  (not visit pairs) are resampled with replacement, so the pairs of one
  patient stay together, as in the CV folds.
* For the CNNs, the test predictions are the mean of models trained with
  three seeds, and the per-seed ROC-AUC range shows how much training noise
  alone moves the score.

The model selected by cross-validation (tuned gradient boosting on
metadata) was fixed before this step; the other models are scored for the
comparison the project question asks for (MRI vs. metadata vs. both).

Usage:
    python -m src.models.final_evaluation
"""

import ast
import json
from datetime import datetime, timezone

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import PredefinedSplit, cross_val_predict

from src import config
from src.data.make_mri_volumes import load_volumes
from src.models.train_cnn import TrainConfig, fit_and_predict, standardize
from src.models.tune import FEATURE_SETS, MODEL_SPACES

TEST_RESULTS_CSV = config.REPORTS_DIR / "test_results.csv"
TEST_PREDICTIONS_CSV = config.REPORTS_DIR / "test_predictions.csv"
TEST_SUMMARY_JSON = config.REPORTS_DIR / "test_summary.json"

N_BOOTSTRAP = 1000
CNN_SEEDS = (42, 43, 44)
# The model chosen by cross-validation before the test set was opened.
SELECTED_MODEL = "gradient_boosting | metadata-only (tuned)"


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the training and test visit pairs."""
    pairs = pd.read_csv(config.ML_READY_CSV)
    train = pairs[pairs["split"] == "train"].reset_index(drop=True)
    test = pairs[pairs["split"] == "test"].reset_index(drop=True)
    if set(train["subject_id"]) & set(test["subject_id"]):
        raise ValueError("Patients overlap between train and test.")
    return train, test


def tuned_tabular_models() -> dict[str, tuple]:
    """Rebuild every tuned model from ``reports/tuning_results.csv``.

    Returns ``{name: (unfitted estimator, feature list)}``. The search space
    functions of ``src.models.tune`` are reused with Optuna's ``FixedTrial``,
    so the estimators are built exactly as during tuning.
    """
    tuned = pd.read_csv(config.TUNING_RESULTS_CSV)
    models = {}
    for row in tuned.itertuples():
        params = ast.literal_eval(row.best_params)
        estimator = MODEL_SPACES[row.model](optuna.trial.FixedTrial(params))
        models[f"{row.model} | {row.feature_set} (tuned)"] = (
            estimator, FEATURE_SETS[row.feature_set])
    return models


def youden_threshold(y: np.ndarray, probs: np.ndarray) -> float:
    """Threshold that maximizes sensitivity + specificity - 1 (Youden's J).

    Called with the training out-of-fold predictions only.
    """
    fpr, tpr, thresholds = roc_curve(y, probs)
    finite = np.isfinite(thresholds)
    best = np.argmax((tpr - fpr)[finite])
    return float(thresholds[finite][best])


def point_metrics(y: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    """Threshold-free and thresholded metrics for one set of predictions."""
    preds = (probs >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, probs),
        "pr_auc": average_precision_score(y, probs),
        "brier": brier_score_loss(y, probs),
        "recall": recall_score(y, preds, zero_division=0),
        "precision": precision_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "accuracy": accuracy_score(y, preds),
        "flagged": int(preds.sum()),
    }


def patient_bootstrap(y: np.ndarray, probs: dict[str, np.ndarray], patients: np.ndarray,
                      n: int = N_BOOTSTRAP, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Bootstrap ROC-AUC and PR-AUC of every model by resampling test patients.

    All models are scored on the same resamples, so differences between them
    can be compared draw by draw. Resamples with only one class are skipped.
    """
    rng = np.random.default_rng(seed)
    ids = np.unique(patients)
    rows_of = {pid: np.flatnonzero(patients == pid) for pid in ids}
    draws = []
    for b in range(n):
        idx = np.concatenate([rows_of[pid] for pid in rng.choice(ids, size=len(ids))])
        if len(np.unique(y[idx])) < 2:
            continue
        for name, p in probs.items():
            draws.append({"draw": b, "model": name,
                          "roc_auc": roc_auc_score(y[idx], p[idx]),
                          "pr_auc": average_precision_score(y[idx], p[idx])})
    return pd.DataFrame(draws)


def evaluate_tabular(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, dict]:
    """Fit the dummy and tuned tabular models; return OOF and test predictions."""
    y_train = train[config.TARGET_COLUMN].to_numpy()
    cv = PredefinedSplit(train["cv_fold"])
    candidates = {"dummy (always prior)": (DummyClassifier(strategy="prior"),
                                           config.MODEL_METADATA_FEATURES),
                  **tuned_tabular_models()}
    out = {}
    for name, (estimator, features) in candidates.items():
        oof = cross_val_predict(clone(estimator), train[features], y_train, cv=cv,
                                method="predict_proba")[:, 1]
        fitted = clone(estimator).fit(train[features], y_train)
        out[name] = {"family": "tabular", "oof": oof,
                     "test": fitted.predict_proba(test[features])[:, 1]}
    return out


def evaluate_cnns(train: pd.DataFrame, test: pd.DataFrame, epochs: int = 10) -> dict[str, dict]:
    """Train each CNN on all training pairs with three seeds; predict the test pairs.

    The thresholds use the out-of-fold predictions saved by
    ``src.models.train_cnn`` (seed 42), the same CV run the model card reports.
    """
    volumes = load_volumes()
    train_vol = np.stack([volumes[m] for m in train["mri_id"]])
    test_vol = np.stack([volumes[m] for m in test["mri_id"]])
    y_train = train[config.TARGET_COLUMN].to_numpy()
    y_test = test[config.TARGET_COLUMN].to_numpy()

    out = {}
    for model, label in [("mri", "3D CNN | MRI volume"),
                         ("multimodal", "3D CNN + metadata | MRI volume + metadata")]:
        tab_train = tab_test = None
        if model == "multimodal":
            tab_train, tab_test = standardize(
                train[config.MODEL_METADATA_FEATURES].to_numpy(dtype=float),
                test[config.MODEL_METADATA_FEATURES].to_numpy(dtype=float))
        seed_probs = []
        for seed in CNN_SEEDS:
            cfg = TrainConfig(model=model, epochs=epochs, seed=seed)
            # Validation inputs are the test pairs, but only the final-epoch
            # predictions are kept: nothing is selected on them.
            _, probs, _ = fit_and_predict(cfg, train_vol, tab_train, y_train,
                                          test_vol, tab_test, y_test)
            seed_probs.append(probs)
            print(f"  {label}, seed {seed}: test ROC-AUC {roc_auc_score(y_test, probs):.3f}")
        oof = pd.read_csv(config.REPORTS_DIR / config.CNN_OOF_TEMPLATE.format(model=model))
        oof = train[["mri_id"]].merge(oof[["mri_id", "predicted_risk"]], on="mri_id", how="left")
        out[label] = {"family": "cnn", "oof": oof["predicted_risk"].to_numpy(),
                      "test": np.mean(seed_probs, axis=0),
                      "seed_roc_auc": [roc_auc_score(y_test, p) for p in seed_probs]}
    return out


def summarize(train: pd.DataFrame, test: pd.DataFrame, models: dict[str, dict],
              n_bootstrap: int = N_BOOTSTRAP) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every model on the test set and attach bootstrap intervals."""
    y_train = train[config.TARGET_COLUMN].to_numpy()
    y_test = test[config.TARGET_COLUMN].to_numpy()
    boot = patient_bootstrap(y_test, {k: v["test"] for k, v in models.items()},
                             test["subject_id"].to_numpy(), n=n_bootstrap)

    rows = []
    for name, m in models.items():
        cv_auc = roc_auc_score(y_train, m["oof"])
        tuned_t = youden_threshold(y_train, m["oof"])
        at_half = point_metrics(y_test, m["test"], 0.5)
        at_tuned = point_metrics(y_test, m["test"], tuned_t)
        b = boot[boot["model"] == name]
        row = {"model": name, "family": m["family"], "selected": name == SELECTED_MODEL,
               "cv_roc_auc": cv_auc,
               "test_roc_auc": at_half["roc_auc"],
               "roc_auc_ci_low": b["roc_auc"].quantile(0.025),
               "roc_auc_ci_high": b["roc_auc"].quantile(0.975),
               "test_pr_auc": at_half["pr_auc"],
               "pr_auc_ci_low": b["pr_auc"].quantile(0.025),
               "pr_auc_ci_high": b["pr_auc"].quantile(0.975),
               "test_brier": at_half["brier"]}
        row.update({f"{k}@0.5": v for k, v in at_half.items()
                    if k not in ("roc_auc", "pr_auc", "brier")})
        row["youden_threshold"] = tuned_t
        row.update({f"{k}@youden": v for k, v in at_tuned.items()
                    if k not in ("roc_auc", "pr_auc", "brier")})
        if "seed_roc_auc" in m:
            row["seed_roc_auc_min"] = min(m["seed_roc_auc"])
            row["seed_roc_auc_max"] = max(m["seed_roc_auc"])
        rows.append(row)
    results = pd.DataFrame(rows).sort_values("test_roc_auc", ascending=False)
    return results.reset_index(drop=True), boot


def sensitivity_without_qc_flags(test: pd.DataFrame, models: dict[str, dict]) -> pd.DataFrame:
    """Test ROC-AUC with and without pairs whose visit-t MRI failed quality control."""
    keep = (test["mri_qc_flag"] == 0).to_numpy()
    y = test[config.TARGET_COLUMN].to_numpy()
    rows = []
    for name, m in models.items():
        rows.append({"model": name, "all_pairs": roc_auc_score(y, m["test"]),
                     "without_qc_flagged": roc_auc_score(y[keep], m["test"][keep]),
                     "pairs_dropped": int((~keep).sum())})
    return pd.DataFrame(rows)


def save(test: pd.DataFrame, models: dict[str, dict], results: pd.DataFrame) -> None:
    """Write the results table, per-pair test predictions, and a JSON summary."""
    results.round(4).to_csv(TEST_RESULTS_CSV, index=False)
    preds = test[["subject_id", "mri_id", "cdr", "mmse", "age", "sex_male", "mri_qc_flag",
                  config.TARGET_COLUMN]].copy()
    for name, m in models.items():
        preds[name] = m["test"].round(4)
    preds.to_csv(TEST_PREDICTIONS_CSV, index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_pairs": int(len(test)), "test_patients": int(test["subject_id"].nunique()),
        "test_declines": int(test[config.TARGET_COLUMN].sum()),
        "bootstrap_resamples": N_BOOTSTRAP, "cnn_seeds": list(CNN_SEEDS),
        "threshold_rule": "Youden's J on training out-of-fold predictions",
        "selected_model": SELECTED_MODEL,
    }
    TEST_SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {TEST_RESULTS_CSV}, {TEST_PREDICTIONS_CSV}, {TEST_SUMMARY_JSON}")


def main() -> None:
    """Run the full final evaluation and print the results table."""
    train, test = load_splits()
    print(f"Train: {len(train)} pairs; test: {len(test)} pairs, "
          f"{test['subject_id'].nunique()} patients, {test[config.TARGET_COLUMN].sum()} declines")
    models = evaluate_tabular(train, test)
    models.update(evaluate_cnns(train, test))
    results, _ = summarize(train, test, models)
    save(test, models, results)
    cols = ["model", "cv_roc_auc", "test_roc_auc", "roc_auc_ci_low", "roc_auc_ci_high",
            "test_pr_auc", "recall@youden", "precision@youden"]
    print(results[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
