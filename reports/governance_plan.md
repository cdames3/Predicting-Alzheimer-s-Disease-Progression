# Monitoring, Governance, and Documentation Plan

This plan covers how the dataset and models are monitored, which cards document them, and how the code and reports are documented. The checks are implemented in `src/monitoring/drift.py` and run with `python -m src.monitoring.drift`, which writes `reports/monitoring_report.json`.

## 1. Monitoring metrics

### Dataset (every new data batch or rebuild)

| Check | Metric | Alert when |
|---|---|---|
| Schema | Expected ID, feature, and MRI QC columns present | Any column missing |
| Completeness | Missing values per feature after imputation | Any missing value |
| Validity | Values outside plausible ranges (age 18-110, MMSE 0-30, CDR 0-3, nWBV 0.5-0.9, eTIV 1000-2200, SES 1-5) | Any value out of range |
| MRI quality | Share of sessions with `mri_qc_flag = 1` (low SNR or motion) | Above 10% (twice the 5.1% rate of all OASIS-2 sessions) |
| Feature drift | Population Stability Index (PSI) per feature vs. the training data | PSI above 0.25 **and** above the 95th percentile of a 200-permutation null (sampling noise) |
| Feature drift | Two-sample Kolmogorov-Smirnov test | p < 0.01 |
| Label drift | Decline rate of the new batch vs. 14.0% in training | Outside the 95% binomial interval |
| Versioning | SHA-256 of the raw files and outputs (metadata JSON) | Hash changes without a new dataset version |

The permutation null matters because batches are small: with 45 test pairs, PSI reaches 0.3 to 0.45 from sampling noise alone, so a fixed 0.25 cutoff would flag almost every feature.

**First run (test patients vs. training patients, features only):** data quality passed (no missing or out-of-range values; QC-flagged rate 8.9%). Continuous features showed no drift beyond sampling noise. Alerts: `ses_imputed` (+0.9 SD; the test patients include more imputed SES values), `ses`, and `days_since_prev`. These come from the random patient split, not from a data problem, and are noted as a limitation of test-set comparisons.

### Models (every training run and every scored batch with known outcomes)

| Metric | Why | Alert when |
|---|---|---|
| ROC-AUC (primary), PR-AUC | Ranking quality, independent of the threshold; PR-AUC focuses on the rare class | ROC-AUC below 0.60 |
| Recall, precision, F1 at the chosen threshold | A missed decline is the costly error | Recall below the baseline's recall |
| Brier score and calibration table (5 risk bins) | Predicted risks should match observed decline rates | Brier above the prior-only model (0.12) |
| Predicted vs. observed positive rate | Detects threshold or score drift | Ratio outside 0.5-2 |
| ROC-AUC by sex and age band (<70, 70-79, 80+) | Fairness across subgroups | Gap between subgroups above 0.10 |
| Training loss curves and epoch time (W&B) | Overfitting and compute cost | Validation loss rising while training loss falls |

Every training run logs its configuration, metrics, and timing to the team Weights & Biases project (`ds-capstone-group-5/predicting-alzheimers-progression`), grouped as `baselines`, `cnn`, and `tuning`.

### Re-evaluation triggers

A model is re-evaluated (and retrained if needed) when any dataset check raises an alert, when model ROC-AUC drops below 0.60 on a labeled batch, when the dataset version changes, or when the preprocessing code changes.

## 2. Model and dataset cards

| Card | File | Updated when |
|---|---|---|
| Dataset card | `reports/dataset_card.md` (attached to the EDA PDF) | New dataset version (hashes, rows, splits, preprocessing) |
| Model card | `reports/model_card.md` | Every model that is compared in a checkpoint or selected as final |

- The **dataset card** follows the Hugging Face dataset card sections, extended with the Data Cards Playbook items for lineage and MRI quality.
- The **model card** follows Mitchell et al. (2019), "Model Cards for Model Reporting": model details, intended use, factors, metrics, evaluation data, training data, quantitative analyses (overall and per subgroup), ethical considerations, and caveats.
- Each card states its version, the dataset version and hashes it refers to, and the W&B runs behind its numbers, so results can be traced back to data and code.

## 3. Documentation plan

| Type | Where | Content |
|---|---|---|
| In-code documentation | Every module under `src/` | Module docstring (purpose, inputs, outputs, usage command); a docstring for every public function and class; comments explain *why*, not *what* |
| API reference | `docs/` (Sphinx, autodoc + napoleon) | Generated from the docstrings; build with `sphinx-build -b html docs docs/_build/html`, publishable on Read the Docs |
| Pipeline guide | `README.md`, `docs/pipeline.rst` | Order of the commands to rebuild the data, train, tune, and monitor |
| Reports | `reports/` | EDA PDF with the dataset card, model card, experiment plan, this plan, result tables |
| Experiment log | Weights & Biases | Configuration, metrics, curves, and timing of every run |

**Style:**
- Docstrings follow the **Google Python style** (`Args:`, `Returns:` sections), rendered by Sphinx's napoleon extension.
- Prose (README, cards, reports) follows the **Google developer documentation style guide**: active voice, present tense, short sentences, sentence-case headings, code in backticks.
- Repository layout follows **Cookiecutter Data Science**: `data/{raw,interim,processed}`, `src/{data,models,monitoring,visualization}`, `reports/`, `notebooks/`, `docs/`, `models/`. Notebooks are for exploration only; reusable code lives in `src/` modules.
