# Predicting Cognitive Decline Using Longitudinal MRI and Patient Data

**Team:** Christina Dames, Cinthia Ochoa Torre, Rayna Alexandre-Petrie

## Project Question
Can we use longitudinal brain MRI and patient metadata available at an earlier visit to predict how a patient's cognitive status related to Alzheimer's disease will change at a later visit?

## Overview
Most machine learning work on Alzheimer's disease focuses on detecting whether a patient *currently* shows signs of dementia. This project instead asks whether information from an earlier visit can predict whether a patient's cognitive status will **remain stable or decline** at a later visit.

Using longitudinal data from the **Open Access Series of Imaging Studies (OASIS)**, earlier visits are paired with later visits so the model learns from baseline information to predict future cognitive outcomes.

## Approach
We compare three models:
1. **MRI-only** – a convolutional neural network (CNN) extracts structural features from brain MRI scans.
2. **Metadata-only** – uses variables such as age, sex, education, socioeconomic status, and earlier cognitive measurements.
3. **Multimodal** – combines CNN-derived MRI features with patient metadata.

Comparing these models measures how much predictive value comes from imaging, patient metadata, and their combination.

**Patient-level splitting** ensures visits from the same individual never appear in both training and test sets.

## Evaluation
Models are evaluated on unseen patients using accuracy, precision, recall, F1-score, and ROC-AUC. Because decline may be less common than stability, F1-score, recall, and ROC-AUC are emphasized alongside accuracy.

## Methodology
The project follows the **CRISP-DM** process, organized into Agile sprints:
- Business & Data Understanding
- Data Preparation
- Modeling & Evaluation
- Deployment / Reporting

## Repository Structure
```
├── CheckPoint_1/                  # Checkpoint 1 notebook (business and data understanding)
├── data/
│   ├── raw/                       # OASIS-2 MRI archives (local only, not committed)
│   ├── interim/                   # MRI slices and 64^3 CNN volumes (local only, not committed)
│   └── processed/                 # ML-ready dataset, MRI tables, metadata JSON (committed)
├── docs/                          # Sphinx documentation (API reference from docstrings)
├── models/                        # Trained CNN weights (cnn_mri.pt, cnn_multimodal.pt)
├── notebooks/
│   └── 02_baseline_models.ipynb   # Tabular baselines (Model 2) logged to W&B
├── reports/
│   ├── EDA_Report_MRIandCvs.pdf   # Checkpoint 2 EDA with the dataset card attached
│   ├── dataset_card.md
│   ├── model_card.md              # Baselines and CNN models
│   ├── experiment_plan.md         # Validation, tuning, compute, and time estimates
│   ├── governance_plan.md         # Monitoring metrics, cards, documentation style
│   ├── *_results.csv              # Baseline, CNN, and tuning results
│   └── figures/                   # Report figures, numbered by report page
├── src/
│   ├── config.py                  # Paths, feature groups, constants
│   ├── data/
│   │   ├── make_mri_dataset.py    # Raw MRI archives -> per-scan / per-session tables
│   │   ├── make_dataset.py        # CSV + MRI sessions -> ML-ready visit pairs
│   │   └── make_mri_volumes.py    # Raw MRI archives -> 64^3 normalized CNN inputs
│   ├── models/
│   │   ├── evaluate.py            # Shared patient-grouped CV evaluation (baselines)
│   │   ├── cnn.py                 # 3D CNN (Model 1) with optional metadata branch (Model 3)
│   │   ├── train_cnn.py           # Grouped-CV training, W&B logging, timing
│   │   └── tune.py                # Optuna tuning of the tabular baselines
│   ├── monitoring/
│   │   └── drift.py               # Data quality, feature drift, model performance checks
│   └── visualization/
│       ├── eda_report.py          # Builds the EDA report (sections 1-5 + dataset card)
│       ├── mri_eda.py             # Section 4: MRI scan pages
│       └── report_layout.py       # Shared page layout and colors
├── oasis_longitudinal.csv         # Raw clinical CSV (Kaggle copy, unmodified)
├── requirements.txt
└── README.md
```

## Data
Data comes from [OASIS](https://www.oasis-brains.org/) (OASIS-2 longitudinal study): a clinical CSV and the raw structural MRI scans, linked by `mri_id`.

| File | Description |
|---|---|
| `oasis_longitudinal.csv` | Raw OASIS-2 clinical and MRI-derived measures (Kaggle copy, unmodified) |
| `data/processed/oasis2_visit_pairs.csv` | ML-ready dataset: 223 visit pairs, 20 features (19 clinical + 1 MRI scan measure), MRI quality columns, target, patient-level splits |
| `data/processed/oasis2_visit_pairs_metadata.json` | Lineage, SHA-256 hashes, imputation values, feature lists, split statistics |
| `data/processed/oasis2_mri_sessions.csv` | One row per MRI session (373): scan measures and quality flags; joins on `mri_id` |
| `data/processed/oasis2_mri_scans.csv` | One row per raw MRI scan (1,367): header metadata, intensity, SNR, CSF proxy |
| `data/processed/oasis2_mri_metadata.json` | Archive SHA-256 hashes, header variants, linkage to visits and pairs, QC list |
| `reports/dataset_card.md` | Dataset card |
| `reports/EDA_Report_MRIandCvs.pdf` | Full EDA on the ML-ready dataset (clinical + MRI), with the dataset card attached |
| `reports/model_card.md` | Model card: baselines, tuned baselines, and CNN prototypes |
| `reports/experiment_plan.md` | Validation plan, tuning plan, compute and time estimates, goals |
| `reports/governance_plan.md` | Monitoring metrics and thresholds, cards, documentation style |
| `reports/cnn_results.csv`, `reports/tuning_results.csv` | Grouped-CV results of the CNNs and the Optuna-tuned baselines |
| `reports/monitoring_report.json` | Data quality, feature drift, and CNN performance checks |

Feature lists for each model are defined in `src/config.py`: `METADATA_FEATURES` (metadata-only model), `MRI_SCAN_FEATURES`, and `FEATURE_COLUMNS` (all inputs). `cdr_next` defines the target and must never be used as a feature.

Raw MRI scans are **not** stored in this repository because of their size and the OASIS data use terms. Download `OAS2_RAW_PART1.tar.gz` and `OAS2_RAW_PART2.tar.gz` from OASIS and place them in `data/raw/` (no need to extract them; the script streams the archives).

### Reproduce the dataset and EDA report
```bash
pip install -r requirements.txt
python -m src.data.make_mri_dataset      # MRI tables (needs the archives in data/raw/; ~2-5 min)
python -m src.data.make_dataset          # ML-ready visit pairs (uses the committed MRI session table)
python -m src.visualization.eda_report   # reports/eda_report.pdf and reports/figures/
```

### Train, tune, and monitor
```bash
python -m src.data.make_mri_volumes                  # 64^3 CNN inputs (needs data/raw/; ~2 min)
python -m src.models.train_cnn --model mri           # Model 1: MRI-only 3D CNN
python -m src.models.train_cnn --model multimodal    # Model 3: CNN + metadata
python -m src.models.tune --trials 30                # Optuna tuning of the tabular baselines
python -m src.monitoring.drift                       # Data quality, drift, model checks
sphinx-build -b html docs docs/_build/html           # API documentation
```
Runs are logged to Weights & Biases in offline mode by default. After `wandb login`, upload them with `wandb sync wandb/` or pass `--wandb online`.

## Tools
- **Communication:** Slack
- **Project management:** Trello
- **Version control:** GitHub
- **Documentation:** Google Drive
- **Experiment tracking:** Weights & Biases

## References
Existing OASIS Kaggle notebooks are used only as technical references for preprocessing and CNN development:
- [OASIS Alzheimer's Detection – Training](https://www.kaggle.com/code/metinusta/oasis-alzheimer-s-detection-training)
- [Final DL Project – Alzheimer's Detection](https://www.kaggle.com/code/romyleferink/final-dl-project-alzheimer-s-detection)

The longitudinal visit-pairing strategy, future-status prediction task, multimodal comparison, and patient-level evaluation are this project's own contributions.
