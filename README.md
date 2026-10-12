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

Comparing these models measures how much predictive value comes from imaging, from patient metadata, and from their combination.

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
├── data/            # Local data only (not committed; see Data section)
├── notebooks/       # EDA and experiments
├── src/             # Preprocessing, models, training, evaluation code
├── reports/         # Figures and checkpoint materials
├── requirements.txt
└── README.md
```

## Data
Data comes from [OASIS](https://www.oasis-brains.org/). Raw data is **not** stored in this repository because of size and the OASIS data use terms. Download it separately and place it in `data/`.

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

*This project is for research and educational purposes and is not intended for clinical diagnosis.*
