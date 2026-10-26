# Model Card: Next-Visit Cognitive Decline (OASIS-2)

Format: Mitchell et al. (2019), "Model Cards for Model Reporting". Version 0.1 (Checkpoint 2 prototypes). Results are patient-grouped cross-validation on the training set; the held-out test set has not been used yet.

## Model details

| Property | Value |
|---|---|
| Task | Binary classification: does the Clinical Dementia Rating (CDR) increase between visit t and the patient's next visit? |
| Models | **Model 1** MRI-only 3D CNN; **Model 2** metadata-only tabular models (logistic regression, random forest, gradient boosting); **Model 3** multimodal 3D CNN + metadata branch |
| Architecture (CNN) | 4 blocks of Conv3d (3x3x3) + BatchNorm + ReLU + 2x max pooling, channels 8-16-32-64, global average pooling, dropout 0.3, linear head. Model 3 adds Linear(18 -> 16) + ReLU on the metadata and concatenates it with the 64-d MRI embedding. About 73k parameters |
| Input (CNN) | First T1-weighted MPRAGE scan of visit t, downsampled to 64 x 64 x 64 and z-scored on head voxels (`src/data/make_mri_volumes.py`) |
| Training (CNN) | AdamW (lr 1e-3, weight decay 1e-4), batch size 8, 10 epochs, class-weighted binary cross-entropy, random left-right flips and shifts of up to 3 voxels, seed 42 |
| Framework | PyTorch 2.14 (CNN), scikit-learn 1.8 (baselines), Optuna 5 (tuning) |
| Code | `src/models/cnn.py`, `src/models/train_cnn.py`, `src/models/tune.py`, `notebooks/02_baseline_models.ipynb` |
| Weights | `models/cnn_mri.pt`, `models/cnn_multimodal.pt` (fit on all 178 training pairs) |
| Tracking | Weights & Biases, `ds-capstone-group-5/predicting-alzheimers-progression`, groups `baselines`, `cnn`, `tuning` |
| Developers | Christina Dames, Cinthia Ochoa Torre, Rayna Alexandre-Petrie (DSCI 4350, Group 5) |

## Intended use

- **Primary use:** academic comparison of how much predictive value comes from MRI, patient metadata, and their combination.
- **Users:** the project team and course instructors.
- **Out of scope:** clinical diagnosis, prognosis for an individual patient, screening, or any treatment decision. The models are research prototypes trained on 178 visit pairs from one research site.

## Factors

- **Groups evaluated:** sex (female, male) and age band at visit t (<70, 70-79, 80+).
- **Other relevant factors not available:** race, ethnicity, scanner (all scans come from one 1.5 T scanner), handedness (all right-handed).
- **Instrumentation:** raw scans are not skull-stripped or registered; 19 of 373 sessions are flagged for low SNR or head motion and are kept.

## Metrics

- **Primary:** ROC-AUC, threshold-free and robust to the 14% positive rate.
- **Secondary:** PR-AUC, recall, precision, and F1 at a 0.5 threshold; recall matters most because a missed decline is the costly error. Accuracy is reported only for completeness (always predicting "stable" gives 86%).
- **Calibration and fairness (monitoring):** Brier score, 5-bin calibration table, ROC-AUC by subgroup (`src/monitoring/drift.py`).
- **Variation:** pooled out-of-fold predictions over 5 patient-grouped folds. With 25 declines, differences below about 0.05 ROC-AUC are within noise.

## Evaluation data

Patient-grouped 5-fold cross-validation on the training split: 178 visit pairs, 119 patients, 25 declines (14.0%). No patient appears in more than one fold. The 45-pair test split (31 patients, 6 declines) is reserved for the final comparison.

## Training data

`data/processed/oasis2_visit_pairs.csv` (dataset card version 2.0, `reports/dataset_card.md`), training split only, plus the MRI volumes of each pair's earlier visit. Metadata features are standardized with training-fold statistics inside each fold.

## Quantitative analyses

### Overall (patient-grouped 5-fold CV, training set)

| Model | Inputs | ROC-AUC | PR-AUC | Recall | Precision | F1 | Accuracy |
|---|---|---|---|---|---|---|---|
| Dummy (prior) | none | 0.496 | 0.140 | 0.00 | 0.000 | 0.000 | 0.860 |
| Logistic regression | metadata | 0.578 | 0.180 | 0.40 | 0.152 | 0.220 | 0.601 |
| Random forest | tabular MRI measures | 0.620 | 0.228 | 0.56 | 0.250 | 0.346 | 0.702 |
| Gradient boosting, tuned (30 Optuna trials) | metadata | 0.655 | 0.286 | 0.40 | 0.222 | 0.286 | 0.719 |
| Gradient boosting, tuned | metadata + MRI scan measure | 0.648 | 0.314 | 0.28 | 0.333 | 0.304 | 0.820 |
| **Model 1:** 3D CNN | MRI volume | 0.527 | 0.187 | 0.52 | 0.133 | 0.211 | 0.455 |
| **Model 3:** 3D CNN + metadata | MRI volume + metadata | 0.517 | 0.203 | 0.48 | 0.148 | 0.226 | 0.539 |

Untuned baselines come from `notebooks/02_baseline_models.ipynb` (`reports/baseline_results.csv`); tuned ones from `reports/tuning_results.csv` (optimistic: the same folds chose the hyperparameters); CNNs from `reports/cnn_results.csv`. A positive rate of 14% makes 0.14 the PR-AUC of a random ranking.

### Subgroups (CNN out-of-fold predictions)

| Model | Female | Male | Age <70 | Age 70-79 | Age 80+ | Brier | Monitoring status |
|---|---|---|---|---|---|---|---|
| Model 1 (MRI-only) | 0.617 | 0.416 | 0.594 | 0.444 | 0.625 | 0.275 | Alert: ROC-AUC < 0.60, subgroup gap 0.20 |
| Model 3 (multimodal) | 0.592 | 0.429 | 0.566 | 0.400 | 0.644 | 0.274 | Alert: ROC-AUC < 0.60, subgroup gap 0.24 |

ROC-AUC per subgroup from `reports/monitoring_report.json`. Each subgroup holds only 5 to 14 declines, so these gaps are not statistically reliable, but they are recorded so later versions can be compared. The Brier score is worse than predicting the base rate (0.12) because the class-weighted loss pushes the predicted risks up; the risks need recalibration (for example Platt scaling inside the CV) before they can be read as probabilities.

### Compute

| Model | Mean epoch (s) | 5-fold CV, 10 epochs (min) | Hardware |
|---|---|---|---|
| Model 1 (MRI-only) | 7.5 | 6.3 | Apple M2 CPU, 8 GB RAM |
| Model 3 (multimodal) | 10.8 | 9.0 | Apple M2 CPU, 8 GB RAM |

## Ethical considerations

- OASIS data are de-identified and used under the OASIS Data Use Agreement; raw scans are not redistributed.
- A false "stable" prediction could falsely reassure a patient; a false "decline" prediction could cause distress. The models are not accurate enough for any individual use, and the intended use excludes it.
- Subgroup sizes are small (for example, 25 declines in total), so subgroup metrics cannot rule out unequal performance.

## Caveats and recommendations

- **Current CNN prototypes do not beat the tabular baselines.** With 178 training volumes, a CNN trained from scratch mostly fits noise. Next steps: tuning (`reports/experiment_plan.md`), skull stripping and registration, and pretrained encoders.
- The 0.5 threshold is not tuned; recall and precision change with it. Use ROC-AUC and PR-AUC to compare models.
- CDR moves in steps of 0.5, so "decline" is coarse; five pairs with a CDR decrease are labeled stable.
- Prediction horizon varies (182 to 1,707 days between visits); `days_to_next` is an input of the metadata models but not of Model 1.
