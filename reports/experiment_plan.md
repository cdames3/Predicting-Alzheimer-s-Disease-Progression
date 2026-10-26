# Experiment Plan

## 1. Models

| # | Model | Inputs | Family | Code |
|---|---|---|---|---|
| 1 | MRI-only | First T1 scan of visit t, 64 x 64 x 64 | 3D CNN, 4 conv blocks (8-16-32-64 channels), about 73k parameters | `src/models/cnn.py`, `src/models/train_cnn.py --model mri` |
| 2 | Metadata-only | 18 clinical, demographic, timing, and change features | Logistic regression, random forest, gradient boosting | `notebooks/02_baseline_models.ipynb`, `src/models/tune.py` |
| 3 | Multimodal | Model 1 inputs + Model 2 features | Same 3D CNN + a 16-unit metadata branch, late fusion | `src/models/train_cnn.py --model multimodal` |
| - | Floor | None | Prior-only dummy classifier | notebook |

**Loss functions.** Binary cross-entropy for every model. The positive class is weighted by the stable/declined ratio (about 6:1): `class_weight="balanced"` for scikit-learn models, `pos_weight` in `BCEWithLogitsLoss` for the CNNs.

**Metrics.** ROC-AUC is the primary metric (threshold-free, robust to the 14% positive rate). PR-AUC, recall, precision, and F1 at a 0.5 threshold are reported alongside; recall matters most because a missed decline is the costly error. Accuracy is reported but not used for selection (the dummy model reaches 86%).

## 2. Validation plan

1. **Patient-level holdout.** 31 patients (45 pairs, 6 declines) are held out in `split = "test"`. They are used **once**, at the end, to compare the selected models. No tuning or model selection looks at them.
2. **Model selection: patient-grouped 5-fold CV** on the 119 training patients (`cv_fold` column, stratified on the target). All visit pairs of a patient stay in the same fold, so the models are always scored on unseen patients.
3. **Pooled out-of-fold scoring.** Each fold has only 5 declines, so per-fold ROC-AUC is very noisy. Metrics are computed once on the pooled out-of-fold predictions, the same way for every model (`src.models.evaluate.evaluate_model` and `src.models.train_cnn.classification_metrics`).
4. **No early stopping on validation folds.** CNNs train for a fixed number of epochs per experiment, so the out-of-fold scores are not selected on the data they are measured on. The number of epochs is a tuned hyperparameter instead.
5. **Uncertainty.** With 25 training declines, differences of a few ROC-AUC points are within noise. The final comparison will report 1,000-sample patient-level bootstrap confidence intervals and repeat the CNN runs with 3 seeds.
6. **Sensitivity analysis.** Re-score the selected models without the QC-flagged MRI sessions (`mri_qc_flag = 1`).

## 3. Hyperparameter optimization plan

**Tabular baselines (Model 2):** Optuna with the TPE sampler, 30 trials per model and feature set, objective = pooled out-of-fold ROC-AUC (`python -m src.models.tune`).

| Model | Search space |
|---|---|
| Logistic regression | C in [0.001, 10] (log), penalty L1 or L2 |
| Random forest | trees 100-600, max depth 2-12, min samples per leaf 1-20, max features 0.2-1.0 |
| Gradient boosting | learning rate 0.01-0.3 (log), iterations 50-400, max depth 2-6, min samples per leaf 5-40, L2 0.0001-10 (log) |

**CNN models (Models 1 and 3):** the same Optuna setup over a smaller space, 20 trials per model, because each trial is a full 5-fold CV:

| Hyperparameter | Values |
|---|---|
| Learning rate | 1e-4 to 3e-3 (log) |
| Epochs | 10, 20, 30 |
| Weight decay | 1e-5 to 1e-2 (log) |
| Dropout | 0.1 to 0.5 |
| Channel widths | (8, 16, 32, 64) or (16, 32, 64, 128) |
| Batch size | 8, 16 |

Ideas to test after tuning, one at a time: skull stripping and registration of the volumes, 2D slice models with an ImageNet-pretrained backbone, and freezing the MRI encoder in Model 3 (training only the fusion head) to reduce overfitting.

## 4. Compute needs and time estimates

Measured on the development laptop (Apple M2, 8 cores, 8 GB RAM). The CNNs run on the CPU: PyTorch's Apple-GPU (MPS) kernels for 3D convolution were 2.5x slower than the CPU for this network.

| Step | Measured | Notes |
|---|---|---|
| Build MRI tables (`make_mri_dataset`) | 2-5 min | Streams 19.3 GB of archives once |
| Build CNN volumes (`make_mri_volumes`) | 2.0 min | 373 volumes, 135 MB |
| Baseline CV, one model | < 5 s | 178 pairs, 5 folds |
| Tuning, one trial | 0.1 s (logistic regression), 1.3-2.3 s (random forest), 1.6-3.9 s (gradient boosting) | 5-fold CV of one configuration |
| CNN epoch (about 142 training pairs) | 7.5 s (Model 1), 10.8 s (Model 3) | Batch size 8, CPU |
| CNN 5-fold CV, 10 epochs | 6.3 min (Model 1), 9.0 min (Model 3) | 50 epochs in total |

**Estimates for the plan:**

| Experiment | Runs | Estimated time |
|---|---|---|
| Baseline tuning: 3 models x 2 feature sets x 30 trials | 180 CV runs | 4.6 min (measured) |
| CNN tuning: 2 models x 20 trials, 10-30 epochs (mean 20) | 40 CV runs, about 4,000 epochs | about 10 hours on the laptop (9 s per epoch) |
| Final runs: 3 seeds x 3 models + bootstrap | 9 runs | about 1.5 hours |

The CNN tuning is the bottleneck. Options if the laptop is too slow: run trials overnight, or move them to Google Colab (free T4 GPU), where a 64^3 3D CNN of this size typically trains 5-10x faster. Memory is not a constraint: all 178 training volumes take 190 MB as float32.

## 5. Platform

- **Code and data versioning:** GitHub (this repository) + SHA-256 hashes in the metadata JSON files.
- **Training:** local laptop (CPU) for baselines and the first CNN runs; Google Colab GPU for CNN tuning if needed.
- **Experiment tracking:** Weights & Biases, team project `ds-capstone-group-5/predicting-alzheimers-progression`, groups `baselines`, `cnn`, and `tuning`. Each run logs its configuration, per-epoch losses, metrics, and timing.
- **Tuning:** Optuna (TPE sampler, fixed seed).

## 6. Initial goals

| Benchmark | Current result (grouped CV) | Goal |
|---|---|---|
| Floor (dummy) | ROC-AUC 0.496 | Every model must beat it |
| Best untuned baseline | ROC-AUC 0.620, F1 0.346 (random forest, tabular MRI measures) | Reference for the CNNs |
| Best tuned baseline (Model 2) | ROC-AUC 0.655, F1 0.286 (gradient boosting, metadata-only; optimistic, see section 3) | Confirm on the test set |
| Model 1 (CNN, MRI-only) | ROC-AUC 0.527, F1 0.211, recall 0.52 | Reach 0.62 (the best untuned baseline) |
| Model 3 (CNN + metadata) | ROC-AUC 0.517, F1 0.226, recall 0.48 | Beat Models 1 and 2 by at least 0.03 ROC-AUC, with recall of 0.5 or more |

**Where we stand.** Tuning raised the best tabular ROC-AUC from 0.620 to 0.655. The first CNN prototypes are barely above chance: per-fold validation ROC-AUC swings from 0.27 to 0.64, training loss falls while validation loss does not, and the CNNs predict "declined" for about half of the pairs (calibration is poor). With 25 declines in 178 training pairs, a CNN trained from scratch has too little signal. The next experiments, in order: tune learning rate and epochs, test registration and skull stripping, and test a frozen pretrained encoder.
