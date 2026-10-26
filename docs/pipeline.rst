Pipeline
========

Run the steps from the repository root, in this order.

.. code-block:: bash

   pip install -r requirements.txt
   python -m src.data.make_mri_dataset      # MRI scan and session tables (~2-5 min)
   python -m src.data.make_dataset          # ML-ready visit pairs
   python -m src.data.make_mri_volumes      # 64^3 CNN inputs (~2 min)
   python -m src.visualization.eda_report   # EDA report and figures
   python -m src.models.train_cnn --model mri         # Model 1 (MRI-only CNN)
   python -m src.models.train_cnn --model multimodal  # Model 3 (CNN + metadata)
   python -m src.models.tune --trials 30    # tune the tabular baselines
   python -m src.monitoring.drift           # data quality, drift, model checks

The baseline models (Model 2, metadata-only) are in
``notebooks/02_baseline_models.ipynb`` and share
:func:`src.models.evaluate.evaluate_model`.

Experiment tracking
-------------------

Training and tuning runs are logged to the team Weights & Biases project
(``ds-capstone-group-5/predicting-alzheimers-progression``). Runs default to
offline mode; after ``wandb login``, upload them with ``wandb sync wandb/``
or pass ``--wandb online``.
