Predicting Cognitive Decline from Longitudinal MRI
==================================================

Can longitudinal brain MRI and patient metadata from an earlier visit predict
whether a patient's Clinical Dementia Rating (CDR) increases by the next
visit? This site documents the code that builds the OASIS-2 visit-pair
dataset, trains the models, and monitors data and model quality.

.. toctree::
   :maxdepth: 2

   pipeline
   api

Project documents
-----------------

* ``reports/dataset_card.md``: dataset card (lineage, composition, limitations).
* ``reports/model_card.md``: model card for the baselines and CNN models.
* ``reports/experiment_plan.md``: validation, tuning, and compute plan.
* ``reports/governance_plan.md``: monitoring metrics, alert thresholds,
  documentation style.
