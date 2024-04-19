.. module:: empulse.models

Models
======

The :mod:`~empulse.models` module contains a collection of profit-driven models.

.. list-table::
   :widths: 25 75

   * - :class:`B2BoostClassifier`
     - :class:`xgboost:xgboost.XGBClassifier` with instance-specific cost function for customer churn
   * - :class:`BiasRelabelingClassifier`
     - Classifier which relabels instances during training to remove bias against a subgroup
   * - :class:`BiasResamplingClassifier`
     - Classifier which resamples instances during training to remove bias against a subgroup
   * - :class:`BiasReweighingClassifier`
     - Classifier which reweighs instances during training to remove bias against a subgroup
   * - :class:`ProfLogitClassifier`
     - Logistic classifier to optimize profit-driven loss functions

