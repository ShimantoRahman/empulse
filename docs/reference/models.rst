.. module:: empulse.models

Models
======

The :mod:`~empulse.models` module contains a collection of profit-driven models.

.. list-table::
   :widths: 25 75

   * - :class:`B2BoostClassifier`
     - :class:`xgboost:xgboost.XGBClassifier` with instance-specific cost loss for customer churn.
   * - :class:`BiasRelabelingClassifier`
     - Classifier which relabels instances during training to remove bias against a subgroup.
   * - :class:`BiasResamplingClassifier`
     - Classifier which resamples instances during training to remove bias against a subgroup.
   * - :class:`BiasReweighingClassifier`
     - Classifier which reweighs instances during training to remove bias against a subgroup.
   * - :class:`CSBoostClassifier`
     - :class:`xgboost:xgboost.XGBClassifier` with instance-specific cost loss.
   * - :class:`CSLogitClassifier`
     - Logistic classifier with instance-specific cost loss.
   * - :class:`CSThresholdClassifier`
     - Classifier which sets the decision threshold to minimize the cost loss.
   * - :class:`ProfLogitClassifier`
     - Logistic classifier to optimize profit-driven score.
