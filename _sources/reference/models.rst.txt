.. module:: empulse.models

======
Models
======

The :mod:`~empulse.models` module contains a collection of cost-sensitive and profit-driven models.

Cost-Sensitive Models
=====================
.. list-table::
   :widths: 25 75

   * - :class:`B2BoostClassifier`
     - :class:`xgboost:xgboost.XGBClassifier` to optimize instance-specific cost loss for customer churn.
   * - :class:`CSBoostClassifier`
     - :class:`xgboost:xgboost.XGBClassifier` to optimize instance-specific cost loss.
   * - :class:`CSLogitClassifier`
     - Logistic classifier to optimize instance-specific cost loss.
   * - :class:`CSThresholdClassifier`
     - Classifier which sets the decision threshold to optimize the instance-specific cost loss.
   * - :class:`RobustCSClassifier`
     - Classifier that fits a cost-sensitive classifier with costs adjusted for outliers.

Profit-Driven Models
====================

.. list-table::
   :widths: 25 75

   * - :class:`B2BoostClassifier`
     - :class:`xgboost:xgboost.XGBClassifier` to optimize instance-specific cost loss for customer churn.
   * - :class:`ProfLogitClassifier`
     - Logistic classifier to optimize profit-driven score.

Bias Mitigation Models
======================

.. list-table::
   :widths: 25 75

   * - :class:`BiasRelabelingClassifier`
     - Classifier which relabels instances during training to remove bias against a subgroup.
   * - :class:`BiasResamplingClassifier`
     - Classifier which resamples instances during training to remove bias against a subgroup.
   * - :class:`BiasReweighingClassifier`
     - Classifier which reweighs instances during training to remove bias against a subgroup.

