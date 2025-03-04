.. module:: empulse.models

==============
empulse.models
==============

The :mod:`~empulse.models` module contains a collection of cost-sensitive and profit-driven models.

Cost-Sensitive and Value-driven Models
======================================

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: base.rst

    B2BoostClassifier
    CSBaggingClassifier
    CSBoostClassifier
    CSForestClassifier
    CSLogitClassifier
    CSThresholdClassifier
    CSTreeClassifier
    RobustCSClassifier
    ProfLogitClassifier

Bias Mitigation Models
======================

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: base.rst

    BiasRelabelingClassifier
    BiasResamplingClassifier
    BiasReweighingClassifier

