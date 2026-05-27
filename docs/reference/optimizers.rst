.. module:: empulse.optimizers

empulse.optimizers
==================

The :mod:`~empulse.optimizers` module contains the optimizers used to
optimize models in the :mod:`empulse.models` module.


Optimizers
==========

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: base.rst

    Optimizer
    Adam
    GeneticAlgorithmOptimizer
    LBFGSBOptimizer
    MemeticOptimizer
    RMSProp
    ScipyOptimizer
    SGD

Optimizer Components
====================

Used to construct optimizers in the :mod:`empulse.optimizers` module.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: base.rst

    Generation
    LamarckianGeneration

Schedulers
==========

Used to schedule the learning rate or smoothing parameters during training.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: base.rst

   Schedule
   ConstantSchedule
   CosineAnnealingSchedule
   ExponentialSchedule
   LinearSchedule
   StepSchedule
   WarmupSchedule