.. module:: empulse.samplers

Samplers
========

The :mod:`~empulse.samplers` module contains a collection of samplers to remove bias in the training data.
The samplers are used within the :mod:`~empulse.models` module, but can also be used independently.

.. list-table::
   :widths: 15 60

   * - :class:`BiasRelabler`
     - Sampler which relabels instances to remove bias against a subgroup
   * - :class:`BiasResampler`
     - Sampler which resamples instances to remove bias against a subgroup