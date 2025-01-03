.. module:: empulse.samplers

Samplers
========

The :mod:`~empulse.samplers` module contains a collection of samplers based on
`Imbalanced-Learn <https://imbalanced-learn.org/stable/introduction.html#api-s-of-imbalanced-learn-samplers>`_.

.. list-table::
   :widths: 15 60

   * - :class:`BiasRelabler`
     - Sampler which relabels instances to remove bias against a subgroup.
   * - :class:`BiasResampler`
     - Sampler which resamples instances to remove bias against a subgroup.
   * - :class:`CostSensitiveSampler`
     - Sampler which performs cost-proportionate resampling.