.. module:: empulse.samplers

empulse.samplers
================

The :mod:`~empulse.samplers` module contains a collection of samplers based on
`Imbalanced-Learn <https://imbalanced-learn.org/stable/introduction.html#api-s-of-imbalanced-learn-samplers>`_.
They need the ``sampling`` extra: ``pip install empulse[sampling]``.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: base.rst

   BiasRelabler
   BiasResampler
   CostSensitiveSampler
