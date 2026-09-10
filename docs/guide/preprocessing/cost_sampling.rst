.. _cost_sampling:

===========================
Cost-Proportionate Sampling
===========================

The :class:`~empulse.samplers.CostSensitiveSampler` lets you resample the training data to minimize the
:func:`~empulse.metrics.cost_loss`.
Two sampling techniques are available:
1. Rejection sampling
2. Oversampling

Rejection Sampling
==================

Cost-proportionate rejection sampling allows you to draw examples independently from a distribution.
That is done by first drawing samples for the distribution and then keeping (or accepting) the samples
with a probability proportional to their cost [1]_.

This can be done using the :class:`~empulse.samplers.CostSensitiveSampler`
with the ``method`` parameter set to 'rejection sampling'.

.. code-block:: python

    from sklearn.datasets import make_classification
    from empulse.samplers import CostSensitiveSampler

    X, y = make_classification(random_state=42)

    sampler = CostSensitiveSampler(method='rejection sampling')
    X_resampled, y_resampled = sampler.fit_resample(X, y)

Oversampling
============

Cost-proportionate oversampling allows you to draw examples from a distribution with replacement.
How many times an example is drawn is proportional to its cost [2]_.
To configure the degree of oversampling, you can set the ``oversampling_norm`` parameter.
The smaller the oversampling norm, the more oversampling is done.

To indicate that you want to use oversampling, set the ``method`` parameter to 'oversampling'.

.. code-block:: python

    from sklearn.datasets import make_classification
    from empulse.samplers import CostSensitiveSampler

    X, y = make_classification(random_state=42)

    sampler = CostSensitiveSampler(method='oversampling', oversampling_norm=0.2)
    X_resampled, y_resampled = sampler.fit_resample(X, y)

Outlier Robustness
==================

When computing the probability of keeping a sample (in the case of rejection sampling) or
the number of times a sample is drawn (in the case of oversampling),
costs above the 97.5th percentile are truncated to the 97.5th percentile to decrease outlier influence.
If you wish to change this behavior, you can set the ``percentile_threshold`` parameter to any number between 0-1.

.. code-block:: python

    from empulse.samplers import CostSensitiveSampler

    sampler = CostSensitiveSampler(method='rejection sampling', percentile_threshold=0.9)
    X_resampled, y_resampled = sampler.fit_resample(X, y)


Sampling from a cost matrix
===========================

The examples above pass the four cost terms directly. The sampler also accepts a
:class:`~empulse.metrics.Metric` as ``loss``, in which case its cost matrix decides the sampling
weights and its symbols become ``fit_resample`` parameters — the same two-track API the models use,
described in :ref:`specifying_costs`.

.. code-block:: python

    import numpy as np
    from empulse.metrics import Cost, CostMatrix, Metric
    from empulse.samplers import CostSensitiveSampler

    matrix = (
        CostMatrix()
        .add_fn_cost('clv')
        .add_fp_cost('contact_cost')
        .set_default(contact_cost=1)
    )

    clv = np.random.default_rng(0).uniform(50, 500, size=len(y))
    sampler = CostSensitiveSampler(loss=Metric(matrix, Cost()), random_state=42)
    X_resampled, y_resampled = sampler.fit_resample(X, y, clv=clv)

    print(X_resampled.shape)

When a ``loss`` is set, the plain ``fp_cost`` and ``fn_cost`` constructor arguments are ignored.

Using the Cost-Sensitive Sampler in a Pipeline
==============================================

This sampler can easily be used inside an imbalanced-learn :class:`imblearn:imblearn.pipeline.Pipeline`
(note that the scikit-learn :class:`sklearn:sklearn.pipeline.Pipeline` does not support samplers):.

.. code-block:: python

    from imblearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from empulse.samplers import CostSensitiveSampler

    pipeline = Pipeline([
        ('sampler', CostSensitiveSampler(method='rejection sampling')),
        ('classifier', LogisticRegression())
    ])
    pipeline.fit(X, y)

Per-sample costs inside a cross-validation loop have to be sliced per fold, which the sampler
supports through ``set_fit_resample_request`` — see :ref:`instance_based_cv`.

References
==========

.. [1] B. Zadrozny, J. Langford, N. Naoki, "Cost-sensitive learning by
       cost-proportionate example weighting", in Proceedings of the
       Third IEEE International Conference on Data Mining, 435-442, 2003.

.. [2] C. Elkan, "The foundations of Cost-Sensitive Learning",
       in Seventeenth International Joint Conference on Artificial Intelligence,
       973-978, 2001.