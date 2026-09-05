.. _datasets:

========
Datasets
========

Empulse bundles five real-world cost-sensitive datasets for benchmarking and for the examples
throughout this documentation. Each one ships not just features and a target, but a
:class:`~empulse.metrics.CostMatrix` encoding the business problem it came from — which is what
makes them useful for value-driven work, where a plain feature matrix is not enough.

Every loader returns a :class:`~empulse.datasets.Dataset` and takes a required, keyword-only
``backend`` argument naming the dataframe library to use:

.. code-block:: python

    import pandas as pd
    from empulse.datasets import fetch_iranian_churn

    dataset = fetch_iranian_churn(backend=pd)
    X, y = dataset.data, dataset.target

Choosing a dataset
==================

.. list-table::
    :widths: 26 12 10 12 18 22
    :header-rows: 1

    * - Dataset
      - Samples
      - Features
      - Positives
      - Availability
      - Cost matrix
    * - :ref:`Iranian churn <iranian_churn>`
      - 3,150
      - 12
      - 15.7%
      - Downloaded
      - Symbolic, per-customer CLV
    * - :ref:`Churn TV subscriptions <churn_tv_subscriptions>`
      - 9,379
      - 46
      - 4.8%
      - Bundled
      - Precomputed, all four terms
    * - :ref:`Bank telemarketing upsell <upsell_bank_telemarketing>`
      - 37,931
      - 10
      - 12.6%
      - Bundled
      - Symbolic, per-customer balance
    * - :ref:`Credit scoring PAKDD <credit_scoring_pakdd>`
      - 38,938
      - 25
      - 19.9%
      - Bundled
      - Symbolic, per-applicant credit line
    * - :ref:`Give Me Some Credit <give_me_some_credit>`
      - 112,915
      - 10
      - 6.7%
      - Downloaded
      - Symbolic, per-applicant credit line

**Bundled** datasets ship inside the package and work offline. **Downloaded** datasets are fetched
on first use and cached under ``~/empulse_data`` (override with ``$EMPULSE_DATA_HOME`` or the
``data_home`` argument), so only the first call needs a network connection.

A note on cost matrices
=======================

The distinction in the last column matters more than it looks.

Most of these datasets carry a **symbolic** cost matrix: the business parameters stay named, with
defaults, so you can override them at call time (``accept_rate=0.5``) and ask what-if questions
without rebuilding anything. Only the genuinely per-row quantities — a customer's lifetime value,
an applicant's credit line — arrive as arrays in ``instance_costs``.

:func:`~empulse.datasets.load_churn_tv_subscriptions` is the exception: it ships **precomputed**
costs, with all four outcome terms supplied directly as arrays. That makes it a natural fit for the
models' plain cost arguments, but its business parameters cannot be varied after the fact.

Where next
==========

The :doc:`../tutorial` uses the Iranian churn dataset end to end, and is the best place to see one
of these datasets put to work.

.. toctree::
    :maxdepth: 2
    :glob:

    datasets/*
