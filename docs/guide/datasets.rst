.. _datasets:

========
Datasets
========

Everything in the previous stages needs a cost matrix. Writing one for your own problem is the
subject of :ref:`cost_matrix`; these datasets come with theirs already written.

Empulse ships loaders for cost-sensitive datasets, for benchmarking and for the examples
throughout this documentation. Each one ships not just features and a target, but a
:class:`~empulse.metrics.CostMatrix` encoding the business problem it came from — which is what
makes them useful for value-driven work, where a plain feature matrix is not enough. They are also
the fastest way to see a realistic cost matrix that somebody else had to derive.

Every loader returns a :class:`~empulse.datasets.Dataset` and takes a required, keyword-only
``backend`` argument. Pass the dataframe library's *module* itself, not a string — neither pandas
nor polars is a hard dependency of Empulse, so the loader takes the one you already have.

.. tab-set::

    .. tab-item:: pandas
        :sync: pandas

        .. code-block:: python

            import pandas as pd
            from empulse.datasets import fetch_iranian_churn

            dataset = fetch_iranian_churn(backend=pd)
            X, y = dataset.data, dataset.target

    .. tab-item:: polars
        :sync: polars

        .. code-block:: python

            import polars as pl
            from empulse.datasets import fetch_iranian_churn

            polars_dataset = fetch_iranian_churn(backend=pl)
            X_polars, y_polars = polars_dataset.data, polars_dataset.target

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
      - 4.79%
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
    * - :ref:`Telco customer churn <telco_customer_churn>`
      - 7,032
      - 19
      - 26.6%
      - Downloaded
      - Symbolic, per-customer monthly charge
    * - :ref:`Cell2Cell churn <cell2cell>`
      - 50,891
      - 56
      - 28.8%
      - Downloaded
      - Symbolic, CLV from monthly revenue
    * - :ref:`KDD Cup 2009 churn <kddcup09_churn>`
      - 50,000
      - 230
      - 7.3%
      - Downloaded
      - Symbolic, constant CLV
    * - :ref:`KDD Cup 1998 direct mailing <kdd98>`
      - 191,779
      - 22
      - 5.1%
      - Downloaded
      - Symbolic, per-donor donation
    * - :ref:`Credit card fraud <credit_card_fraud>`
      - 282,982
      - 29
      - 0.16%
      - Downloaded
      - Symbolic, per-transaction amount
    * - :ref:`IEEE-CIS fraud detection <ieee_fraud_detection>`
      - 590,540
      - 431
      - 3.5%
      - Downloaded
      - Symbolic, per-transaction amount
    * - :ref:`VUB credit scoring <vub_credit_scoring>`
      - 18,917
      - 16
      - 16.9%
      - Bundled
      - Symbolic, per-applicant loan amount
    * - :ref:`Home equity (HMEQ) <home_equity>`
      - 5,960
      - 12
      - 19.9%
      - Downloaded
      - Symbolic, per-applicant loan amount
    * - :ref:`South German credit <south_german_credit>`
      - 1,000
      - 20
      - 30.0%
      - Downloaded
      - Symbolic, per-applicant loan amount
    * - :ref:`Default of credit card clients <default_credit_card_clients>`
      - 30,000
      - 23
      - 22.1%
      - Downloaded
      - Symbolic, per-client credit limit

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
    :hidden:

    datasets/iranian_churn.rst
    datasets/churn_tv_subscriptions.rst
    datasets/bank_telemarketing.rst
    datasets/credit_scoring_pakdd.rst
    datasets/give_me_some_credit.rst
    datasets/vub_credit_scoring.rst
    datasets/telco_customer_churn.rst
    datasets/cell2cell.rst
    datasets/kddcup09_churn.rst
    datasets/kdd98.rst
    datasets/credit_card_fraud.rst
    datasets/ieee_fraud_detection.rst
    datasets/home_equity.rst
    datasets/south_german_credit.rst
    datasets/default_credit_card_clients.rst
