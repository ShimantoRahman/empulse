.. _installation:

============
Installation
============

Empulse requires **Python 3.11 or higher**.

.. code-block:: bash

    pip install empulse

That gives you every metric, the linear and tree-based models, the samplers and the optimizers.

Optional extras
===============

Two features need extra dependencies, both installable as extras:

.. list-table::
    :widths: 22 30 48
    :header-rows: 1

    * - Extra
      - Install
      - What it unlocks
    * - ``optional``
      - ``pip install empulse[optional]``
      - XGBoost, LightGBM and CatBoost, the backends behind
        :class:`~empulse.models.CSBoostClassifier` and
        :class:`~empulse.models.B2BoostClassifier`.
    * - ``symbolic``
      - ``pip install empulse[symbolic]``
      - gplearn, required by :class:`~empulse.models.ProfSRClassifier`
        (genetic-programming symbolic regression).

To install everything:

.. code-block:: bash

    pip install empulse[optional,symbolic]

If you use a backend that is not installed, Empulse raises an error telling you exactly what to
install — nothing fails silently.

Dataframe support
=================

The bundled datasets in :mod:`empulse.datasets` return dataframes, so they need **pandas or
polars** installed. Neither is a hard dependency of Empulse itself, because the rest of the package
works fine on plain NumPy arrays.

.. code-block:: bash

    pip install pandas

Every loader takes a required, keyword-only ``backend`` argument — you pass the library module
itself, which is how Empulse stays agnostic between pandas and polars:

.. code-block:: python

    import pandas as pd
    from empulse.datasets import fetch_iranian_churn

    dataset = fetch_iranian_churn(backend=pd)

Verifying the install
=====================

.. code-block:: python

    import empulse

    print(empulse.__version__)

Note that Empulse has no top-level re-exports: import from the submodules
(:mod:`empulse.metrics`, :mod:`empulse.models`, :mod:`empulse.samplers`,
:mod:`empulse.optimizers`, :mod:`empulse.datasets`) rather than from ``empulse`` directly.

Next steps
==========

- :doc:`quickstart` — a working cost-sensitive model in five minutes.
- :doc:`overview` — which part of Empulse solves your problem.
