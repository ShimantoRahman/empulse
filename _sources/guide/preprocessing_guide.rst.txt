.. _model_independent_preprocessing:

===============================
Model-Independent Preprocessing
===============================

An easy way to make existing cost-insensitive models cost-sensitive is to preprocess the data.
Empulse provides a number of preprocessing techniques that can be used to make models cost-sensitive.

The first stream takes a page out of fairness literature and
uses bias-mitigation techniques to remove the bias against a subgroup.
This subgroup can strategically defined to serve your business needs.
For instance, in a customer churn case,
the subgroup can be defined as customers with high customer lifetime value (CLV).
By removing the bias, you model will target more high-CLV customers.

The second stream uses sampling techniques based on the cost matrix.

.. toctree::
    :maxdepth: 2

    preprocessing/bias_mitigation.rst
    preprocessing/cost_sampling.rst