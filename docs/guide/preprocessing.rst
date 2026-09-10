.. _model_independent_preprocessing:

=========================
Changing the data instead
=========================

An easy way to make an existing cost-insensitive model cost-sensitive is to preprocess the data.
Nothing about the estimator changes; the training set it sees does.

:doc:`preprocessing/imbalance` places these techniques next to the alternatives — changing the
objective (:doc:`training`) or changing the threshold (:doc:`deciding`) — and covers how class
imbalance relates to cost asymmetry.

The two techniques themselves come from different literatures. Cost-proportionate sampling draws
directly from the cost matrix. Bias mitigation takes a page from fairness research and removes the
model's bias against a subgroup, which you can define strategically to serve a business goal — in a
churn problem, for instance, customers with a high lifetime value.

.. toctree::
    :maxdepth: 2

    preprocessing/imbalance.rst
    preprocessing/cost_sampling.rst
    preprocessing/bias_mitigation.rst
