.. _model_independent_preprocessing:

=========================
Changing the data instead
=========================

An easy way to make an existing cost-insensitive model cost-sensitive is to preprocess the data.
Nothing about the estimator changes; the training set it sees does.

The two techniques come from different literatures. Cost-proportionate sampling draws directly from
the cost matrix. Bias mitigation takes a page from fairness research and removes the model's bias
against a subgroup, which you can define strategically to serve a business goal — in a churn
problem, for instance, customers with a high lifetime value.

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: Cost sensitivity and class imbalance
        :link: imbalance
        :link-type: ref

        How class imbalance relates to cost asymmetry, and where resampling sits next to changing
        the objective or the threshold.

    .. grid-item-card:: Cost-Proportionate Sampling
        :link: cost_sampling
        :link-type: ref

        Resample the training set in proportion to the cost matrix, making any estimator
        cost-sensitive.

    .. grid-item-card:: Bias Mitigation
        :link: bias_mitigation
        :link-type: ref

        Remove the model's bias against a subgroup you define — high-value customers, for
        instance.

.. toctree::
    :maxdepth: 2
    :hidden:

    preprocessing/imbalance.rst
    preprocessing/cost_sampling.rst
    preprocessing/bias_mitigation.rst
