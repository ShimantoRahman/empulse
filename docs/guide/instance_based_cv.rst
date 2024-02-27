.. _instance_based_cv:

Cross-Validation with Instance Based Metrics
============================================

Instance-based metrics in the empulse library,
such as :func:`empulse.metrics.empb_score`,
provide the flexibility to incorporate instance-based weights into the metrics.
This feature is particularly useful when dealing with imbalanced datasets or
when different instances have different importance.

In a simple train-validation-test split scenario,
using instance-based weights is straightforward.
However, when performing cross-validation, the weights for each fold change, which requires special handling.

As of scikit-learn 1.4.0,
some cross-validation methods support
`metadata routing <https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_metadata_routing.html>`_.
This feature allows instance-based weights to be passed to scorers,
and these weights are split accordingly for each fold.
For a list of cross-validation methods that support metadata routing,
refer to `this link <https://scikit-learn.org/stable/metadata_routing.html#metadata-routing-models>`_.
Please note that metadata routing is an experimental feature and needs to be enabled manually.


.. code-block:: python

    from sklearn import set_config

    set_config(enable_metadata_routing=True)

The above code snippet enables metadata routing in scikit-learn.
Once this is done, instance-based weights can be used with cross-validation.
The following example demonstrates this.

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split

    from empulse.metrics import empb_score

    X, y = make_classification()
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y)

    estimator = LogisticRegression()

    # define EMPB parameters
    clv = np.random.rand(100) * 200
    alpha = 6
    beta = 14
    incentive_cost_fraction = 0.05
    contact_cost = 15

    scoring = make_scorer(
        empb_score,
        greater_is_better=True,
        response_method='predict_proba',
        alpha=alpha,  # pass fixed EMPB parameters to scorer
        beta=beta,
        incentive_cost_fraction=incentive_cost_fraction,
        contact_cost=contact_cost,
    ).set_score_request(clv=True)  # enable passing of clv to scorer

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cross_val_score(estimator, X, y, cv=cv, scoring=scoring, params={"clv": clv})  # pass clv to cross_val_score

In the above example, we first create a classification dataset using
:py:func:`~sklearn:sklearn.datasets.make_classification`.
We then split the dataset into training and testing sets.
We define an estimator using :py:class:`~sklearn:sklearn.linear_model.LogisticRegression`.

Next, we define the parameters for the :func:`~empulse.metrics.empb_score` metric,
including ``clv``, ``alpha``, ``beta``, ``incentive_cost_fraction``, and ``contact_cost``.
We create a scorer using :py:func:`~sklearn:sklearn.metrics.make_scorer` and
set ``clv=True`` to enable the passing of ``clv`` to the scorer.

Finally, we perform cross-validation using :py:class:`~sklearn:sklearn.model_selection.StratifiedKFold` and
compute the cross-validation score using :py:func:`~sklearn:sklearn.model_selection.cross_val_score`,
passing ``clv`` as a parameter.