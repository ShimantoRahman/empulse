============================================
Cross-Validation with Instance Based Metrics
============================================

Some of the value metrics (e.g., :func:`empulse.metrics.empb_score`)
allow users to pass instance based weights for the metrics.
This is usually no issue when just doing a simple train-val-test split.
However, when using cross-validation, the weights per for fold change.
As of scikit-learn 1.4.0, some cross-validation methods allow for
`metadata routing <https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_metadata_routing.html>`_
which enables passing instance based weights to scorers that are also split accordingly.
To see which cross-validation methods support metadata routing, see
`here <https://scikit-learn.org/stable/metadata_routing.html#metadata-routing-models>`_.
This is an experimental feature and currently has to be enabled manually.

.. code-block:: python

    from sklearn import set_config

    set_config(enable_metadata_routing=True)

After enabling metarouting for scikit-learn, the following example shows how to use
instance based weights with cross-validation.

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split

    from empulse.metrics import empb_score

    X, y = make_classification(n_samples=100, n_features=10, n_informative=2, n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42)

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