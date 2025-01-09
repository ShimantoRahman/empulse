.. _instance_based_cv:

==========================================
Cross-Validation with Instance-Based Costs
==========================================

Cost-sensitive, models, samplers and metrics depend on instance-based costs.
In a simple train-validation-test split scenario,
using instance-based weights is straightforward,
since they can just be passed to the ``fit``, ``fit_resample`` or ``score`` methods.
However, when performing cross-validation, the costs for each fold change, which requires special handling.

As of scikit-learn 1.4.0,
some cross-validation methods support
`metadata routing <https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_metadata_routing.html>`_.
This feature allows instance-based costs to be passed to estimators, samplers, and scorers,
and these costs are split accordingly for each fold.
For a list of cross-validation methods that support metadata routing,
refer to `this link <https://scikit-learn.org/stable/metadata_routing.html#metadata-routing-models>`_.
Please note that metadata routing is an experimental feature and needs to be enabled manually.

Enabling Metadata Routing
=========================

To enable metadata routing in scikit-learn, use the following code snippet:

.. code-block:: python

    from sklearn import set_config

    set_config(enable_metadata_routing=True)

Alternatively, you can also use the context manager to not enable metadata routing globally:

.. code-block:: python

    from sklearn import config_context

    with config_context(enable_metadata_routing=True):
        # code that uses metadata routing
        ...

What is Metadata Routing
========================

A full explanation of how metadata routing works can be found in sklearn's
`User Guide <https://scikit-learn.org/stable/metadata_routing.html>`_.

But for a brief summary to be able to use the tools in Empulse,
all methods on an estimator like ``fit``, ``fit_resample``, ``score`` can request metadata.
Metadata routers like :func:`~sklearn:sklearn.model_selection.cross_val_score` or
:class:`~sklearn:sklearn.model_selection.GridSearchCV` can pass metadata to the metadata requesters.
A metadata requester can request metadata through calling ``set_***_request`` methods.

So for a cost-sensitive model can request the ``fp_cost`` metadata to its ``fit`` method like this:

.. code-block:: python

    from empulse.models import CSLogitClassifier

    cslogit = CSLogitClassifier().set_fit_request(fp_cost=True)

A sampler can request the ``fp_cost`` metadata to its ``fit_resample`` method like this:

.. code-block:: python

    from empulse.samplers import CostSensitiveSampler

    sampler = CostSensitiveSampler().set_fit_resample_request(fp_cost=True)

.. note::

    When using a sampler inside a pipeline, it should be an imbalanced-learn
    :class:`~imblearn:imblearn.pipeline.Pipeline`.
    Otherwise the parameters will not be passed to the sampler.

A scorer can request the ``fp_cost`` metadata to its ``score`` method like this:

.. code-block:: python

    from empulse.metrics import expected_savings_score
    from sklearn.metrics import make_scorer

    scorer = make_scorer(
        expected_savings_score,
        greater_is_better=True,
        response_method='predict_proba',
    ).set_score_request(fp_cost=True)

Then, when using :func:`~sklearn:sklearn.model_selection.cross_val_score` or
:class:`~sklearn:sklearn.model_selection.GridSearchCV`, you can pass the metadata to the method like this:

.. code-block:: python

    import numpy as np
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import make_classification

    X, y = make_classification()
    fp_cost = np.random.rand(X.shape[0])  # instance-dependent costs

    cross_val_score(cslogit, X, y, scoring=scorer, params={"fp_cost": fp_cost})

Now the `fp_cost` metadata will be passed to the `fit` method of the :class:`~empulse.models.CSLogitClassifier`
and the `score` method of the :func:`~empulse.metrics.expected_savings_score` scorer.


Gridsearch Example
==================

In this example we want to train a cost-sensitive logistic regression model.
We will find the best hyperparameters using a grid search optimizing the expected cost loss.
The model and scorer are set up to request the instance-dependent costs.

.. code-block:: python

    import numpy as np
    from sklearn import set_config
    from sklearn.datasets import make_classification
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.metrics import make_scorer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from empulse.models import CSLogitClassifier
    from empulse.metrics import expected_cost_loss

    set_config(enable_metadata_routing=True)

    X, y = make_classification()
    fp_cost = np.random.rand(X.shape[0])
    fn_cost = np.random.rand(X.shape[0])

    scorer = make_scorer(
        expected_cost_loss,
        greater_is_better=False,
        response_method='predict_proba',
    ).set_score_request(fp_cost=True, fn_cost=True)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', CSLogitClassifier().set_fit_request(fp_cost=True, fn_cost=True))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {'model__C': [0.1, 1]}
    grid_search = GridSearchCV(pipe, param_grid, cv=cv, scoring=scorer)
    grid_search.fit(X, y, fp_cost=fp_cost, fn_cost=fn_cost)
