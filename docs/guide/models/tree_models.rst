.. _cstree:
.. _csforest:
.. _csbagging:
.. _proftree:

================================
Tree-Based Cost-Sensitive Models
================================

Empulse provides four tree-based classifiers that incorporate cost information
directly into the learning process.  Three of them (:class:`~empulse.models.CSTreeClassifier`,
:class:`~empulse.models.CSForestClassifier`, and :class:`~empulse.models.CSBaggingClassifier`)
use a cost-sensitive splitting criterion, so each tree node is grown by maximising
cost reduction rather than a purity measure like Gini impurity.
The fourth (:class:`~empulse.models.ProfTreeClassifier`) takes a different approach
and uses an evolutionary genetic algorithm to evolve trees that directly maximise a
user-defined profit metric.

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Model
     - Base algorithm
     - Key characteristic
   * - :class:`~empulse.models.CSTreeClassifier`
     - Decision tree
     - Single cost-sensitive tree; interpretable
   * - :class:`~empulse.models.CSForestClassifier`
     - Random forest
     - Ensemble of cost-sensitive trees; feature importances
   * - :class:`~empulse.models.CSBaggingClassifier`
     - Bagging / Pasting / Random Patches
     - Flexible ensemble; custom base estimator
   * - :class:`~empulse.models.ProfTreeClassifier`
     - Genetic algorithm
     - Evolves trees directly against a profit metric


Quick Start
===========

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from empulse.models import (
        CSTreeClassifier,
        CSForestClassifier,
        CSBaggingClassifier,
        ProfTreeClassifier,
    )

    X, y = make_classification(n_samples=1_000, random_state=0)

    # Cost-sensitive decision tree
    tree = CSTreeClassifier(fp_cost=5, fn_cost=1)
    tree.fit(X, y)

    # Cost-sensitive random forest
    forest = CSForestClassifier(n_estimators=100, fp_cost=5, fn_cost=1)
    forest.fit(X, y)

    # Cost-sensitive bagging
    bagging = CSBaggingClassifier(n_estimators=10, fp_cost=5, fn_cost=1)
    bagging.fit(X, y)

    # Profit-maximising evolutionary tree
    proftree = ProfTreeClassifier(tp_cost=300, fp_cost=10)
    proftree.fit(X, y)

    y_proba = forest.predict_proba(X)[:, 1]


Cost Matrix
===========

All four models accept the same four cost terms:

* ``tp_cost`` — benefit / cost of a true positive
* ``tn_cost`` — benefit / cost of a true negative
* ``fp_cost`` — cost of a false positive
* ``fn_cost`` — cost of a false negative

Constant costs
--------------

Pass a scalar to apply the same cost to every sample:

.. code-block:: python

    from empulse.models import CSTreeClassifier

    model = CSTreeClassifier(fp_cost=5, fn_cost=1, tp_cost=0, tn_cost=0)

Instance-dependent costs
------------------------

Pass a 1-D array of length ``n_samples`` to ``fit`` to assign a unique cost
to each individual observation:

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from empulse.models import CSForestClassifier

    X, y = make_classification(n_samples=500, random_state=0)
    clv = np.random.default_rng(0).uniform(100, 1_000, size=len(y))
    contact_cost = 10

    model = CSForestClassifier(fn_cost=1)
    model.fit(X, y, tp_cost=clv - contact_cost, fp_cost=contact_cost)

.. note::

    Costs passed to ``fit`` take priority over costs passed to ``__init__``.
    It is best practice to pass instance-dependent costs through ``fit``
    rather than the constructor, because scikit-learn cloners do not carry
    sample arrays.


Cost-Sensitive Decision Tree (CSTreeClassifier)
===============================================

:class:`~empulse.models.CSTreeClassifier` is a single decision tree whose
splitting criterion directly maximises cost savings at each node [1]_.
It wraps scikit-learn's :class:`~sklearn.tree.DecisionTreeClassifier` and
exposes the same tree structure, pruning utilities, and feature importances.

Split criterion
---------------

The ``criterion`` parameter controls how the cost signal is weighted at each split:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - ``criterion``
     - Description
   * - ``"cost"`` *(default)*
     - Pure cost reduction — splits are evaluated by the expected-cost gain.
   * - ``"gini"``
     - Cost gain is weighted by Gini impurity — blends class separation with cost.
   * - ``"entropy"`` / ``"log_loss"``
     - Cost gain is weighted by Shannon information gain.

.. code-block:: python

    from empulse.models import CSTreeClassifier

    # Default: use the raw cost impurity
    tree = CSTreeClassifier(fp_cost=5, fn_cost=1)

    # Weight by Gini impurity
    tree_gini = CSTreeClassifier(fp_cost=5, fn_cost=1, criterion="gini")

    # Weight by entropy
    tree_entropy = CSTreeClassifier(fp_cost=5, fn_cost=1, criterion="entropy")

Controlling tree size
---------------------

Use the standard scikit-learn parameters to regularise the tree:

.. code-block:: python

    tree = CSTreeClassifier(
        fp_cost=5,
        fn_cost=1,
        max_depth=5,
        min_samples_leaf=20,
        min_samples_split=50,
    )

Post-training pruning via ``ccp_alpha``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Minimal Cost-Complexity Pruning is available through the ``ccp_alpha`` parameter.
To find a good value, inspect the pruning path first:

.. code-block:: python

    from empulse.models import CSTreeClassifier
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=500, random_state=0)
    tree = CSTreeClassifier(fp_cost=5, fn_cost=1).fit(X, y)

    path = tree.cost_complexity_pruning_path(X, y)
    print(path.ccp_alphas)    # candidate alpha values
    print(path.impurities)    # impurity of the corresponding subtrees

    # Apply pruning with a chosen alpha
    pruned_tree = CSTreeClassifier(fp_cost=5, fn_cost=1, ccp_alpha=0.01).fit(X, y)

Inspecting the tree
-------------------

:class:`~empulse.models.CSTreeClassifier` exposes the underlying sklearn tree
object and several inspection helpers:

.. code-block:: python

    tree.fit(X, y)

    print(tree.get_depth())          # maximum depth reached
    print(tree.get_n_leaves())       # number of leaf nodes
    print(tree.feature_importances_) # impurity-based importances
    print(tree.tree_)                # the raw sklearn Tree object

    # Leaf indices for each sample
    leaf_idx = tree.apply(X)

    # Indicator matrix: which nodes does each sample pass through?
    path = tree.decision_path(X)


Cost-Sensitive Random Forest (CSForestClassifier)
==================================================

:class:`~empulse.models.CSForestClassifier` builds an ensemble of
:class:`~empulse.models.CSTreeClassifier` trees using bootstrap sampling and
random feature subsets, identical to scikit-learn's
:class:`~sklearn.ensemble.RandomForestClassifier` except each tree is grown
with a cost-sensitive splitting criterion [1]_.

Number of estimators
--------------------

.. code-block:: python

    from empulse.models import CSForestClassifier

    # Larger forests are more stable but slower to train
    forest = CSForestClassifier(n_estimators=200, fp_cost=5, fn_cost=1)

Combining predictions
---------------------

The ``combination`` parameter controls how individual tree predictions are
aggregated into a single ensemble prediction:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - ``combination``
     - Description
   * - ``"majority_voting"`` *(default)*
     - Each tree casts one vote; the majority class wins.
   * - ``"weighted_voting"``
     - Trees are weighted by their out-of-bag (OOB) score; requires ``oob_score=True``.

.. code-block:: python

    forest = CSForestClassifier(
        n_estimators=100,
        fp_cost=5,
        fn_cost=1,
        combination="weighted_voting",
        oob_score=True,
    )
    forest.fit(X, y)
    print(forest.oob_score_)

Parallelism and memory
----------------------

Like :class:`~sklearn.ensemble.RandomForestClassifier`, fitting and prediction
can be parallelised across CPU cores with ``n_jobs``:

.. code-block:: python

    forest = CSForestClassifier(
        n_estimators=500,
        fp_cost=5,
        fn_cost=1,
        n_jobs=-1,  # use all available cores
    )

Warm-start incremental training
--------------------------------

Set ``warm_start=True`` to add more trees to an already-fitted forest without
starting from scratch:

.. code-block:: python

    forest = CSForestClassifier(n_estimators=50, fp_cost=5, fn_cost=1, warm_start=True)
    forest.fit(X, y)

    forest.n_estimators = 100    # grow the forest to 100 trees
    forest.fit(X, y)

Feature importances
-------------------

.. code-block:: python

    forest.fit(X, y)
    importances = forest.feature_importances_   # shape (n_features,)

    # For a more reliable estimate use permutation importances
    from sklearn.inspection import permutation_importance
    result = permutation_importance(forest, X, y, n_repeats=10, random_state=0)


Cost-Sensitive Bagging (CSBaggingClassifier)
=============================================

:class:`~empulse.models.CSBaggingClassifier` is the most flexible of the
ensemble models.  It is an ensemble meta-estimator that fits copies of a base
classifier on random subsets of the dataset [1]_.
By default the base estimator is :class:`~empulse.models.CSTreeClassifier`,
but any compatible classifier can be used.

Sampling strategies
-------------------

The four classical bagging variants are all available through combinations of
``bootstrap``, ``bootstrap_features``, ``max_samples``, and ``max_features``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variant
     - Parameter settings
   * - Pasting
     - ``bootstrap=False``, ``bootstrap_features=False``
   * - Bagging *(default)*
     - ``bootstrap=True``, ``bootstrap_features=False``
   * - Random Subspaces
     - ``bootstrap=False``, ``bootstrap_features=True``, ``max_features < 1.0``
   * - Random Patches
     - ``bootstrap=True``, ``bootstrap_features=True``, ``max_features < 1.0``

.. code-block:: python

    from empulse.models import CSBaggingClassifier

    # Standard bagging (default)
    bagging = CSBaggingClassifier(n_estimators=20, fp_cost=5, fn_cost=1)

    # Random Patches: subsample both samples and features
    patches = CSBaggingClassifier(
        n_estimators=50,
        fp_cost=5,
        fn_cost=1,
        max_samples=0.8,
        max_features=0.7,
        bootstrap=True,
        bootstrap_features=True,
    )

Custom base estimator
---------------------

Any classifier that accepts cost arrays in its ``fit`` method can be used:

.. code-block:: python

    from empulse.models import CSBaggingClassifier, CSTreeClassifier

    # Shallow cost-sensitive base trees
    base = CSTreeClassifier(max_depth=3)
    bagging = CSBaggingClassifier(
        estimator=base,
        n_estimators=50,
        fp_cost=5,
        fn_cost=1,
    )
    bagging.fit(X, y)

Out-of-bag evaluation
---------------------

.. code-block:: python

    bagging = CSBaggingClassifier(
        n_estimators=20,
        fp_cost=5,
        fn_cost=1,
        oob_score=True,
    )
    bagging.fit(X, y)
    print(bagging.oob_score_)

Inspecting sub-estimators
--------------------------

.. code-block:: python

    bagging.fit(X, y)

    # List of fitted base estimators
    print(len(bagging.estimators_))

    # Bootstrap sample indices used for each estimator
    print(bagging.estimators_samples_[0])

    # Feature subset used for each estimator
    print(bagging.estimators_features_[0])


Profit-Driven Evolutionary Tree (ProfTreeClassifier)
====================================================

:class:`~empulse.models.ProfTreeClassifier` takes a fundamentally different
approach: instead of growing a tree greedily by splitting nodes, it uses a
**genetic algorithm** to evolve a population of complete trees over many
generations.  At each generation, trees are selected, crossed over, and mutated;
their fitness is measured by a profit metric (default:
:class:`~empulse.metrics.MaxProfit`).
Because the search is gradient-free, it can optimise non-smooth and non-convex
objectives that are intractable for gradient-based methods [1]_.

Basic usage
-----------

.. code-block:: python

    from empulse.models import ProfTreeClassifier
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=500, random_state=0)

    proftree = ProfTreeClassifier(
        tp_cost=300,
        fp_cost=10,
        max_depth=5,
        max_iter=500,
        random_state=42,
    )
    proftree.fit(X, y)

Controlling the genetic algorithm
----------------------------------

The GA is configured through five complementary **variation operators**; their
probabilities must sum to exactly 1.0:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``crossover_rate``
     - Probability of crossing sub-trees between two parent trees.
   * - ``grow_rate``
     - Probability of attaching a new random split to a leaf.
   * - ``prune_rate``
     - Probability of removing a split (collapse an internal node to a leaf).
   * - ``mutate_split_rate``
     - Probability of replacing a split's feature *and* threshold.
   * - ``mutate_value_rate``
     - Probability of replacing only a split's threshold.

.. code-block:: python

    proftree = ProfTreeClassifier(
        tp_cost=300,
        fp_cost=10,
        crossover_rate=0.3,
        grow_rate=0.2,
        prune_rate=0.2,
        mutate_split_rate=0.15,
        mutate_value_rate=0.15,    # must sum to 1.0
        population_size=200,
        max_iter=1_000,
    )

Tree size constraints
---------------------

Computation time scales exponentially with depth, so it is important to
constrain the tree size appropriately:

.. code-block:: python

    proftree = ProfTreeClassifier(
        tp_cost=300,
        fp_cost=10,
        max_depth=6,            # default 10; be careful above 8
        min_samples_split=30,   # default 20
        min_samples_leaf=10,    # default 7
    )

Early stopping
--------------

The GA stops early if no improvement greater than ``tolerance`` is observed for
``patience`` consecutive generations:

.. code-block:: python

    proftree = ProfTreeClassifier(
        tp_cost=300,
        fp_cost=10,
        patience=200,         # wait 200 generations without improvement
        tolerance=1e-5,       # minimum relative improvement to count
        max_iter=2_000,
    )

Complexity regularisation
--------------------------

Set ``alpha > 0`` to penalise trees with many nodes, which can help reduce
overfitting on small datasets:

.. code-block:: python

    proftree = ProfTreeClassifier(
        tp_cost=300,
        fp_cost=10,
        alpha=0.01,
    )

Parallelising the GA
---------------------

Set ``n_jobs`` to the number of CPU cores to use when evaluating the population
in parallel:

.. code-block:: python

    proftree = ProfTreeClassifier(
        tp_cost=300,
        fp_cost=10,
        n_jobs=4,
    )

Custom fitness metric
---------------------

Any :class:`~empulse.metrics.Metric` from :mod:`empulse.metrics` can be used
as the fitness function:

.. code-block:: python

    from empulse.metrics import Metric, CostMatrix, Savings
    from empulse.models import ProfTreeClassifier

    savings_metric = Metric(
        cost_matrix=CostMatrix().add_fp_cost('fp').add_fn_cost('fn'),
        strategy=Savings(),
    )

    proftree = ProfTreeClassifier(loss=savings_metric)
    proftree.fit(X, y, fp=5, fn=1)


sklearn Integration
===================

All four models are fully scikit-learn compatible: they can be embedded in
:class:`~sklearn.pipeline.Pipeline`, evaluated with
:func:`~sklearn.model_selection.cross_val_score`, and tuned with
:class:`~sklearn.model_selection.GridSearchCV`.
When instance-dependent costs are used,
:ref:`metadata routing <sklearn:metadata_routing>` must be enabled.

Pipeline with cross-validation
-------------------------------

.. code-block:: python

    import numpy as np
    from sklearn import set_config
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from empulse.models import CSForestClassifier

    set_config(enable_metadata_routing=True)

    X, y = make_classification(n_samples=500, random_state=0)
    fp_cost = np.random.default_rng(0).uniform(1, 10, size=len(y))

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', CSForestClassifier(n_estimators=50, fn_cost=1)
                    .set_fit_request(fp_cost=True)),
    ])

    scores = cross_val_score(pipeline, X, y, params={'fp_cost': fp_cost})

Hyperparameter search
---------------------

.. code-block:: python

    import numpy as np
    from sklearn import set_config
    from sklearn.datasets import make_classification
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from empulse.metrics import expected_cost_loss
    from empulse.models import CSForestClassifier

    set_config(enable_metadata_routing=True)

    X, y = make_classification(n_samples=200, random_state=0)
    fp_cost = np.random.default_rng(0).uniform(1, 10, size=len(y))
    fn_cost = 1.0

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', CSForestClassifier().set_fit_request(fp_cost=True)),
    ])

    scorer = make_scorer(
        expected_cost_loss,
        response_method='predict_proba',
        greater_is_better=False,
        normalize=True,
        fn_cost=fn_cost,
    ).set_score_request(fp_cost=True)

    grid_search = GridSearchCV(
        pipeline,
        param_grid={'model__n_estimators': [50, 100, 200]},
        scoring=scorer,
    )
    grid_search.fit(X, y, fp_cost=fp_cost)
    print(f"Best n_estimators: {grid_search.best_params_['model__n_estimators']}")


Choosing the Right Model
========================

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Situation
     - Recommended model
   * - Interpretability is important
     - :class:`~empulse.models.CSTreeClassifier` — a single, visualisable tree
   * - Best predictive performance
     - :class:`~empulse.models.CSForestClassifier` — the most accurate option in most cases
   * - Non-standard base estimator needed
     - :class:`~empulse.models.CSBaggingClassifier` — fully customisable ensemble
   * - Objective is non-smooth or non-convex
     - :class:`~empulse.models.ProfTreeClassifier` — gradient-free evolutionary search
   * - Small dataset
     - :class:`~empulse.models.ProfTreeClassifier` with a small ``population_size``
       or :class:`~empulse.models.CSTreeClassifier` with ``ccp_alpha`` pruning


References
==========

.. [1] Correa Bahnsen, A., Aouada, D., & Ottersten, B.
       "Example-Dependent Cost-Sensitive Decision Trees."
       *Expert Systems with Applications*, 42(19), 6609–6619, 2015.
       https://doi.org/10.1016/j.eswa.2015.04.042

.. [2] Höppner, S., Stripling, E., Baesens, B., Broucke, S. V., & Verdonck, T. (2017).
       Profit driven decision trees for churn prediction. arXiv preprint arXiv:1712.08101.

