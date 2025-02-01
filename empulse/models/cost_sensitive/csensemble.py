from numbers import Integral
from typing import ClassVar, Literal

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import HasMethods, Interval, RealNotInt, StrOptions

from .csbagging import BaggingClassifier
from .cstree import CSTreeClassifier


class CSForestClassifier(BaggingClassifier):
    """
    Random Forest classifier to optimize instance-dependent cost loss.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

    tp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true positives. If ``float``, then all true positives have the same cost.
        If array-like, then it is the cost of each true positive classification.
        Is overwritten if another `tp_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    fp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false positives. If ``float``, then all false positives have the same cost.
        If array-like, then it is the cost of each false positive classification.
        Is overwritten if another `fp_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    tn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true negatives. If ``float``, then all true negatives have the same cost.
        If array-like, then it is the cost of each true negative classification.
        Is overwritten if another `tn_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    fn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false negatives. If ``float``, then all false negatives have the same cost.
        If array-like, then it is the cost of each false negative classification.
        Is overwritten if another `fn_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    combination : string, optional default="majority_voting"
        Which combination method to use:
          - If ``"majority_voting"`` then combine by majority voting
          - If ``"weighted_voting"`` then combine by weighted voting using the
            out of bag savings as the weight for each estimator.

    max_depth : int, default=None
        The maximum depth of the tree.
        If None, then nodes are expanded until all leaves are pure or
        until all leaves contain less than ``min_samples_split`` samples.

    max_features : {'auto', 'sqrt', 'log2', None}, int or float, default='auto'
        The number of features to consider when looking for the best split in each tree:
          - If int, then consider `max_features` features at each split.
          - If float, then `max_features` is a percentage and
            `int(max_features * n_features)` features are considered at each
            split.
          - If ``"auto"``, then ``max_features=sqrt(n_features)``.
          - If ``"sqrt"``, then ``max_features=sqrt(n_features)``.
          - If ``"log2"``, then ``max_features=log2(n_features)``.
          - If None, then ``max_features=n_features``.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_samples : int or float, default=1.0
        The number of samples to draw from X to train each base estimator.
          - If None (default), then draw ``X.shape[0]`` samples.
          - If int, then draw max_samples samples.
          - If float, then draw ``max(round(n_samples * max_samples), 1)`` samples.
            Thus, ``max_samples`` should be in the interval ``(0.0, 1.0]``.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:
          - If int, then consider ``min_samples_split`` as the minimum number.
          - If float, then ``min_samples_split`` is a fraction and ``ceil(min_samples_split * n_samples)``
          are the minimum number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at least ``min_samples_leaf`` training samples
        in each of the left and right branches.
        This may have the effect of smoothing the model, especially in regression.
          - If int, then consider ``min_samples_leaf`` as the minimum number.
          - If float, then ``min_samples_leaf`` is a fraction and ``ceil(min_samples_leaf * n_samples)``
            are the minimum number of samples for each node.

    pruned : bool, optional (default=True)
        Whenever or not to prune the decision tree using cost-based pruning

    bootstrap: bool, default=True
        Whether samples are drawn with replacement.
        If False, sampling without replacement is performed.

    bootstrap_features: bool, default=False
        Whether features are drawn with replacement.

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    verbose : int, optional (default=0)
        Controls the verbosity of the building process.

    random_state : int, RandomState instance or None, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    `base_estimator_`: list of estimators
        The base estimator from which the ensemble is grown.

    `estimators_`: list of estimators
        The collection of fitted base estimators.

    `estimators_samples_`: list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    `estimators_features_`: list of arrays
        The subset of drawn features for each base estimator.

    See Also
    --------
    costcla.models.CostSensitiveDecisionTreeClassifier

    References
    ----------

    .. [1] Correa Bahnsen, A., Aouada, D., & Ottersten, B.
           `"Ensemble of Example-Dependent Cost-Sensitive Decision Trees" <http://arxiv.org/abs/1505.04637>`__,
           2015, http://arxiv.org/abs/1505.04637.
    """

    _parameter_constraints: ClassVar[dict[str, list]] = {
        'combination': [StrOptions({'majority_voting', 'weighted_voting'})],
        'max_depth': [Interval(Integral, 1, None, closed='left'), None],
        'min_samples_split': [
            Interval(Integral, 2, None, closed='left'),
            Interval(RealNotInt, 0.0, 1.0, closed='right'),
        ],
        'min_samples_leaf': [
            Interval(Integral, 1, None, closed='left'),
            Interval(RealNotInt, 0.0, 1.0, closed='neither'),
        ],
        'pruned': ['boolean'],
        **BaggingClassifier._parameter_constraints,
    }

    def __init__(
        self,
        n_estimators=100,
        *,
        tp_cost: ArrayLike | float = 0.0,
        tn_cost: ArrayLike | float = 0.0,
        fn_cost: ArrayLike | float = 0.0,
        fp_cost: ArrayLike | float = 0.0,
        combination: Literal['majority_voting', 'weighted_voting'] = 'majority_voting',
        max_features: Literal['auto', 'sqrt', 'log2'] | int | float = 'auto',
        max_samples: int | float = 1.0,
        max_depth: int | None = None,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        n_jobs: int = 1,
        verbose: bool | int = False,
        pruned: bool = False,
        random_state: int | np.random.RandomState | None = None,
    ):
        super().__init__(
            estimator=CSTreeClassifier(max_features=1.0),
            n_estimators=n_estimators,
            estimator_params=(
                'max_depth',
                'min_samples_split',
                'min_samples_leaf',
                'pruned',
            ),
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            tp_cost=tp_cost,
            tn_cost=tn_cost,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            combination=combination,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.pruned = pruned


class CSBaggingClassifier(BaggingClassifier):
    """
    Bagging classifier to optimize instance-dependent cost loss.

    Parameters
    ----------
    estimator : estimator, default=None
        The base estimator to fit on random subsets of the dataset.

    final_estimator : estimator, default=None
        The estimator to train on the weighted combination of the base estimators.
        By default, a Cost Sensitive Logistic Regression is used.
        Only used if ``combination`` is ``"stacking"`` or ``"stacking_proba"``.

    n_estimators : int, default=10
        The number of base estimators in the ensemble.

    tp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true positives. If ``float``, then all true positives have the same cost.
        If array-like, then it is the cost of each true positive classification.
        Is overwritten if another `tp_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    fp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false positives. If ``float``, then all false positives have the same cost.
        If array-like, then it is the cost of each false positive classification.
        Is overwritten if another `fp_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    tn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true negatives. If ``float``, then all true negatives have the same cost.
        If array-like, then it is the cost of each true negative classification.
        Is overwritten if another `tn_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    fn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false negatives. If ``float``, then all false negatives have the same cost.
        If array-like, then it is the cost of each false negative classification.
        Is overwritten if another `fn_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    max_samples : int or float, default=1.0
        The number of samples to draw from X to train each base estimator.
          - If None (default), then draw ``X.shape[0]`` samples.
          - If int, then draw max_samples samples.
          - If float, then draw ``max(round(n_samples * max_samples), 1)`` samples.
            Thus, ``max_samples`` should be in the interval ``(0.0, 1.0]``.

    combination : string, default="majority_voting"
        Which combination method to use:
          - If ``"majority_voting"`` then combine by majority voting
          - If ``"weighted_voting"`` then combine by weighted voting using the
            out of bag savings as the weight for each estimator.
          - If ``"stacking"`` then the ``final_estimator`` is used to learn the combination.
          - If ``"stacking_proba"`` then the ``final_estimator`` is trained
            with the estimated probabilities is used to learn the combination.

    bootstrap: bool, default=True
        Whether samples are drawn with replacement.
        If False, sampling without replacement is performed.

    bootstrap_features: bool, default=False
        Whether features are drawn with replacement.

    n_jobs : int, default=1
        The number of jobs to run in parallel for both ``fit`` and ``predict``.
        If -1, then the number of jobs is set to the number of cores.

    verbose : int, default=0
        Controls the verbosity of the building process.

    random_state : int, RandomState instance or None, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.


    Attributes
    ----------
    `base_estimator_`: list of estimators
        The base estimator from which the ensemble is grown.

    `estimators_`: list of estimators
        The collection of fitted base estimators.

    `estimators_samples_`: list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    `estimators_features_`: list of arrays
        The subset of drawn features for each base estimator.

    References
    ----------

    .. [1] Correa Bahnsen, A., Aouada, D., & Ottersten, B.
           `"Ensemble of Example-Dependent Cost-Sensitive Decision Trees" <http://arxiv.org/abs/1505.04637>`__,
           2015, http://arxiv.org/abs/1505.04637.
    """

    _parameter_constraints: ClassVar[dict[str, list]] = {
        'estimator': [HasMethods(['fit', 'predict']), HasMethods(['fit', 'predict_proba']), None],
        'final_estimator': [HasMethods(['fit', 'predict']), HasMethods(['fit', 'predict_proba']), None],
        'combination': [StrOptions({'majority_voting', 'weighted_voting', 'stacking', 'stacking_proba'})],
        **BaggingClassifier._parameter_constraints,
    }

    def __init__(
        self,
        estimator=None,
        *,
        final_estimator=None,
        n_estimators: int = 10,
        tp_cost: ArrayLike | float = 0.0,
        tn_cost: ArrayLike | float = 0.0,
        fn_cost: ArrayLike | float = 0.0,
        fp_cost: ArrayLike | float = 0.0,
        max_samples: float | int = 0.5,
        max_features: Literal['auto', 'sqrt', 'log2'] | int | float = 'auto',
        combination: Literal['majority_voting', 'weighted_voting', 'stacking', 'stacking_proba'] = 'majority_voting',
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        n_jobs: int = 1,
        verbose: bool | int = False,
        random_state: int | np.random.RandomState | None = None,
    ):
        super().__init__(
            estimator=estimator,
            final_estimator=final_estimator,
            n_estimators=n_estimators,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            tp_cost=tp_cost,
            tn_cost=tn_cost,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            combination=combination,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
