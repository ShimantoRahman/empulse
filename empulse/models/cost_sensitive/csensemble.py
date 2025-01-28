from typing import Literal

from numpy.typing import ArrayLike

from .csbagging import BaggingClassifier
from .cstree import CSTreeClassifier

Combination = Literal[
    "majority_voting", "weighted_voting", "stacking", "stacking_proba"
]
MaxFeatures = Literal["auto", "sqrt", "log2"]


class CSForestClassifier(BaggingClassifier):
    """
    Random Forest classifier to optimize instance-dependent cost loss.

    Parameters
    ----------

    n_estimators : int, default=100
        The number of trees in the forest.

    combination : string, optional default="majority_voting"
        Which combination method to use:
          - If "majority_voting" then combine by majority voting
          - If "weighted_voting" then combine by weighted voting using the
            out of bag savings as the weight for each estimator.
          - If "stacking" then a Cost Sensitive Logistic Regression is used
            to learn the combination.
          - If "stacking_proba" then a Cost Sensitive Logistic Regression trained
            with the estimated probabilities is used to learn the combination.

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
          - If "auto", then `max_features=sqrt(n_features)`.
          - If "sqrt", then `max_features=sqrt(n_features)`.
          - If "log2", then `max_features=log2(n_features)`.
          - If None, then `max_features=n_features`.

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

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    verbose : int, optional (default=0)
        Controls the verbosity of the building process.

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

    See also
    --------
    costcla.models.CostSensitiveDecisionTreeClassifier

    References
    ----------

    .. [1] Correa Bahnsen, A., Aouada, D., & Ottersten, B.
           `"Ensemble of Example-Dependent Cost-Sensitive Decision Trees" <http://arxiv.org/abs/1505.04637>`__,
           2015, http://arxiv.org/abs/1505.04637.
    """

    def __init__(
        self,
        n_estimators=100,
        *,
        tp_cost: ArrayLike | float = 0.0,
        tn_cost: ArrayLike | float = 0.0,
        fn_cost: ArrayLike | float = 0.0,
        fp_cost: ArrayLike | float = 0.0,
        combination: Combination = "majority_voting",
        max_features: MaxFeatures | int | float = "auto",
        max_samples=1.0,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=1,
        verbose=False,
        pruned=False,
        random_state=None,
    ):
        super().__init__(
            estimator=CSTreeClassifier(max_features=1.0),
            n_estimators=n_estimators,
            estimator_params=(
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "pruned",
            ),
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            tp_cost=tp_cost,
            tn_cost=tn_cost,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=True,
            bootstrap_features=False,
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

    estimator : ...

    final_estimator : ...

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default=0.5)
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.

    combination : string, optional (default="majority_voting")
        Which combination method to use:
          - If "majority_voting" then combine by majority voting
          - If "weighted_voting" then combine by weighted voting using the
            out of bag savings as the weight for each estimator.
          - If "stacking" then a Cost Sensitive Logistic Regression is used
            to learn the combination.
          - If "stacking_proba" then a Cost Sensitive Logistic Regression trained
            with the estimated probabilities is used to learn the combination.

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    verbose : int, optional (default=0)
        Controls the verbosity of the building process.

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

    def __init__(
        self,
        estimator=None,
        final_estimator=None,
        n_estimators=10,
        max_samples=0.5,
        combination="majority_voting",
        n_jobs=1,
        verbose=False,
    ):
        super().__init__(
            estimator=estimator,
            final_estimator=final_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=1.0,
            bootstrap=True,
            bootstrap_features=False,
            combination=combination,
            n_jobs=n_jobs,
            random_state=None,
            verbose=verbose,
        )
