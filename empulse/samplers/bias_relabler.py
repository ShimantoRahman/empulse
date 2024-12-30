import warnings
from typing import Union, Callable, Optional, TypeVar, TYPE_CHECKING

import numpy as np
from imblearn.base import BaseSampler
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.utils import _safe_indexing, ClassifierTags
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.multiclass import type_of_target

from empulse.samplers._strategies import Strategy, StrategyFn

if TYPE_CHECKING:
    X_arrays = [np.ndarray]
    y_arrays = [np.ndarray]
    try:
        import pandas as pd

        X_arrays.append(pd.DataFrame)
        y_arrays.append(pd.Series)
    except ImportError:
        pass
    try:
        import polars as pl

        X_arrays.append(pl.DataFrame)
        y_arrays.append(pl.Series)
    except ImportError:
        pass
    _XT = TypeVar('_XT', *X_arrays, ArrayLike)
    _YT = TypeVar('_YT', *y_arrays, ArrayLike)
else:
    _XT = TypeVar('_XT', bound=ArrayLike)
    _YT = TypeVar('_YT', bound=ArrayLike)


def _independent_pairs(y_true: ArrayLike, sensitive_feature: np.ndarray) -> int:
    """
    Determine the number of promotion and demotion pairs
    so that y would be statistically independent of the sensitive feature.
    """
    sensitive_indices = np.where(sensitive_feature == 0)[0]
    not_sensititive_indices = np.where(sensitive_feature == 1)[0]
    n_sensitive = len(sensitive_indices)
    n_not_sensitive = len(not_sensititive_indices)
    n = n_sensitive + n_not_sensitive

    # no swapping needed if one of the groups is empty
    if n_sensitive == 0 or n_not_sensitive == 0:
        warnings.warn("sensitive_feature only contains one class, no relabeling is performed.", UserWarning)
        return 0

    pos_ratio_sensitive = np.sum(_safe_indexing(y_true, sensitive_indices)) / n_sensitive
    pos_ratio_not_sensitive = np.sum(_safe_indexing(y_true, not_sensititive_indices)) / n_not_sensitive

    discrimination = pos_ratio_not_sensitive - pos_ratio_sensitive

    # number of pairs to swap label
    return abs(round((discrimination * n_sensitive * n_not_sensitive) / n))


class BiasRelabler(BaseSampler):
    """
    Sampler which relabels instances to remove bias against a subgroup

    Parameters
    ----------
    estimator : Estimator instance
        Base estimator which is used to determine the number of promotion and demotion pairs.
    strategy : {'statistical parity', 'demographic parity'} or Callable, default='statistical parity'
        Determines how the group weights are computed.
        Group weights determine how many instances to relabel for each combination of target and sensitive_feature.

        - ``'statistical_parity'`` or ``'demographic parity'``: \
        probability of positive predictions are equal between subgroups of sensitive feature.

        - ``Callable``: function which computes the number of labels swaps based on the target and sensitive feature. \
        Callable accepts two arguments: \
        y_true and sensitive_feature and returns the number of pairs needed to be swapped.
    transform_feature : Optional[Callable[[numpy.ndarray], numpy.ndarray]], default=None
        Function which transforms sensitive feature before resampling the training data.
        The function takes in the sensitive feature in the form of a numpy.ndarray
        and outputs the transformed sensitive feature as a numpy.ndarray.
        This can be useful if you want to transform a continuous variable to a binary variable at fit time.

    Attributes
    ----------
    estimator_ : Estimator instance
        Fitted estimator.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        from empulse.samplers import BiasRelabler
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression

        X, y = make_classification()
        high_clv = np.random.randint(0, 2, y.shape)

        sampler = BiasRelabler(LogisticRegression())
        sampler.fit_resample(X, y, sensitive_feature=high_clv)

    Example with passing high-clv indicator through cross-validation:

    .. code-block:: python

        import numpy as np
        from empulse.samplers import BiasRelabler
        from imblearn.pipeline import Pipeline
        from sklearn import set_config
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        set_config(enable_metadata_routing=True)

        X, y = make_classification()
        high_clv = np.random.randint(0, 2, y.shape)

        pipeline = Pipeline([
            ('sampler', BiasRelabler(
                LogisticRegression()
            ).set_fit_resample_request(sensitive_feature=True)),
            ('model', LogisticRegression())
        ])

        cross_val_score(pipeline, X, y, params={'sensitive_feature': high_clv})

    Example with passing clv through a grid search and dynamically determining high_clv customer based on training data:

    .. code-block:: python

        import numpy as np
        from empulse.samplers import BiasRelabler
        from imblearn.pipeline import Pipeline
        from sklearn import set_config
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV

        set_config(enable_metadata_routing=True)

        X, y = make_classification()
        clv = np.random.rand(y.size)

        def to_high_clv(clv: np.ndarray) -> np.ndarray:
            return (clv > np.median(clv)).astype(np.int8)

        pipeline = Pipeline([
            ('sampler', BiasRelabler(
                LogisticRegression(),
                transform_feature=to_high_clv
            ).set_fit_resample_request(sensitive_feature=True)),
            ('model', LogisticRegression())
        ])
        param_grid = {'model__C': np.logspace(-5, 2, 10)}

        grid_search = GridSearchCV(pipeline, param_grid=param_grid)
        grid_search.fit(X, y, sensitive_feature=clv)

    References
    ----------

    .. [1] Rahman, S., Janssens, B., & Bogaert, M. (2025).
           Profit-driven pre-processing in B2B customer churn modeling using fairness techniques.
           Journal of Business Research, 189, 115159. doi:10.1016/j.jbusres.2024.115159
    """
    _estimator_type = "sampler"
    _sampling_type = 'bypass'
    _parameter_constraints = {
        'strategy': [StrOptions({'statistical parity', 'demographic parity'}), callable],
        'transform_attr': [callable, None],
        'random_state': ['random_state'],
    }
    _strategy_mapping: dict[str, StrategyFn] = {
        'statistical parity': _independent_pairs,
        'demographic parity': _independent_pairs,
    }

    if TYPE_CHECKING:
        # BaseEstimator should dynamically generate the method signature at runtime
        def set_fit_resample_request(self, sensitive_feature=False): pass

    def __init__(
            self,
            estimator,
            *,
            strategy: Union[Callable, Strategy] = 'statistical parity',
            transform_feature: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        super().__init__()
        self.estimator = estimator
        self.transform_feature = transform_feature
        self.strategy = strategy

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags = ClassifierTags(multi_class=False)
        return tags

    def fit_relabel(self, X: _XT, y: _YT, *, sensitive_feature: Optional[ArrayLike] = None) -> tuple[_XT, _YT]:
        """
        Fit the estimator and relabel the data according to the strategy.

        Calls the fit_resample method to be compatible with the imbalance-learn API.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_features)
        y : 1D array-like, shape=(n_samples,)
        sensitive_feature : 1D array-like, shape=(n_samples,)
            Sensitive feature used to determine the number of promotion and demotion pairs.

        Returns
        -------
        X : 2D array-like, shape=(n_samples, n_features)
            Original training data.
        y : np.ndarray
            Relabeled target values.
        """
        return self.fit_resample(X, y, sensitive_feature=sensitive_feature)

    def fit_resample(
            self,
            X: _XT,
            y: _YT,
            *,
            sensitive_feature: Optional[ArrayLike] = None,
    ) -> tuple[_XT, _YT]:
        """
        Fit the estimator and relabel the data according to the strategy.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_features)
        y : 1D array-like, shape=(n_samples,)
        sensitive_feature : 1D array-like, shape=(n_samples,)
            Sensitive feature used to determine the number of promotion and demotion pairs.

        Returns
        -------
        X : 2D array-like, shape=(n_samples, n_features)
            Original training data.
        y : np.ndarray
            Relabeled target values.
        """
        return super().fit_resample(X, y, sensitive_feature=sensitive_feature)

    def _fit_resample(
            self,
            X: _XT,
            y: _YT,
            *,
            sensitive_feature: Optional[ArrayLike] = None
    ) -> tuple[_XT, _YT]:
        """
        Fit the estimator and relabel the data according to the strategy.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_features)
        y : 1D array-like, shape=(n_samples,)
        sensitive_feature : 1D array-like, shape=(n_samples,)
            Sensitive feature used to determine the number of promotion and demotion pairs.

        Returns
        -------
        X : 2D array-like, shape=(n_samples, n_features)
            Original training data.
        y : np.ndarray
            Relabeled target values.
        """
        sensitive_feature = np.asarray(sensitive_feature)
        y_type = type_of_target(y, input_name='y', raise_unknown=True)
        if y_type != 'binary':
            raise ValueError(
                'Only binary classification is supported. The type of the target '
                f'is {y_type}.'
            )
        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            return X, y
        y_binarized = np.where(y == self.classes_[1], 1, 0)

        if self.transform_feature is not None:
            sensitive_feature = self.transform_feature(sensitive_feature)

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        y_pred = self.estimator_.predict_proba(X)[:, 1]

        if isinstance(self.strategy, str):
            strategy = self._strategy_mapping[self.strategy]
        else:
            strategy = self.strategy

        n_pairs = strategy(y_binarized, sensitive_feature)
        if n_pairs <= 0:
            return X, np.asarray(y)

        sensitive_indices = np.where(sensitive_feature == 0)[0]
        non_sensitive = np.where(sensitive_feature == 1)[0]
        probas_non_sensitive = y_pred[non_sensitive]
        probas_sensitive = y_pred[sensitive_indices]

        demotion_candidates = _get_demotion_candidates(
            probas_non_sensitive,
            _safe_indexing(y, non_sensitive),
            n_pairs
        )
        promotion_candidates = _get_promotion_candidates(
            probas_sensitive,
            _safe_indexing(y, sensitive_indices),
            n_pairs
        )

        # map promotion and demotion candidates to original indices
        indices = np.arange(len(y))
        demotion_candidates = indices[non_sensitive][demotion_candidates]
        promotion_candidates = indices[sensitive_indices][promotion_candidates]

        # relabel the data
        if hasattr(y, 'copy'):
            relabled_y = y.copy()
        elif hasattr(y, 'clone'):
            relabled_y = y.clone()
        else:
            relabled_y = np.copy(y)

        if hasattr(relabled_y, 'loc'):
            relabled_y.loc[demotion_candidates] = 0
            relabled_y.loc[promotion_candidates] = 1
        else:
            relabled_y[demotion_candidates] = 0
            relabled_y[promotion_candidates] = 1

        return X, relabled_y


def _get_demotion_candidates(y_pred, y_true, n_pairs):
    """Returns the n_pairs instances with the lowest probability of being positive class label"""
    positive_indices = np.where(y_true == 1)[0]
    positive_predictions = y_pred[positive_indices]
    return positive_indices[np.argsort(positive_predictions)[:n_pairs]]


def _get_promotion_candidates(y_pred, y_true, n_pairs):
    """Returns the n_pairs instances with the lowest probability of being negative class label"""
    negative_indices = np.where(y_true == 0)[0]
    negative_predictions = y_pred[negative_indices]
    return negative_indices[np.argsort(negative_predictions)[-n_pairs:]]
