import warnings
from typing import Union, Callable, Optional, TypeVar, TYPE_CHECKING

import numpy as np
from imblearn.base import BaseSampler
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.utils import _safe_indexing, ClassifierTags
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import validate_data

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

        - ``Callable``: function which computes the group weights based on the target and sensitive feature. \
        Callable accepts two arguments: y_true and sensitive_feature and returns the group weights. \
        Group weights are a 2x2 matrix where the rows represent the target variable and the columns represent the \
        sensitive feature. \
        The element at position (i, j) is the weight for the pair (y_true == i, sensitive_feature == j).
    transform_attr : Optional[Callable], default=None
        Function which transforms sensitive feature before resampling the training data.


    Attributes
    ----------
    estimator_ : Estimator instance
        Fitted estimator.
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
            transform_attr: Optional[Callable] = None,
    ):
        super().__init__()
        self.estimator = estimator
        self.transform_attr = transform_attr
        self.strategy = strategy

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags = ClassifierTags(multi_class=False)
        return tags


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
        X, y = validate_data(self, X, y)
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

        if self.transform_attr is not None:
            sensitive_feature = self.transform_attr(sensitive_feature)

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
