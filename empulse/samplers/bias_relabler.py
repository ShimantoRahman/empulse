import sys
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import numpy as np
from imblearn.base import BaseSampler
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from sklearn.utils import _safe_indexing
from sklearn.utils._param_validation import HasMethods, StrOptions

from .._types import FloatNDArray, IntNDArray, ParameterConstraint
from ..utils._sklearn_compat import ClassifierTags, Tags, type_of_target  # type: ignore
from ._strategies import Strategy

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    _XT = TypeVar('_XT', NDArray[Any], pd.DataFrame, ArrayLike)
    _YT = TypeVar('_YT', NDArray[Any], pd.Series, ArrayLike)
else:
    _XT = TypeVar('_XT', NDArray[Any], ArrayLike)
    _YT = TypeVar('_YT', NDArray[Any], ArrayLike)

StrategyFn = Callable[[NDArray[Any], NDArray[Any]], int]


def _independent_pairs(y_true: ArrayLike, sensitive_feature: NDArray[Any]) -> int:
    """Determine promotion and demotion pairs so that y is statistically independent of sensitive feature."""
    sensitive_indices = np.where(sensitive_feature == 0)[0]
    not_sensititive_indices = np.where(sensitive_feature == 1)[0]
    n_sensitive = len(sensitive_indices)
    n_not_sensitive = len(not_sensititive_indices)
    n = n_sensitive + n_not_sensitive

    # no swapping needed if one of the groups is empty
    if n_sensitive == 0 or n_not_sensitive == 0:
        warnings.warn(
            'sensitive_feature only contains one class, no relabeling is performed.',
            UserWarning,
            stacklevel=2,
        )
        return 0

    pos_ratio_sensitive = np.sum(_safe_indexing(y_true, sensitive_indices)) / n_sensitive
    pos_ratio_not_sensitive = np.sum(_safe_indexing(y_true, not_sensititive_indices)) / n_not_sensitive

    discrimination = pos_ratio_not_sensitive - pos_ratio_sensitive

    # number of pairs to swap label
    return int(abs(round((discrimination * n_sensitive * n_not_sensitive) / n)))


class BiasRelabler(BaseSampler):  # type: ignore[misc]
    """
    Sampler which relabels instances to remove bias against a subgroup.

    Read more in the :ref:`User Guide <bias_mitigation>`.

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

    _estimator_type: ClassVar[str] = 'sampler'
    _sampling_type: ClassVar[str] = 'bypass'
    _parameter_constraints: ClassVar[ParameterConstraint] = {
        'estimator': [HasMethods(['fit', 'predict_proba'])],
        'strategy': [StrOptions({'statistical parity', 'demographic parity'}), callable],
        'transform_feature': [callable, None],
    }
    _strategy_mapping: ClassVar[dict[Strategy, StrategyFn]] = {
        'statistical parity': _independent_pairs,
        'demographic parity': _independent_pairs,
    }

    if TYPE_CHECKING:  # pragma: no cover
        # BaseEstimator should dynamically generate the method signature at runtime
        def set_fit_resample_request(self, sensitive_feature: bool = False) -> Self:  # noqa: D102
            pass

    def __init__(
        self,
        estimator: Any,
        *,
        strategy: StrategyFn | Strategy = 'statistical parity',
        transform_feature: Callable[[NDArray[Any]], IntNDArray] | None = None,
    ):
        super().__init__()
        self.estimator = estimator
        self.transform_feature = transform_feature
        self.strategy = strategy

    def _more_tags(self) -> dict[str, bool]:
        return {
            'binary_only': True,
            'poor_score': True,
        }

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.classifier_tags = ClassifierTags(multi_class=False)
        return tags

    def fit_resample(self, X: _XT, y: _YT, *, sensitive_feature: ArrayLike | None = None) -> tuple[_XT, _YT]:
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
        X, y = super().fit_resample(X, y, sensitive_feature=sensitive_feature)
        X: _XT
        y: _YT
        return X, y

    def _fit_resample(
        self, X: NDArray[Any], y: NDArray[Any], *, sensitive_feature: ArrayLike | None = None
    ) -> tuple[NDArray[Any], NDArray[Any]]:
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
            raise ValueError(f'Only binary classification is supported. The type of the target is {y_type}.')
        self.classes_: NDArray[np.int64] = np.unique(y)
        if len(self.classes_) == 1:
            return X, y
        y_binarized = np.where(y == self.classes_[1], 1, 0)

        if self.transform_feature is not None:
            sensitive_feature = self.transform_feature(sensitive_feature)

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        y_pred = self.estimator_.predict_proba(X)[:, 1]

        strategy = self._strategy_mapping[self.strategy] if isinstance(self.strategy, str) else self.strategy
        n_pairs = strategy(y_binarized, sensitive_feature)
        if n_pairs <= 0:
            return X, np.asarray(y)

        sensitive_indices = np.where(sensitive_feature == 0)[0]
        non_sensitive = np.where(sensitive_feature == 1)[0]
        probas_non_sensitive = y_pred[non_sensitive]
        probas_sensitive = y_pred[sensitive_indices]

        demotion_candidates = _get_demotion_candidates(probas_non_sensitive, _safe_indexing(y, non_sensitive), n_pairs)
        promotion_candidates = _get_promotion_candidates(
            probas_sensitive, _safe_indexing(y, sensitive_indices), n_pairs
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


def _get_demotion_candidates(y_pred: FloatNDArray, y_true: FloatNDArray, n_pairs: int) -> FloatNDArray:
    """Return the n_pairs instances with the lowest probability of being positive class label."""
    positive_indices = np.where(y_true == 1)[0]
    positive_predictions = y_pred[positive_indices]
    demotion_candidates: FloatNDArray = positive_indices[np.argsort(positive_predictions)[:n_pairs]]
    return demotion_candidates


def _get_promotion_candidates(y_pred: FloatNDArray, y_true: FloatNDArray, n_pairs: int) -> FloatNDArray:
    """Return the n_pairs instances with the lowest probability of being negative class label."""
    negative_indices = np.where(y_true == 0)[0]
    negative_predictions = y_pred[negative_indices]
    promotion_candidates: FloatNDArray = negative_indices[np.argsort(negative_predictions)[-n_pairs:]]
    return promotion_candidates
