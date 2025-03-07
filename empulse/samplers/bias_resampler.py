import sys
import warnings
from collections.abc import Callable
from itertools import product
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import numpy as np
from imblearn.base import BaseSampler
from numpy.random import RandomState
from numpy.typing import ArrayLike, NDArray
from sklearn.utils import _safe_indexing, check_random_state
from sklearn.utils._param_validation import StrOptions

from .._types import IntNDArray, ParameterConstraint
from ..utils._sklearn_compat import ClassifierTags, Tags, type_of_target  # type: ignore
from ._strategies import Strategy, StrategyFn, _independent_weights

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    _XT = TypeVar('_XT', NDArray[Any], pd.DataFrame, ArrayLike)
    _YT = TypeVar('_YT', NDArray[Any], pd.Series, ArrayLike)
else:
    _XT = TypeVar('_XT', bound=ArrayLike)
    _YT = TypeVar('_YT', bound=ArrayLike)


class BiasResampler(BaseSampler):  # type: ignore[misc]
    """
    Sampler which resamples instances to remove bias against a subgroup.

    Read more in the :ref:`User Guide <bias_mitigation>`.

    Parameters
    ----------
    strategy : {'statistical parity', 'demographic parity'} or Callable, default='statistical parity'
        Determines how the group weights are computed.
        Group weights determine how much to over or undersample each combination of target and sensitive feature.
        For example, a weight of 2 for the pair (y_true == 1, sensitive_feature == 0) means that the resampled dataset
        should have twice as many instances with y_true == 1 and sensitive_feature == 0
        compared to the original dataset.

        - ``'statistical_parity'`` or ``'demographic parity'``: \
        probability of positive predictions are equal between subgroups of sensitive feature.

        - ``Callable``: function which computes the group weights based on the target and sensitive feature. \
        Callable accepts two arguments: y_true and sensitive_feature and returns the group weights. \
        Group weights are a 2x2 matrix where the rows represent the target variable and the columns represent the \
        sensitive feature. \
        The element at position (i, j) is the weight for the pair (y_true == i, sensitive_feature == j).
    transform_feature : Optional[Callable], default=None
        Function which transforms sensitive_feature before resampling the training data.
        The function takes in the sensitive feature in the form of a numpy.ndarray
        and outputs the transformed sensitive feature as a numpy.ndarray.
        This can be useful if you want to transform a continuous variable to a binary variable at fit time.
    random_state : int or :class:`numpy:numpy.random.RandomState`, optional
        Random number generator seed for reproducibility.

    Attributes
    ----------
    sample_indices_ : numpy.ndarray
        Indices of the samples that were selected.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        from empulse.samplers import BiasResampler
        from sklearn.datasets import make_classification

        X, y = make_classification()
        high_clv = np.random.randint(0, 2, y.shape)

        sampler = BiasResampler()
        sampler.fit_resample(X, y, sensitive_feature=high_clv)

    Example with passing high-clv indicator through cross-validation:

    .. code-block:: python

        import numpy as np
        from empulse.samplers import BiasResampler
        from imblearn.pipeline import Pipeline
        from sklearn import set_config
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        set_config(enable_metadata_routing=True)

        X, y = make_classification()
        high_clv = np.random.randint(0, 2, y.shape)

        pipeline = Pipeline([
            ('sampler', BiasResampler().set_fit_resample_request(sensitive_feature=True)),
            ('model', LogisticRegression())
        ])

        cross_val_score(pipeline, X, y, params={'sensitive_feature': high_clv})

    Example with passing clv through a grid search and dynamically determining high_clv customer based on training data:

    .. code-block:: python

        import numpy as np
        from empulse.samplers import BiasResampler
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
            ('sampler', BiasResampler(
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
        'strategy': [StrOptions({'statistical parity', 'demographic parity'}), callable],
        'transform_feature': [callable, None],
        'random_state': ['random_state'],
    }
    _strategy_mapping: ClassVar[dict[Strategy, StrategyFn]] = {
        'statistical parity': _independent_weights,
        'demographic parity': _independent_weights,
    }

    if TYPE_CHECKING:  # pragma: no cover
        # BaseEstimator should dynamically generate the method signature at runtime
        def set_fit_resample_request(self, sensitive_feature: bool = False) -> Self:  # noqa: D102
            pass

    def __init__(
        self,
        *,
        strategy: StrategyFn | Strategy = 'statistical parity',
        transform_feature: Callable[[NDArray[Any]], IntNDArray] | None = None,
        random_state: RandomState | int | None = None,
    ):
        super().__init__()
        self.strategy = strategy
        self.transform_feature = transform_feature
        self.random_state = random_state

    def _more_tags(self) -> dict[str, bool]:
        return {
            'binary_only': True,
            'poor_score': True,
        }

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.classifier_tags = ClassifierTags(multi_class=False)
        tags.sampler_tags.sample_indices = True
        return tags

    def fit_resample(
        self,
        X: _XT,
        y: _YT,
        *,
        sensitive_feature: ArrayLike | None = None,
    ) -> tuple[_XT, _YT]:
        """
        Resample the data according to the strategy.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_features)
        y : 1D array-like, shape=(n_samples,)
        sensitive_feature : 1D array-like, shape=(n_samples,)
            Sensitive attribute used to determine which instances to resample.

        Returns
        -------
        X : 2D array-like, shape=(n_samples, n_features)
            Resampled training data.
        y : 1D array-like, shape=(n_samples,)
            Resampled target values.
        """
        X, y = super().fit_resample(X, y, sensitive_feature=sensitive_feature)
        X: _XT
        y: _YT
        return X, y

    def _fit_resample(
        self,
        X: NDArray[Any],
        y: IntNDArray,
        *,
        sensitive_feature: ArrayLike | None = None,
    ) -> tuple[NDArray[Any], IntNDArray]:
        """
        Resample the data according to the strategy.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_features)
        y : 1D array-like, shape=(n_samples,)
        sensitive_feature : 1D array-like, shape=(n_samples,)
            Sensitive attribute used to determine which instances to resample.

        Returns
        -------
        X : 2D array-like, shape=(n_samples, n_features)
            Resampled training data.
        y : 1D array-like, shape=(n_samples,)
            Resampled target values.
        """
        y_type = type_of_target(y, input_name='y', raise_unknown=True)
        if y_type != 'binary':
            raise ValueError(f'Only binary classification is supported. The type of the target is {y_type}.')
        self.classes_: NDArray[np.int64] = np.unique(y)
        if len(self.classes_) == 1:
            self.sample_indices_ = np.arange(len(y))
            return X, y
        y_binarized = np.where(y == self.classes_[1], 1, 0)

        random_state = check_random_state(self.random_state)
        if sensitive_feature is None:
            self.sample_indices_ = np.arange(len(y))
            return X, y
        sensitive_feature = np.asarray(sensitive_feature)

        if self.transform_feature is not None:
            sensitive_feature = self.transform_feature(sensitive_feature)

        strategy = self._strategy_mapping[self.strategy] if isinstance(self.strategy, str) else self.strategy
        class_weights = strategy(y_binarized, sensitive_feature)
        # if class_weights are all 1, no resampling is needed
        if np.allclose(class_weights, np.ones(class_weights.shape)):
            self.sample_indices_ = np.arange(len(y))
            return X, y
        indices = np.empty((0,), dtype=int)

        unique_attr = np.unique(sensitive_feature)
        if len(unique_attr) == 1:
            warnings.warn(
                'sensitive_feature only contains one class, no resampling is performed.',
                UserWarning,
                stacklevel=2,
            )
            self.sample_indices_ = np.arange(len(y))
            return X, y

        # determine the number of samples to be drawn for each class and sensitive_feature value
        for target_class, sensitive_val in product(np.unique(y_binarized), unique_attr):
            sensitive_val = int(sensitive_val)
            idx_class = np.flatnonzero(y_binarized == target_class)
            idx_sensitive_feature = np.flatnonzero(sensitive_feature == sensitive_val)
            idx_class_sensitive = np.intersect1d(idx_class, idx_sensitive_feature)
            n_samples = int(class_weights[target_class, sensitive_val] * len(idx_class_sensitive))
            if n_samples > len(idx_class_sensitive):  # oversampling
                indices = np.concatenate((indices, idx_class_sensitive))
                indices = np.concatenate((
                    indices,
                    random_state.choice(idx_class_sensitive, n_samples - len(idx_class_sensitive), replace=True),
                ))
            else:  # undersampling
                indices = np.concatenate((indices, random_state.choice(idx_class_sensitive, n_samples, replace=False)))

        self.sample_indices_ = indices

        return _safe_indexing(X, indices), _safe_indexing(y, indices)
