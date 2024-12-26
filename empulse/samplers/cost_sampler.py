from typing import Literal, TYPE_CHECKING

import numpy as np
from imblearn.base import BaseSampler
from numpy.typing import ArrayLike, NDArray
from sklearn.utils import check_random_state, ClassifierTags
from sklearn.utils._param_validation import Interval, Real, StrOptions


class CostSensitiveSampler(BaseSampler):
    """
    Cost-proportionate sampling.

    This method adjusts the sampling probability of each sample based on the cost of misclassification.
    This is done either by rejection sampling [1]_ or oversampling [2]_.

    Parameters
    ----------
        method : {'rejection sampling', 'oversampling'}, default='rejection sampling'
            Method to perform the cost-proportionate sampling,
            either 'RejectionSampling' or 'OverSampling'.

        oversampling_norm: float, default=0.1
            Oversampling norm for the cost.

        percentile_threshold: float, default=97.5
            Outlier adjustment for the cost.
            Costs are normalized and cost values above the percentile_threshold'th percentile are set to 1.

        random_state : int or :class:`numpy:numpy.random.RandomState`, optional
            Random number generator seed for reproducibility.

    Attributes
    ----------
    sample_indices_ : ndarray
        Indices of the samples that were selected.

    References
    ----------
    .. [1] B. Zadrozny, J. Langford, N. Naoki, "Cost-sensitive learning by
           cost-proportionate example weighting", in Proceedings of the
           Third IEEE International Conference on Data Mining, 435-442, 2003.

    .. [2] C. Elkan, "The foundations of Cost-Sensitive Learning",
           in Seventeenth International Joint Conference on Artificial Intelligence,
           973-978, 2001.

    Notes
    -----
    code modified from `costcla.sampling.cost_sampling`.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        from sklearn.datasets import make_classification

        X, y = make_classification()
        fp_cost = np.ones_like(y) * 10
        fn_cost = np.ones_like(y)

        sampler = CostSensitiveSampler(method='oversampling', random_state=42)
        X_re, y_re = sampler.fit_resample(X, y, fp_cost=fp_cost, fn_cost=fn_cost)

    """

    _sampling_type = 'bypass'
    _parameter_constraints = {
        'method': [StrOptions({'oversampling', 'rejection sampling'})],
        'oversampling_norm': [Interval(Real, 0, 1, closed='both')],
        'percentile_threshold': [Interval(Real, 0, 100, closed='both')],
        'random_state': ['random_state'],
    }

    if TYPE_CHECKING:
        # BaseEstimator should dynamically generate the method signature at runtime
        def set_fit_resample_request(self, *, fp_cost=False, fn_cost=False):
            pass

    def __init__(
            self,
            method: Literal['rejection sampling', 'oversampling'] = 'rejection sampling',
            oversampling_norm: float = 0.1,
            percentile_threshold: float = 97.5,
            random_state: int | np.random.RandomState | None = None
    ):
        super().__init__()
        self.method = method
        self.oversampling_norm = oversampling_norm
        self.percentile_threshold = percentile_threshold
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags = ClassifierTags(multi_class=False)
        tags.sampler_tags.sample_indices = True
        return tags

    def fit_resample(
            self,
            X: ArrayLike,
            y: ArrayLike,
            *,
            fp_cost: float | ArrayLike = 0.0,
            fn_cost: float | ArrayLike = 0.0
    ) -> tuple[NDArray, NDArray]:
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        y : array-like of shape (n_samples,)

        fp_cost : float or array-like, shape=(n_samples,), default=0.0
            Cost of false positives. If ``float``, then all false positives have the same cost.
            If array-like, then it is the cost of each false positive classification.

        fn_cost : float or array-like, shape=(n_samples,), default=0.0
            Cost of false negatives. If ``float``, then all false negatives have the same cost.
            If array-like, then it is the cost of each false negative classification.

        Returns
        -------
        X_resampled : ndarray of shape (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray of shape (n_samples_new,)
            The corresponding label of `X_resampled`.
        """
        return super().fit_resample(X, y, fp_cost=fp_cost, fn_cost=fn_cost)

    def _fit_resample(
            self,
            X: NDArray,
            y: NDArray,
            fp_cost: float | ArrayLike = 0.0,
            fn_cost: float | ArrayLike = 0.0,
    ) -> tuple[NDArray, NDArray]:
        if isinstance(fp_cost, (int, float)):
            fp_cost = np.full_like(y, fp_cost)
        else:
            fp_cost = np.asarray(fp_cost)
        if isinstance(fn_cost, (int, float)):
            fn_cost = np.full_like(y, fn_cost)
        else:
            fn_cost = np.asarray(fn_cost)
        rng = check_random_state(self.random_state)

        misclassification_costs = fp_cost
        misclassification_costs[y == 1] = fn_cost[y == 1]

        normalized_costs = np.minimum(
            misclassification_costs / np.percentile(misclassification_costs, self.percentile_threshold),
            1
        )

        n_samples = X.shape[0]

        if self.method == 'rejection sampling':
            rejection_probability = rng.rand(n_samples)
            self.sample_indices_ = np.arange(len(y))[rejection_probability <= normalized_costs]
        elif self.method == 'oversampling':
            # repeat each sample based on the normalized costs
            sample_repeats = np.ceil(normalized_costs / self.oversampling_norm).astype(np.int64)
            self.sample_indices_ = np.repeat(np.arange(n_samples), sample_repeats)
        else:
            raise ValueError(f'Method not valid. Expected RejectionSampling or OverSampling, got {self.method}.')

        X_re = X[self.sample_indices_]
        y_re = y[self.sample_indices_]

        return X_re, y_re
