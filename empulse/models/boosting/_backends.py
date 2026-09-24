"""
Per-boosting-library dispatch for :class:`~empulse.models.CSBoostClassifier`.

XGBoost, LightGBM and CatBoost each want the AEC/metric objective wired up differently -- a plain
callable, a callable object, or a ``(objective, metric)`` pair -- and each has its own quirk around
the shared ``_BASE_SCORE`` initialization (XGBoost's ``base_score`` is a genuine model parameter;
LightGBM's ``init_score`` and CatBoost's ``baseline`` bias training only and must be added back
manually in :meth:`~empulse.models.CSBoostClassifier.predict_proba`). :class:`BoostingBackend`
collects those differences as data instead of scattering them across ``isinstance`` chains in
``csboost.py``: one for which classifier matches, one for how the estimator is built, one for how
it is fit, one for how predictions are reconstructed.

:func:`backend_for` still needs to guard against ``isinstance(x, XGBClassifier)`` where
``XGBClassifier`` is itself the ``TypeVar`` placeholder ``csboost.py`` substitutes when a library
is not installed -- ``isinstance()``'s second argument must be a type, and a bare ``TypeVar`` is
not one. The classifier references are passed in by the caller (``csboost.py``) rather than
imported here a second time, since ``tests/models/test_csboost.py`` patches
``empulse.models.boosting.csboost.XGBClassifier`` (etc.) directly to simulate a missing library;
a second, independent import in this module would not see that patch.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, TypeVar

import numpy as np
from scipy.special import expit, logit

from ..._types import FloatNDArray, IntNDArray
from ...metrics import BaseMetric
from ...metrics._loss import cy_boost_grad_hess

# Hessian is 0 at score 0.5 because the AEC objective's hessian evaluates to p*(1-p), which is
# exactly 0 when p=0.5. A nudge of 1e-2 is large enough to produce a non-zero hessian at
# initialization (kick-starting the optimizer) yet small enough not to meaningfully bias the
# starting point away from 0.5.
#
# XGBoost's `base_score` is a probability, but LightGBM's `init_score` and CatBoost's `baseline`
# are raw (log-odds) scores - the same literal nudge does not mean "start at ~51% probability" in
# both spaces (expit(0.51) != 0.51). Two separate constants keep the *actual* starting probability
# consistent across backends.
_BASE_SCORE_PROBA = 0.5 + 1e-2
_BASE_SCORE_RAW = float(logit(_BASE_SCORE_PROBA))

_CATBOOST_WARNING_FILTERS: tuple[tuple[str, type[Warning]], ...] = (
    ('Can\'t optimize method "calc_ders_range" because self argument is used', UserWarning),
    ('Can\'t optimize method "evaluate" because self argument is used', UserWarning),
)


class LGBMObjective:
    """AEC objective for lightgbm."""

    def __init__(self, gradient_const: FloatNDArray):
        self.gradient_const = gradient_const

    def __call__(self, y_true: FloatNDArray, y_score: FloatNDArray) -> tuple[FloatNDArray, FloatNDArray]:
        """
        Compute the gradient and hessian of the AEC objective.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels.
        y_score : np.ndarray
            Raw model scores.

        Returns
        -------
        gradient : np.ndarray
            Gradient of the objective function.
        hessian : np.ndarray
            Hessian of the objective function.
        """
        gradient: FloatNDArray
        hessian: FloatNDArray
        # cy_boost_grad_hess (Cython) requires float64 memoryviews
        gradient, hessian = cy_boost_grad_hess(
            np.asarray(y_true, dtype=np.float64),
            np.asarray(y_score, dtype=np.float64),
            np.asarray(self.gradient_const, dtype=np.float64),
        )
        return gradient, hessian


class LGBMMetricObjective:
    """Metric objective wrapper for lightgbm using dynamic gradient/hessian evaluation."""

    def __init__(self, metric: BaseMetric, **loss_params: FloatNDArray | float):
        self.metric = metric
        self.loss_params = loss_params

    def __call__(self, y_true: FloatNDArray, y_score: FloatNDArray) -> tuple[FloatNDArray, FloatNDArray]:
        """Compute the gradient and hessian of the metric objective."""
        gradient, hessian = self.metric._gradient_boost_objective(y_true, y_score, **self.loss_params)
        return gradient, hessian


def _catboost_training_data(grad_const: FloatNDArray, y: IntNDArray) -> tuple[IntNDArray, FloatNDArray]:
    """
    Recast the AEC objective as a weighted classification problem CatBoost can train on natively.

    CatBoost hands its objective callback chunks of rows, out of order (and permuted, under ordered
    boosting), with nothing identifying which training rows they are. Its only per-row channels are
    the targets and the sample weights, and it also uses the weights itself when scoring splits and
    estimating leaves, so they cannot double as row indices without skewing training.

    The AEC gradient is ``grad_const * p * (1 - p)`` for a per-row constant ``grad_const``, so a row
    is described completely by the constant's sign and size. Training on the target
    ``y' = [grad_const < 0]`` (predicting positive lowers this row's cost) with sample weight
    ``|grad_const|`` (how much getting this row right is worth) carries both: the callback recovers
    ``grad_const = (1 - 2 * y') * weight``, and CatBoost's own use of the weights is then correct.

    A row with ``grad_const == 0`` has weight 0 and so no influence on training, whatever its target;
    it keeps its original label, so that costs which only ever penalize one kind of error (e.g. only
    ``fp_cost`` set) still leave CatBoost the two classes it insists on.

    Parameters
    ----------
    grad_const : ndarray of shape (n_samples,)
        The metric's constant gradient term, from ``BaseMetric._prepare_boost_objective``.
    y : ndarray of shape (n_samples,)
        The (0/1-encoded) training target.

    Returns
    -------
    target : ndarray of shape (n_samples,)
        The reformulated binary target ``y'``.
    sample_weight : ndarray of shape (n_samples,)
        The reformulated sample weight ``|grad_const|``.
    """
    target: IntNDArray = np.where(grad_const < 0, 1, np.where(grad_const > 0, 0, y)).astype(np.int64)
    sample_weight: FloatNDArray = np.abs(grad_const)
    return target, sample_weight


class CatBoostObjective:
    """AEC objective for catboost, trained on the reformulation built by :func:`_catboost_training_data`."""

    def calc_ders_range(
        self, predictions: Sequence[float], targets: Sequence[float], weights: Sequence[float] | None
    ) -> list[tuple[float, float]]:
        """
        Compute first and second derivative of the loss function with respect to the predicted value for each object.

        Parameters
        ----------
        predictions : indexed container of floats
            Current predictions (raw scores) for each object.

        targets : indexed container of floats
            The reformulated target ``y'`` of each object.

        weights : indexed container of floats
            The reformulated sample weight ``|grad_const|`` of each object.

        Returns
        -------
        list of (float, float)
            The first and second derivative of the loss w.r.t. the prediction, per object.
        """
        y_prime = np.asarray(targets, dtype=np.float64)
        weight = np.ones_like(y_prime) if weights is None else np.asarray(weights, dtype=np.float64)
        gradient, hessian = cy_boost_grad_hess(
            y_prime, np.asarray(predictions, dtype=np.float64), (1.0 - 2.0 * y_prime) * weight
        )
        # CatBoost maximizes, so it wants the derivatives of the negated loss.
        return list(zip(-gradient, -hessian, strict=True))


class CatBoostMetric:
    """
    AEC metric for catboost, evaluated on the reformulation built by :func:`_catboost_training_data`.

    On the training data, the weighted expected misclassification of ``y'`` differs from the
    expected cost only by a positive factor and a constant, so it ranks models the same way. An
    ``eval_set`` passed through ``fit_params`` is not reformulated, and is scored by its plain
    (unweighted) expected misclassification instead.
    """

    def is_max_optimal(self) -> bool:
        """Return whether greater values of metric are better."""
        return False

    def evaluate(
        self, approxes: Sequence[Sequence[float]], targets: Sequence[float], weights: Sequence[float] | None
    ) -> tuple[float, float]:
        """
        Evaluate metric value.

        Parameters
        ----------
        approxes : sequence of indexed containers of float
            Raw model outputs (logits) for each instance, one container per model dimension.

        targets : sequence of float
            The reformulated target ``y'`` of each instance.

        weights : sequence of float or None
            The reformulated sample weight ``|grad_const|`` of each instance.

        Returns
        -------
        weighted_error : float
            The summed weighted expected misclassification.
        total_weight : float
            The summed weights; CatBoost divides ``weighted_error`` by this to form the final error.
        """
        y_proba = expit(np.asarray(approxes[0], dtype=np.float64))
        y_prime = np.asarray(targets, dtype=np.float64)
        weight = np.ones_like(y_prime) if weights is None else np.asarray(weights, dtype=np.float64)
        error = np.where(y_prime == 1, 1.0 - y_proba, y_proba)
        return float(np.sum(weight * error)), float(np.sum(weight))

    def get_final_error(self, error: float, weight: float) -> float:
        """
        Return final value of metric based on error and weight.

        Parameters
        ----------
        error : float
            Sum of weighted errors in all instances.

        weight : float
            Sum of weights of all instances.

        Returns
        -------
        float
            The final metric value.
        """
        return error / weight if weight > 0 else 0.0


@dataclass(frozen=True)
class BoostingBackend:
    """
    The boosting-library-specific half of fitting and predicting with a :class:`CSBoostClassifier`.

    Parameters
    ----------
    name : {'xgboost', 'lightgbm', 'catboost'}
        Which library this backend wraps.
    classifier : type or None
        The library's classifier class, or ``None`` if the library is not installed.
    """

    name: Literal['xgboost', 'lightgbm', 'catboost']
    classifier: type | None

    @property
    def warning_filters(self) -> tuple[tuple[str, type[Warning]], ...]:
        """Warnings to suppress while calling this backend's ``fit`` (only catboost has any)."""
        if self.name == 'catboost':
            return _CATBOOST_WARNING_FILTERS
        return ()

    def build_default(self, objective: Any) -> Any:
        """
        Build the default estimator (only ever called for the xgboost backend).

        The caller is responsible for checking :attr:`classifier` is not ``None`` first (and
        raising an import error naming itself, which this method cannot do).
        """
        classifier = self.classifier
        if classifier is None:
            raise TypeError(f'{self.name} package is not installed.')
        return classifier(objective=objective, base_score=_BASE_SCORE_PROBA)

    def apply_objective(self, estimator: Any, objective: Any) -> Any:
        """Set *objective* (as built by :meth:`wrap_objective`) on a cloned user-supplied estimator."""
        if self.name == 'xgboost':
            return estimator.set_params(objective=objective, base_score=_BASE_SCORE_PROBA)
        if self.name == 'lightgbm':
            return estimator.set_params(objective=objective)
        loss_function, eval_metric = objective
        return estimator.set_params(loss_function=loss_function, eval_metric=eval_metric)

    def wrap_objective(
        self, loss: BaseMetric, y: FloatNDArray, loss_params: dict[str, Any], *, precomputed: bool
    ) -> Any:
        """
        Build this backend's objective (and, for catboost, metric) callable(s).

        Parameters
        ----------
        loss : BaseMetric
            The metric being optimized.
        y : ndarray
            Training targets, used only for their shape.
        loss_params : dict
            The metric's parameter values for this fit.
        precomputed : bool
            ``True`` when the strategy's gradient is a constant that can be computed once up
            front (``Capability.PRECOMPUTED_BOOST_OBJECTIVE``); ``False`` when it must be
            recomputed from the current round's predictions every iteration
            (``Capability.BOOST_OBJECTIVE``). The caller has already checked the metric declares
            one of the two.

        Raises
        ------
        ValueError
            For catboost when ``precomputed`` is ``False``: CatBoost only ever shows its objective a
            chunk of the rows, so it can neither compute an objective that depends on all of them
            (MaxProfit's) nor tell which rows' costs a per-row objective should use (LogCost's).
        """
        if not precomputed:
            if self.name == 'xgboost':
                return partial(loss._gradient_boost_objective, **loss_params)
            if self.name == 'lightgbm':
                return LGBMMetricObjective(loss, **loss_params)
            raise ValueError(
                f'The CatBoost backend does not support the {loss.strategy.name!r} strategy: CatBoost '
                'computes its objective on chunks of the training rows, which this strategy cannot be '
                'evaluated on. Use XGBClassifier or LGBMClassifier as the estimator, or a Cost or '
                'Savings loss.'
            )

        if self.name == 'catboost':
            # The per-row gradient constants reach CatBoost as its targets and sample weights, built
            # by `fit_arguments`; the objective itself needs no parameters.
            return CatBoostObjective(), CatBoostMetric()
        grad_const = loss._prepare_boost_objective(y, **loss_params).reshape(-1)
        if self.name == 'xgboost':
            # cy_boost_grad_hess (Cython) requires a float64 memoryview.
            return partial(cy_boost_grad_hess, grad_const=np.asarray(grad_const, dtype=np.float64))
        return LGBMObjective(grad_const)

    def fit_arguments(
        self,
        y: IntNDArray,
        loss: BaseMetric,
        loss_params: dict[str, Any],
        fit_params: dict[str, Any],
    ) -> tuple[IntNDArray, dict[str, Any]]:
        """
        Return the target and keyword arguments to pass to this backend's ``fit`` alongside ``X``.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            The (0/1-encoded) training target.
        loss : BaseMetric
            The metric being optimized.
        loss_params : dict
            The metric's parameter values for this fit.
        fit_params : dict
            The caller's extra arguments for the estimator's ``fit``; not modified.

        Returns
        -------
        y_fit : ndarray of shape (n_samples,)
            The target to fit on: ``y`` itself, except for catboost (see :func:`_catboost_training_data`).
        fit_kwargs : dict
            *fit_params* plus whatever this backend adds.
        """
        if self.name == 'lightgbm':
            return y, {'init_score': np.full(y.shape, _BASE_SCORE_RAW), **fit_params}
        if self.name == 'catboost':
            fit_kwargs = dict(fit_params)
            grad_const = np.asarray(loss._prepare_boost_objective(y, **loss_params), dtype=np.float64).reshape(-1)
            y_fit, sample_weight = _catboost_training_data(grad_const, y)
            if (user_weight := fit_kwargs.pop('sample_weight', None)) is not None:
                # A weighted AEC scales each row's gradient constant, so the weights just multiply.
                sample_weight = sample_weight * np.asarray(user_weight, dtype=np.float64).reshape(-1)
            if np.unique(y_fit).size < 2:
                raise ValueError(
                    'With these costs, the same prediction is the cheapest for every training sample, '
                    'so there is nothing for CatBoostClassifier to learn.'
                )
            fit_kwargs.update(sample_weight=sample_weight, baseline=np.full(y.shape, _BASE_SCORE_RAW))
            return y_fit, fit_kwargs
        return y, dict(fit_params)

    def raw_score(self, estimator: Any, X: Any) -> FloatNDArray | None:
        """
        Return the raw (pre-offset) score for *X*, or ``None`` when no reconstruction is needed.

        LightGBM's ``init_score`` and CatBoost's ``baseline`` (both set to ``_BASE_SCORE_RAW`` by
        :meth:`fit_kwargs`) bias training gradients only -- neither library persists them into the
        saved model, so the caller must add that same offset back before converting to a
        probability. XGBoost's ``base_score`` has no such issue: it is a genuine model parameter
        that its own ``predict_proba`` already accounts for, so this returns ``None`` for it (and
        for any estimator that matches no known backend).
        """
        if self.name == 'lightgbm':
            score: FloatNDArray = estimator.predict_proba(X, raw_score=True)
            return score
        if self.name == 'catboost':
            score = estimator.predict(X, prediction_type='RawFormulaVal')
            return score
        return None


def backend_for(estimator: Any, *, xgb_cls: type, lgbm_cls: type, catboost_cls: type) -> BoostingBackend | None:
    """
    Return the :class:`BoostingBackend` matching *estimator*'s type, or ``None`` if none does.

    *xgb_cls*/*lgbm_cls*/*catboost_cls* are passed in by the caller rather than imported here --
    see the module docstring for why.
    """
    if not isinstance(xgb_cls, TypeVar) and isinstance(estimator, xgb_cls):
        return BoostingBackend(name='xgboost', classifier=xgb_cls)
    if not isinstance(lgbm_cls, TypeVar) and isinstance(estimator, lgbm_cls):
        return BoostingBackend(name='lightgbm', classifier=lgbm_cls)
    if not isinstance(catboost_cls, TypeVar) and isinstance(estimator, catboost_cls):
        return BoostingBackend(name='catboost', classifier=catboost_cls)
    return None
