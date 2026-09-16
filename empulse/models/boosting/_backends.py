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

from ..._types import FloatNDArray
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


class CatBoostObjective:
    """AEC objective for catboost."""

    def __init__(self, metric_or_gradient_const: BaseMetric | FloatNDArray, **loss_params: FloatNDArray | float):
        self.metric = metric_or_gradient_const if isinstance(metric_or_gradient_const, BaseMetric) else None
        self.gradient_const = metric_or_gradient_const if isinstance(metric_or_gradient_const, np.ndarray) else None
        self.loss_params = loss_params

    def calc_ders_range(
        self, predictions: Sequence[float], targets: FloatNDArray, weights: FloatNDArray
    ) -> list[tuple[float, float]]:
        """
        Compute first and second derivative of the loss function with respect to the predicted value for each object.

        Parameters
        ----------
        predictions : indexed container of floats
            Current predictions for each object.

        targets : indexed container of floats
            Target values you provided with the dataset.

        weights : ndarray of float
            Here instance weights are used to pass the indices of the instances, not actual weights.

        Returns
        -------
        list of (float, float)
            The first and second derivative of the loss w.r.t. the prediction, per object.
        """
        weights = weights.astype(int)
        predictions = np.array(predictions, dtype=np.float64)

        if self.metric is not None:
            # Use weights as a proxy to index instance-dependent parameters.
            loss_params = {
                name: value[weights] if isinstance(value, np.ndarray) else value
                for (name, value) in self.loss_params.items()
            }
            gradient, hessian = self.metric._gradient_boost_objective(targets, predictions, **loss_params)
        else:
            gradient_const = self.gradient_const[weights]  # type: ignore[index]
            # cy_boost_grad_hess (Cython) requires a float64 memoryview; targets is typed as the
            # broader FloatNDArray to match catboost's own callback convention.
            gradient, hessian = cy_boost_grad_hess(np.asarray(targets, dtype=np.float64), predictions, gradient_const)
        # convert from two arrays to one list of tuples
        gradient_f = np.asarray(gradient, dtype=np.float32)
        hessian_f = np.asarray(hessian, dtype=np.float32)
        return list(zip(-gradient_f, -hessian_f, strict=False))


class CatBoostMetric:
    """AEC metric for catboost."""

    def __init__(self, metric: BaseMetric, **loss_params: FloatNDArray | float):
        self.metric = metric
        self.loss_params = loss_params

    def is_max_optimal(self) -> bool:
        """Return whether greater values of metric are better."""
        # `evaluate` reports `BaseMetric._loss`, which is minimized whatever the metric's direction.
        return False

    def evaluate(
        self, predictions: Sequence[float], targets: Sequence[float], weights: FloatNDArray
    ) -> tuple[float, float]:
        """
        Evaluate metric value.

        Parameters
        ----------
        predictions : sequence of float
            Raw model outputs (logits) for each instance.

        targets : sequence of float
            Vectors of true labels.

        weights : ndarray of float
            Here instance weights are used to pass the indices of the instances, not actual weights.

        Returns
        -------
        weighted_error : float
            The metric value, reported as a loss to be minimized.
        total_weight : float
            Always ``1``; CatBoost divides ``weighted_error`` by this to form the final error.
        """
        weights = weights.astype(int)
        # Use weights as a proxy to index the costs
        loss_params = {
            name: value[weights] if isinstance(value, np.ndarray) else value
            for (name, value) in self.loss_params.items()
        }

        y_proba = expit(predictions)
        return self.metric._loss(targets, y_proba, validate=False, **loss_params), 1

    def get_final_error(self, error: float, weight: float) -> float:
        """
        Return final value of metric based on error and weight.

        Parameters
        ----------
        error : float
            Sum of errors in all instances.

        weight : float
            Sum of weights of all instances.

        Returns
        -------
        float
            The final metric value.
        """
        return error


def _broadcast_loss_params(loss_params: dict[str, Any], shape: tuple[int, ...]) -> dict[str, Any]:
    """Normalize every loss parameter to shape ``(n_samples,)``, as catboost's callbacks need."""
    return {
        name: np.full(shape, param) if np.isscalar(param) else param.reshape(-1) for name, param in loss_params.items()
    }


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

    def check_fit_params(self, fit_params: dict[str, Any]) -> None:
        """
        Raise if *fit_params* contains something this backend cannot accept.

        Only catboost rejects ``sample_weight``: it is repurposed internally as an index proxy
        for instance-dependent loss parameters (see :meth:`fit_kwargs`), so a user-supplied
        weight would silently be discarded rather than used.
        """
        if self.name == 'catboost' and 'sample_weight' in fit_params:
            raise ValueError('Sample weights are not allowed when training CatBoostClassifier.')

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
        """
        if not precomputed:
            if self.name == 'xgboost':
                return partial(loss._gradient_boost_objective, **loss_params)
            if self.name == 'lightgbm':
                return LGBMMetricObjective(loss, **loss_params)
            catboost_params = _broadcast_loss_params(loss_params, y.shape)
            return CatBoostObjective(loss, **catboost_params), CatBoostMetric(loss, **catboost_params)

        grad_const = loss._prepare_boost_objective(y, **loss_params).reshape(-1)
        if self.name == 'xgboost':
            # cy_boost_grad_hess (Cython) requires a float64 memoryview.
            return partial(cy_boost_grad_hess, grad_const=np.asarray(grad_const, dtype=np.float64))
        if self.name == 'lightgbm':
            return LGBMObjective(grad_const)
        catboost_params = _broadcast_loss_params(loss_params, y.shape)
        return CatBoostObjective(grad_const), CatBoostMetric(loss, **catboost_params)

    def fit_kwargs(self, X: FloatNDArray, y: FloatNDArray) -> dict[str, Any]:
        """Extra keyword arguments this backend's ``fit`` needs beyond ``X``/``y``/``fit_params``."""
        if self.name == 'lightgbm':
            return {'init_score': np.full(y.shape, _BASE_SCORE_RAW)}
        if self.name == 'catboost':
            # CatBoost uses sample_weight internally as an index proxy (see check_fit_params).
            return {'sample_weight': np.arange(X.shape[0]), 'baseline': np.full(y.shape, _BASE_SCORE_RAW)}
        return {}

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
