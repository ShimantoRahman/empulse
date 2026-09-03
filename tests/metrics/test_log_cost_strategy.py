"""Tests for the LogCost metric strategy.

LogCost mirrors Cost's structure (see test_strategies.py / cost_strategy.py) but its per-sample
loss uses log(s) / log(1 - s) instead of s / (1 - s), drawing a direct line to cross-entropy /
log loss. These tests check:

* LogCost reproduces :func:`~empulse.metrics.expected_log_cost_loss` through the Metric interface.
* The special case tp_cost=tn_cost=-1, fp_cost=fn_cost=0 reduces to standard log loss, including
  its well-known closed-form gradient sigmoid(x) - y.
* The logit and gradient-boosting gradients/hessians match finite differences.
* Every object handed back by the strategy (used inside cost-sensitive models) is a plain,
  picklable class -- no closures -- since these get pickled by sklearn (cloning, cross-validation,
  bagging/forest parallel estimators).
* End-to-end training works with CSLogitClassifier and CSBoostClassifier.
"""

import pickle

import numpy as np
import pytest
import sympy
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss

from empulse.metrics import Cost, CostMatrix, LogCost, Metric, expected_log_cost_loss
from empulse.metrics.metric.strategies.log_cost_strategy import (
    LogCostBoostGradient,
    LogCostLogitObjective,
    LogCostLoss,
)
from empulse.models import CSBoostClassifier, CSLogitClassifier


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


@pytest.fixture(scope='module')
def dataset():
    X, y = make_classification(n_samples=100, n_features=4, random_state=0, class_sep=1.5)
    return X, y.astype(float)


def test_log_cost_score_matches_native_function(dataset):
    """Metric(cost_matrix, LogCost()) should reproduce expected_log_cost_loss (normalize=True)."""
    _, y = dataset
    rng = np.random.default_rng(1)
    y_proba = rng.uniform(0.01, 0.99, size=y.shape)

    tp, tn, fp, fn = sympy.symbols('tp tn fp fn')
    cost_matrix = CostMatrix().add_tp_cost(tp).add_tn_cost(tn).add_fp_cost(fp).add_fn_cost(fn)
    metric = Metric(cost_matrix, LogCost())

    result = metric(y, y_proba, tp=0.3, tn=0.6, fp=1.2, fn=2.1)
    expected = expected_log_cost_loss(y, y_proba, tp_cost=0.3, tn_cost=0.6, fp_cost=1.2, fn_cost=2.1, normalize=True)

    assert result == pytest.approx(expected)


def test_log_cost_instance_dependent_costs(dataset):
    """LogCost should support instance-dependent (array-like) costs, like Cost does."""
    _, y = dataset
    rng = np.random.default_rng(2)
    y_proba = rng.uniform(0.01, 0.99, size=y.shape)
    fn_cost = rng.uniform(0.5, 2.0, size=y.shape)

    fn = sympy.symbols('fn')
    cost_matrix = CostMatrix().add_fn_cost(fn).add_fp_cost(1.0)
    metric = Metric(cost_matrix, LogCost())

    result = metric(y, y_proba, fn=fn_cost)
    expected = expected_log_cost_loss(y, y_proba, fn_cost=fn_cost, fp_cost=1.0, normalize=True)

    assert result == pytest.approx(expected)


def test_log_cost_reduces_to_log_loss(dataset):
    """tp_cost=tn_cost=-1, fp_cost=fn_cost=0 should be equivalent to standard log loss."""
    _, y = dataset
    rng = np.random.default_rng(3)
    y_proba = rng.uniform(0.01, 0.99, size=y.shape)

    cost_matrix = CostMatrix().add_tp_benefit(1.0).add_tn_benefit(1.0)
    metric = Metric(cost_matrix, LogCost())

    result = metric(y, y_proba)
    expected = log_loss(y, y_proba)

    assert result == pytest.approx(expected)


def test_log_cost_loss_is_picklable():
    cost_expr = sympy.symbols('tp tn fp fn')
    instance = LogCostLoss(*cost_expr)
    restored = pickle.loads(pickle.dumps(instance))
    assert restored is not None


def test_log_cost_boost_gradient_is_picklable():
    cost_expr = sympy.symbols('tp tn fp fn')
    instance = LogCostBoostGradient(*cost_expr)
    restored = pickle.loads(pickle.dumps(instance))
    assert restored is not None


def test_log_cost_logit_objective_is_picklable(dataset):
    X, y = dataset
    objective = LogCostLogitObjective(
        tp_benefit=1.0,
        tn_benefit=1.0,
        fp_cost=0.0,
        fn_cost=0.0,
        features=X,
        y_true=y,
        C=1.0,
        l1_ratio=0.0,
        soft_threshold=False,
        fit_intercept=True,
    )
    restored = pickle.loads(pickle.dumps(objective))
    weights = np.zeros(X.shape[1])
    loss_before, grad_before = objective.logit_loss_gradient(weights)
    loss_after, grad_after = restored.logit_loss_gradient(weights)
    assert loss_before == pytest.approx(loss_after)
    np.testing.assert_allclose(grad_before, grad_after)


def test_log_cost_full_metric_is_picklable():
    """The full Metric(cost_matrix, LogCost()) object must survive a pickle round-trip.

    This is what actually gets pickled when used as the ``loss`` of a cost-sensitive model
    (sklearn clone / cross-validation / bagging & forest parallel estimators).
    """
    tp, fp = sympy.symbols('tp fp')
    cost_matrix = CostMatrix().add_tp_cost(tp).add_fp_cost(fp)
    metric = Metric(cost_matrix, LogCost())

    restored = pickle.loads(pickle.dumps(metric))
    y = np.array([0, 1, 0, 1])
    y_proba = np.array([0.2, 0.8, 0.3, 0.9])
    assert restored(y, y_proba, tp=1.0, fp=2.0) == pytest.approx(metric(y, y_proba, tp=1.0, fp=2.0))


def test_log_cost_no_closures_in_objective_state():
    """None of the callable-bearing attributes on the objective should be closures/lambdas.

    Closures over enclosing-scope variables cannot be pickled by the standard ``pickle`` module;
    everything here must be a plain object/class instance instead.
    """
    import types

    X = np.eye(3)
    y = np.array([0.0, 1.0, 0.0])
    objective = LogCostLogitObjective(
        tp_benefit=1.0,
        tn_benefit=1.0,
        fp_cost=0.0,
        fn_cost=0.0,
        features=X,
        y_true=y,
        C=1.0,
        l1_ratio=0.0,
        soft_threshold=False,
        fit_intercept=True,
    )
    for value in vars(objective).values():
        assert not isinstance(value, types.FunctionType), 'objective state should not hold closures'


def test_log_cost_logit_gradient_matches_finite_difference(dataset):
    X, y = dataset
    rng = np.random.default_rng(4)
    tp_val, tn_val, fp_val, fn_val = 2.0, 1.0, 0.5, 1.5

    objective = LogCostLogitObjective(
        tp_benefit=tp_val,
        tn_benefit=tn_val,
        fp_cost=fp_val,
        fn_cost=fn_val,
        features=X,
        y_true=y,
        C=1e6,
        l1_ratio=0.0,
        soft_threshold=False,
        fit_intercept=False,
    )
    weights = rng.normal(scale=0.1, size=X.shape[1])
    _loss, gradient = objective.logit_loss_gradient(weights)

    eps = 1e-6
    numeric_grad = np.zeros_like(weights)
    for i in range(weights.size):
        wp, wm = weights.copy(), weights.copy()
        wp[i] += eps
        wm[i] -= eps
        numeric_grad[i] = (objective.logit_loss(wp) - objective.logit_loss(wm)) / (2 * eps)

    np.testing.assert_allclose(gradient, numeric_grad, atol=1e-5)


def test_log_cost_logit_gradient_reduces_to_logistic_regression_gradient(dataset):
    """In the log-loss special case, the gradient should be X.T @ (sigmoid(Xw) - y) / n."""
    X, y = dataset
    objective = LogCostLogitObjective(
        tp_benefit=1.0,
        tn_benefit=1.0,
        fp_cost=0.0,
        fn_cost=0.0,
        features=X,
        y_true=y,
        C=1e6,
        l1_ratio=0.0,
        soft_threshold=False,
        fit_intercept=False,
    )
    weights = np.full(X.shape[1], 0.05)
    _, gradient = objective.logit_loss_gradient(weights)

    s = _expit(X @ weights)
    expected_gradient = X.T @ (s - y) / X.shape[0]

    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-6)


def test_log_cost_gradient_boost_matches_finite_difference():
    rng = np.random.default_rng(5)
    n = 50
    y_true = rng.integers(0, 2, size=n).astype(float)
    y_score = rng.normal(scale=1.5, size=n)  # raw (pre-sigmoid) boosting scores

    boost_gradient = LogCostBoostGradient(*sympy.symbols('tp tn fp fn'))
    gradient, hessian = boost_gradient(y_true, y_score, tp=2.0, tn=1.0, fp=0.5, fn=1.5)

    def per_sample_loss(y_true_i, x_i, tp, tn, fp, fn):
        s = _expit(x_i)
        eps = np.finfo(float).eps
        s = np.clip(s, eps, 1 - eps)
        loss_const1 = y_true_i * -tp + (1 - y_true_i) * fp
        loss_const2 = y_true_i * fn - (1 - y_true_i) * tn
        return np.log(s) * loss_const1 + np.log(1 - s) * loss_const2

    eps_grad = 1e-6
    eps_hess = 1e-4  # larger step: central second differences amplify roundoff error at 1e-6
    numeric_grad = np.zeros(n)
    numeric_hess = np.zeros(n)
    for i in range(n):
        f = lambda x: per_sample_loss(y_true[i], x, tp=2.0, tn=1.0, fp=0.5, fn=1.5)  # noqa: B023
        numeric_grad[i] = (f(y_score[i] + eps_grad) - f(y_score[i] - eps_grad)) / (2 * eps_grad)
        numeric_hess[i] = (f(y_score[i] + eps_hess) - 2 * f(y_score[i]) + f(y_score[i] - eps_hess)) / eps_hess**2

    np.testing.assert_allclose(gradient, numeric_grad, atol=1e-4)
    np.testing.assert_allclose(hessian, np.abs(numeric_hess), atol=1e-3)


def test_log_cost_optimal_threshold_matches_cost():
    """The optimal decision threshold only depends on the (linear) cost matrix, not the training

    loss, so LogCost and Cost must agree.
    """
    tp, tn, fp, fn = sympy.symbols('tp tn fp fn')
    cost_matrix = CostMatrix().add_tp_cost(tp).add_tn_cost(tn).add_fp_cost(fp).add_fn_cost(fn)
    log_cost_metric = Metric(cost_matrix, LogCost())
    cost_metric = Metric(cost_matrix, Cost())

    y = np.array([0, 1, 0, 1])
    y_proba = np.array([0.2, 0.8, 0.3, 0.9])
    params = {'tp': 0.3, 'tn': 0.6, 'fp': 1.2, 'fn': 2.1}

    log_cost_threshold = log_cost_metric.optimal_threshold(y, y_proba, **params)
    cost_threshold = cost_metric.optimal_threshold(y, y_proba, **params)

    assert log_cost_threshold == pytest.approx(cost_threshold)


def test_log_cost_optimal_rate_matches_cost():
    tp, fp = sympy.symbols('tp fp')
    cost_matrix = CostMatrix().add_tp_cost(tp).add_fp_cost(fp)
    log_cost_metric = Metric(cost_matrix, LogCost())
    cost_metric = Metric(cost_matrix, Cost())

    y = np.array([0, 1, 0, 1])
    y_proba = np.array([0.2, 0.8, 0.3, 0.9])
    params = {'tp': 1.0, 'fp': 2.0}

    log_cost_rate = log_cost_metric.optimal_rate(y, y_proba, **params)
    cost_rate = cost_metric.optimal_rate(y, y_proba, **params)
    assert log_cost_rate == pytest.approx(cost_rate)


def test_log_cost_repr_and_latex_smoke():
    tp, fp = sympy.symbols('tp fp')
    cost_matrix = CostMatrix().add_tp_cost(tp).add_fp_cost(fp)
    metric = Metric(cost_matrix, LogCost())
    assert 'LogCost' in repr(metric)
    latex = metric._repr_latex_()
    assert isinstance(latex, str)
    assert latex.startswith('$')


def test_cslogit_accepts_log_cost_metric(dataset):
    from scipy.optimize import OptimizeResult

    X, y = dataset
    tp, fp = sympy.symbols('tp fp')
    cost_matrix = CostMatrix().add_tp_cost(tp).add_fp_cost(fp)
    metric = Metric(cost_matrix, LogCost())

    model = CSLogitClassifier(loss=metric, C=1e6, l1_ratio=0.0, soft_threshold=False)
    model.fit(X, y, tp=0.0, fp=1.0)

    assert isinstance(model.result_, OptimizeResult)
    assert model.coef_.shape == (X.shape[1],)
    assert np.all(np.isfinite(model.coef_))

    y_proba = model.predict_proba(X)
    assert y_proba.shape == (X.shape[0], 2)
    assert np.all(np.isfinite(y_proba))


def test_cslogit_log_cost_is_picklable_after_fit(dataset):
    """The fitted model (holding the LogCost-based Metric as `loss`) must be picklable,

    mirroring what sklearn does during cross-validation / grid search.
    """
    X, y = dataset
    cost_matrix = CostMatrix().add_tp_benefit(1.0).add_fp_cost(1.0)
    metric = Metric(cost_matrix, LogCost())

    model = CSLogitClassifier(loss=metric, C=1e6, l1_ratio=0.0, soft_threshold=False)
    model.fit(X, y)

    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.predict_proba(X), model.predict_proba(X))


def test_csboost_dispatches_log_cost_through_dynamic_gradient_boost_objective(dataset):
    """CSBoostClassifier must route LogCost through the dynamic gradient_boost_objective path.

    LogCost's per-sample loss is non-linear in the predicted probability, so it cannot use the
    generic ``cy_boost_grad_hess`` kernel that Cost/Savings rely on (that kernel assumes a single
    precomputed constant gradient, which only holds for a loss that is linear in the score). This
    does not require xgboost/lightgbm/catboost to be installed: it inspects what callable
    ``_get_objective`` builds for the 'xgboost' framework.
    """
    from functools import partial

    _X, y = dataset
    tp, fp = sympy.symbols('tp fp')
    cost_matrix = CostMatrix().add_tp_cost(tp).add_fp_cost(fp)
    metric = Metric(cost_matrix, LogCost())

    model = CSBoostClassifier(loss=metric)
    objective = model._get_objective('xgboost', y=y, loss=metric, tp=0.0, fp=1.0)

    assert isinstance(objective, partial)
    assert objective.func == metric._gradient_boost_objective

    # Sanity check: the returned objective actually computes finite gradients/hessians.
    y_score = np.zeros_like(y)
    gradient, hessian = objective(y, y_score)
    assert np.all(np.isfinite(gradient))
    assert np.all(np.isfinite(hessian))


def test_csboost_accepts_log_cost_metric(dataset):
    xgboost = pytest.importorskip('xgboost')
    X, y = dataset

    tp, fp = sympy.symbols('tp fp')
    cost_matrix = CostMatrix().add_tp_cost(tp).add_fp_cost(fp)
    metric = Metric(cost_matrix, LogCost())

    model = CSBoostClassifier(
        estimator=xgboost.XGBClassifier(n_estimators=5, max_depth=1, verbosity=0),
        loss=metric,
    )
    model.fit(X, y, tp=0.0, fp=1.0)
    y_proba = model.predict_proba(X)

    assert y_proba.shape == (X.shape[0], len(np.unique(y)))
    assert np.all(np.isfinite(y_proba))
