import pickle
import re
import warnings

import numpy as np
import pytest
import sympy
import sympy.stats
from sklearn.datasets import make_classification

from empulse.metrics import (
    BaseMetric,
    Capability,
    Cost,
    CostMatrix,
    LogCost,
    MaxProfit,
    Metric,
    MixtureComponent,
    MixtureMetric,
    Profit,
    Savings,
    classification_threshold,
    empcs_score,
    mpcs_score,
)

from .reference.credit_scoring import empcs, mpcs

Y_TRUE = np.array([1, 0, 1, 0, 1, 0])
Y_SCORE = np.array([0.9, 0.1, 0.8, 0.3, 0.6, 0.4])


@pytest.fixture(scope='module')
def credit_scoring_metrics():
    """The EMPCS/MPCS cost structure: profit = gamma * pi_0 * F_0 - roi * pi_1 * F_1."""
    gamma, roi = sympy.symbols('gamma roi')
    credit_matrix_det = CostMatrix().add_tp_benefit(gamma).add_fp_cost(roi)
    metric_det = Metric(credit_matrix_det, MaxProfit())

    gamma_rv = sympy.stats.Uniform('gamma', 0, 1)
    credit_matrix_stoch = CostMatrix().add_tp_benefit(gamma_rv).add_fp_cost(roi)
    metric_stoch = Metric(credit_matrix_stoch, MaxProfit())

    return metric_det, metric_stoch


@pytest.fixture()
def empcs_mixture(credit_scoring_metrics):
    metric_det, metric_stoch = credit_scoring_metrics
    return MixtureMetric([
        MixtureComponent('success_rate', metric_det, {'gamma': 0.0}),
        MixtureComponent('default_rate', metric_det, {'gamma': 1.0}),
        MixtureComponent('1 - success_rate - default_rate', metric_stoch, {}),
    ])


PARAMETER_SETS = [
    {'success_rate': 0.55, 'default_rate': 0.1, 'roi': 0.2644},  # matches empcs_score's own defaults
    {'success_rate': 0.7, 'default_rate': 0.1, 'roi': 0.2644},
    {'success_rate': 0.55, 'default_rate': 0.01, 'roi': 0.2644},
    {'success_rate': 0.55, 'default_rate': 0.1, 'roi': 0.1},
    {'success_rate': 0.01, 'default_rate': 0.7, 'roi': 0.7},
]


@pytest.mark.parametrize('params', PARAMETER_SETS)
def test_mixture_score_matches_empcs(empcs_mixture, y_true_and_prediction, params):
    y, y_proba = y_true_and_prediction
    native_val = empcs_score(y, y_proba, **params)
    mixture_val = empcs_mixture(y, y_proba, **params)
    assert pytest.approx(mixture_val) == native_val


@pytest.mark.parametrize('params', PARAMETER_SETS)
def test_mixture_rate_matches_empcs(empcs_mixture, y_true_and_prediction, params):
    y, y_proba = y_true_and_prediction
    _, native_rate = empcs(y, y_proba, **params)
    mixture_rate = empcs_mixture.optimal_rate(y, y_proba, **params)
    assert pytest.approx(mixture_rate) == native_rate


def test_mixture_optimal_threshold_matches_rate(empcs_mixture, y_true_and_prediction):
    y, y_proba = y_true_and_prediction
    params = {'success_rate': 0.55, 'default_rate': 0.1, 'roi': 0.2644}
    rate = empcs_mixture.optimal_rate(y, y_proba, **params)
    expected_threshold = classification_threshold(y, y_proba, rate)
    assert empcs_mixture.optimal_threshold(y, y_proba, **params) == pytest.approx(expected_threshold)


@pytest.mark.parametrize(
    'loan_lost_rate,roi',
    [(0.275, 0.2644), (0.7, 0.2644), (0.01, 0.1), (0.9, 0.9)],
)
def test_single_deterministic_component_matches_mpcs(
    credit_scoring_metrics, y_true_and_prediction, loan_lost_rate, roi
):
    """A one-component mixture (weight=1) should reduce exactly to the deterministic metric."""
    metric_det, _ = credit_scoring_metrics
    y, y_proba = y_true_and_prediction
    mixture = MixtureMetric([MixtureComponent(1.0, metric_det, {'gamma': loan_lost_rate})])

    native_val = mpcs_score(y, y_proba, loan_lost_rate=loan_lost_rate, roi=roi)
    mixture_val = mixture(y, y_proba, roi=roi)
    assert pytest.approx(mixture_val) == native_val

    _, native_rate = mpcs(y, y_proba, loan_lost_rate=loan_lost_rate, roi=roi)
    mixture_rate = mixture.optimal_rate(y, y_proba, roi=roi)
    assert pytest.approx(mixture_rate) == native_rate


def test_mixture_requires_at_least_one_component():
    with pytest.raises(ValueError, match='at least one component'):
        MixtureMetric([])


@pytest.mark.parametrize(
    ('make_component', 'message'),
    [
        (lambda metric: (metric, 0.5), 'must be a MixtureComponent'),
        (lambda metric: (metric, 0.5, {}), 'metric of MixtureMetric component 0 must be a BaseMetric'),
        (lambda metric: (None, metric, {}), 'weight of MixtureMetric component 0'),
        (lambda metric: (0.5, metric, None), 'parameters of MixtureMetric component 0'),
    ],
    ids=['two_fields', 'fields_out_of_order', 'no_weight', 'no_parameters'],
)
def test_mixture_rejects_malformed_components(make_component, message):
    metric = Metric(CostMatrix().add_fp_cost('a'), Cost())
    with pytest.raises(TypeError, match=message):
        MixtureMetric([make_component(metric)])


def test_mixture_missing_weight_parameter_raises(empcs_mixture, y_true_and_prediction):
    y, y_proba = y_true_and_prediction
    with pytest.raises(ValueError, match=re.escape("expected a value for weight parameter(s) ['default_rate']")):
        empcs_mixture(y, y_proba, success_rate=0.55, roi=0.2644)


def test_mixture_constant_weight():
    gamma = sympy.symbols('gamma')
    metric = Metric(CostMatrix().add_tp_benefit(gamma), MaxProfit())
    mixture = MixtureMetric([MixtureComponent(0.5, metric, {}), MixtureComponent(0.5, metric, {})])
    y_true = [1, 0, 1, 0, 1]
    y_score = [0.9, 0.1, 0.8, 0.2, 0.7]
    # two half-weighted copies of the same metric should equal the metric itself
    assert mixture(y_true, y_score, gamma=1.0) == pytest.approx(metric(y_true, y_score, gamma=1.0))


def test_mixture_direction_mismatch_raises():
    a, b = sympy.symbols('a b')
    max_metric = Metric(CostMatrix().add_tp_benefit(a), MaxProfit())
    cost_metric = Metric(CostMatrix().add_tp_cost(b), Cost())
    with pytest.raises(ValueError, match='inconsistent optimization directions'):
        MixtureMetric([
            MixtureComponent(0.5, max_metric, {}),
            MixtureComponent(0.5, cost_metric, {}),
        ])


def test_mixture_direction_consistent(empcs_mixture):
    assert empcs_mixture.direction == MaxProfit().direction


def test_mixture_repr(empcs_mixture):
    assert 'MixtureMetric' in repr(empcs_mixture)


def test_mixture_name_property(empcs_mixture):
    assert 'MixtureMetric' in empcs_mixture.__name__


# ---- BaseMetric interface: what makes MixtureMetric usable as a model `loss` ----
# The cost-sensitive models accept any BaseMetric as their `loss`, not just a plain Metric. These
# tests verify that MixtureMetric satisfies that shared interface, including the extra members
# (`strategy`, `_all_symbols`, `_is_deterministic`, `_evaluate_costs`) that model code relies on
# beyond score/rate/threshold/gradient.


def test_mixture_is_a_base_metric(empcs_mixture, credit_scoring_metrics):
    metric_det, _ = credit_scoring_metrics
    assert isinstance(metric_det, BaseMetric)
    assert isinstance(empcs_mixture, BaseMetric)


def test_base_metric_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseMetric()  # type: ignore[abstract]


def test_mixture_strategy_matches_components(credit_scoring_metrics):
    metric_det, _ = credit_scoring_metrics
    mixture = _two_point_mixture(metric_det, 0.6, 0.4)
    assert isinstance(mixture.strategy, MaxProfit)


def test_mixture_strategy_mismatch_raises():
    a, b = sympy.symbols('a b')
    max_metric = Metric(CostMatrix().add_tp_benefit(a), MaxProfit())
    profit_metric = Metric(CostMatrix().add_tp_benefit(b), Profit())
    mixture = MixtureMetric([
        MixtureComponent(0.5, max_metric, {}),
        MixtureComponent(0.5, profit_metric, {}),
    ])
    with pytest.raises(ValueError, match='inconsistent strategies'):
        _ = mixture.strategy


def test_mixture_all_symbols_includes_weight_and_component_names(empcs_mixture, credit_scoring_metrics):
    _, metric_stoch = credit_scoring_metrics
    symbols = empcs_mixture._all_symbols
    assert {'success_rate', 'default_rate', 'roi'} <= symbols
    # 'gamma' is fixed via `parameters` on the two deterministic components, so it is only
    # contributed by the stochastic component -- exactly as it would be for that Metric alone.
    assert symbols - {'success_rate', 'default_rate', 'roi'} == metric_stoch._all_symbols - {'roi'}


def test_mixture_missing_parameters_nothing_supplied(empcs_mixture):
    """Without a mixture-level default, every weight name and component symbol is required."""
    assert empcs_mixture._missing_parameters([]) == {'success_rate', 'default_rate', 'roi'}


def test_mixture_missing_parameters_all_supplied(empcs_mixture):
    assert empcs_mixture._missing_parameters(['success_rate', 'default_rate', 'roi']) == set()


def test_mixture_missing_parameters_partial(empcs_mixture):
    assert empcs_mixture._missing_parameters(['success_rate']) == {'default_rate', 'roi'}


def test_mixture_missing_parameters_mixture_level_default_covers_every_component():
    """A parameter covered by MixtureMetric's own `defaults` counts as supplied everywhere."""
    gamma, roi = sympy.symbols('gamma roi')
    metric_det = Metric(CostMatrix().add_tp_benefit(gamma).add_fp_cost(roi), MaxProfit())
    mixture = MixtureMetric(
        [
            MixtureComponent('success_rate', metric_det, {'gamma': 0.0}),
            MixtureComponent('default_rate', metric_det, {'gamma': 1.0}),
        ],
        defaults={'roi': 0.2644},
    )
    assert mixture._missing_parameters([]) == {'success_rate', 'default_rate'}
    assert mixture._missing_parameters(['success_rate', 'default_rate']) == set()


def test_mixture_missing_parameters_real_prebuilt_metric_has_no_missing_by_default():
    """empcs_score sets defaults for every one of its parameters, so nothing is ever missing."""
    assert empcs_score._missing_parameters([]) == set()


def test_mixture_is_deterministic_false_with_stochastic_component(empcs_mixture):
    assert empcs_mixture._is_deterministic is False


def test_mixture_is_deterministic_true_for_point_masses_only(credit_scoring_metrics):
    metric_det, _ = credit_scoring_metrics
    mixture = _two_point_mixture(metric_det, 0.6, 0.4)
    assert mixture._is_deterministic is True


@pytest.mark.parametrize('replace_stochastic', [False, True])
def test_mixture_evaluate_costs_matches_manual_combination(credit_scoring_metrics, replace_stochastic):
    metric_det, _ = credit_scoring_metrics
    mixture = _two_point_mixture(metric_det, 0.6, 0.4)
    roi = 0.2644

    fp, fn, tp, tn = mixture._evaluate_costs(replace_stochastic=replace_stochastic, roi=roi)
    fp0, fn0, tp0, tn0 = metric_det._evaluate_costs(replace_stochastic=replace_stochastic, gamma=0.0, roi=roi)
    fp1, fn1, tp1, tn1 = metric_det._evaluate_costs(replace_stochastic=replace_stochastic, gamma=1.0, roi=roi)

    assert pytest.approx(fp) == 0.6 * fp0 + 0.4 * fp1
    assert pytest.approx(fn) == 0.6 * fn0 + 0.4 * fn1
    assert pytest.approx(tp) == 0.6 * tp0 + 0.4 * tp1
    assert pytest.approx(tn) == 0.6 * tn0 + 0.4 * tn1


def test_mixture_evaluate_costs_replace_stochastic_with_stochastic_component(credit_scoring_metrics):
    """A mixture with a stochastic component can still evaluate scalar costs when means

    are substituted for its random variables (replace_stochastic=True), the same way a
    standalone stochastic Metric can. This is what lets a MaxProfit-strategy MixtureMetric
    be used by the scalar-cost model family (ProfLogit, ProfSR, ProfMPM, ProfMEMPM).
    """
    metric_det, metric_stoch = credit_scoring_metrics
    mixture = MixtureMetric([
        MixtureComponent(0.55, metric_det, {'gamma': 0.0}),
        MixtureComponent(0.1, metric_det, {'gamma': 1.0}),
        MixtureComponent(0.35, metric_stoch, {}),
    ])
    fp, fn, tp, tn = mixture._evaluate_costs(replace_stochastic=True, roi=0.2644)
    for cost in (fp, fn, tp, tn):
        assert np.isfinite(cost)


# ---- gradient-boosting / logit objective plumbing checks ----
# There is no independent native EMPCS gradient to compare against, so these tests check that
# MixtureMetric's combination is *self-consistent*: it must equal manually combining each
# component's own gradient/hessian/loss with the same weights.
#
# They use an all-deterministic 2-component mixture (gamma fixed to 0 and 1), not the full
# 3-component EMPCS mixture with its Uniform(0, 1)-distributed component. MaxProfit's
# gradient_boost_objective/logit_objective only support deterministic metrics and stochastic
# metrics with closed-form partial moments of a positive variable (Beta, Gamma, Pareto, etc.).
# Uniform is not one of them, so metric_stoch._gradient_boost_objective(...)/._logit_objective(...)
# raise NotImplementedError regardless of MixtureMetric; see
# test_full_mixture_gradient_training_not_yet_supported below.


def _two_point_mixture(metric_det, weight_at_0=0.6, weight_at_1=0.4):
    return MixtureMetric([
        MixtureComponent(weight_at_0, metric_det, {'gamma': 0.0}),
        MixtureComponent(weight_at_1, metric_det, {'gamma': 1.0}),
    ])


def _with_intercept(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))


def test_gradient_boost_objective_matches_manual_combination(y_true_and_prediction):
    gamma, roi = sympy.symbols('gamma roi')
    metric_det = Metric(CostMatrix().add_tp_benefit(gamma).add_fp_cost(roi), MaxProfit())
    y, y_proba = y_true_and_prediction
    w0, w1, roi_val = 0.6, 0.4, 0.2644
    mixture = _two_point_mixture(metric_det, w0, w1)

    grad, hess = mixture._gradient_boost_objective(y, y_proba, roi=roi_val)

    grad0, hess0 = metric_det._gradient_boost_objective(y, y_proba, gamma=0.0, roi=roi_val)
    grad1, hess1 = metric_det._gradient_boost_objective(y, y_proba, gamma=1.0, roi=roi_val)

    expected_grad = w0 * grad0 + w1 * grad1
    expected_hess = w0 * hess0 + w1 * hess1

    np.testing.assert_allclose(grad, expected_grad)
    np.testing.assert_allclose(hess, expected_hess)
    assert np.all(np.isfinite(grad))
    assert np.all(np.isfinite(hess))


def test_prepare_boost_objective_matches_manual_combination(y_true_and_prediction):
    # MaxProfit does not implement prepare_boost_objective (only Cost/Savings do), so this uses a
    # Cost-strategy mixture, which exercises the same MixtureMetric plumbing.
    a, b = sympy.symbols('a b')
    metric_cost = Metric(CostMatrix().add_fp_cost(a).add_fn_cost(b), Cost())
    y, _ = y_true_and_prediction
    w0, w1 = 0.6, 0.4
    mixture = MixtureMetric([
        MixtureComponent(w0, metric_cost, {'a': 5.0, 'b': 2.0}),
        MixtureComponent(w1, metric_cost, {'a': 1.0, 'b': 8.0}),
    ])

    const = mixture._prepare_boost_objective(y)

    const0 = metric_cost._prepare_boost_objective(y, a=5.0, b=2.0)
    const1 = metric_cost._prepare_boost_objective(y, a=1.0, b=8.0)
    expected = w0 * const0 + w1 * const1

    np.testing.assert_allclose(const, expected)


def test_logit_objective_matches_manual_combination(credit_scoring_metrics):
    metric_det, _ = credit_scoring_metrics
    X_raw, y = make_classification(n_samples=60, n_features=4, n_informative=2, n_redundant=0, random_state=3)
    X = _with_intercept(X_raw)
    w0, w1, roi = 0.6, 0.4, 0.2644
    mixture = _two_point_mixture(metric_det, w0, w1)

    common_kwargs = {'C': 1.0, 'l1_ratio': 0.0, 'fit_intercept': True}
    objective = mixture._logit_objective(features=X, y_true=y, roi=roi, **common_kwargs)
    objective0 = metric_det._logit_objective(features=X, y_true=y, gamma=0.0, roi=roi, **common_kwargs)
    objective1 = metric_det._logit_objective(features=X, y_true=y, gamma=1.0, roi=roi, **common_kwargs)

    weights = np.zeros(X.shape[1], dtype=np.float64)
    loss, grad = objective.logit_loss_gradient(weights)

    loss0, grad0 = objective0.logit_loss_gradient(weights)
    loss1, grad1 = objective1.logit_loss_gradient(weights)
    expected_loss = w0 * loss0 + w1 * loss1
    expected_grad = w0 * grad0 + w1 * grad1

    assert pytest.approx(loss) == expected_loss
    np.testing.assert_allclose(grad, expected_grad)
    assert grad.shape == weights.shape
    assert np.all(np.isfinite(grad))


def test_logit_objective_with_indices_and_set_alpha(credit_scoring_metrics):
    metric_det, _ = credit_scoring_metrics
    X_raw, y = make_classification(n_samples=60, n_features=4, n_informative=2, n_redundant=0, random_state=3)
    X = _with_intercept(X_raw)
    roi = 0.2644
    mixture = _two_point_mixture(metric_det, 0.6, 0.4)
    common_kwargs = {'C': 1.0, 'l1_ratio': 0.0, 'fit_intercept': True}
    objective = mixture._logit_objective(features=X, y_true=y, roi=roi, **common_kwargs)

    # set_alpha should not raise, even though this cost structure has no annealed component
    objective.set_alpha(2.5)

    indices = np.arange(0, 30)
    sub_objective = objective.with_indices(indices)
    weights = np.zeros(X.shape[1], dtype=np.float64)
    loss, grad = sub_objective.logit_loss_gradient(weights)
    assert np.isfinite(loss)
    assert grad.shape == weights.shape
    assert np.all(np.isfinite(grad))


def test_logit_gradient_steps_matches_direct_call():
    gamma, roi_sym = sympy.symbols('gamma roi')
    metric_det = Metric(CostMatrix().add_tp_benefit(gamma).add_fp_cost(roi_sym), MaxProfit())
    X_raw, y = make_classification(n_samples=40, n_features=3, n_informative=2, n_redundant=0, random_state=5)
    X = _with_intercept(X_raw)
    roi = 0.2644
    mixture = _two_point_mixture(metric_det, 0.6, 0.4)
    common_kwargs = {'C': 1.0, 'l1_ratio': 0.0, 'fit_intercept': True}
    objective = mixture._logit_objective(features=X, y_true=y, roi=roi, **common_kwargs)

    weights = np.zeros(X.shape[1], dtype=np.float64)
    direct_grad = objective.logit_gradient(weights)

    steps = objective.logit_gradient_steps()
    step_grad = steps.send(weights)
    steps.close()

    np.testing.assert_allclose(step_grad, direct_grad)


def test_full_mixture_gradient_training_not_yet_supported(credit_scoring_metrics, y_true_and_prediction):
    """MaxProfit's gradient_boost_objective/logit_objective only support deterministic metrics and
    stochastic metrics with closed-form partial moments of a positive variable (Beta, Gamma, Pareto, ...).

    Uniform is not one of those, so the EMPCS mixture's third (Uniform) component cannot be used for
    gradient-based training. MixtureMetric propagates that NotImplementedError rather than silently
    producing a wrong gradient.
    """
    metric_det, metric_stoch = credit_scoring_metrics
    y, y_proba = y_true_and_prediction
    mixture = MixtureMetric([
        MixtureComponent(0.55, metric_det, {'gamma': 0.0}),
        MixtureComponent(0.1, metric_det, {'gamma': 1.0}),
        MixtureComponent(0.35, metric_stoch, {}),
    ])
    with pytest.raises(NotImplementedError):
        mixture._gradient_boost_objective(y, y_proba, roi=0.2644)


# ---- optimal_rate with hardcoded Uniform bounds ----


@pytest.mark.parametrize('integration_method', ['auto', 'quad'])
def test_hardcoded_uniform_bounds_optimal_rate_no_error(y_true_and_prediction, integration_method):
    """``optimal_rate()`` (not just ``__call__``) works when the Uniform distribution's bounds are
    hardcoded literals with no free symbols.
    """
    y, y_proba = y_true_and_prediction
    clv, d, f = sympy.symbols('clv d f')
    gamma = sympy.stats.Uniform('gamma', 0, 1)
    cost_matrix = (
        CostMatrix().add_tp_benefit(gamma * (clv - d - f)).add_tp_benefit((1 - gamma) * -f).add_fp_cost('d + f')
    )
    metric = Metric(cost_matrix, MaxProfit(integration_method=integration_method, random_state=12))

    rate = metric.optimal_rate(y, y_proba, clv=100, d=10, f=1)
    assert isinstance(rate, float)
    assert np.isfinite(rate)


def test_hardcoded_uniform_bounds_optimal_rate_matches_across_methods(y_true_and_prediction):
    y, y_proba = y_true_and_prediction
    clv, d, f = sympy.symbols('clv d f')
    gamma = sympy.stats.Uniform('gamma', 0, 1)
    cost_matrix = (
        CostMatrix().add_tp_benefit(gamma * (clv - d - f)).add_tp_benefit((1 - gamma) * -f).add_fp_cost('d + f')
    )

    rate_auto = Metric(cost_matrix, MaxProfit(integration_method='auto')).optimal_rate(y, y_proba, clv=100, d=10, f=1)
    rate_quad = Metric(cost_matrix, MaxProfit(integration_method='quad')).optimal_rate(y, y_proba, clv=100, d=10, f=1)
    # 'auto' uses the exact scipy-CDF piecewise path while 'quad' numerically integrates;
    # they should agree closely but not to full float precision.
    assert pytest.approx(rate_auto, rel=1e-3) == rate_quad


class TestOutlierSensitiveParameters:
    """`MixtureMetric._outlier_sensitive_parameters` merges each component's own mapping."""

    def test_merges_across_components(self):
        a, b = sympy.symbols('a b')
        first = Metric(CostMatrix().add_tp_benefit(a).mark_outlier_sensitive(a), Cost())
        second = Metric(CostMatrix().add_fp_cost(b).mark_outlier_sensitive(b), Cost())
        mixture = MixtureMetric([MixtureComponent(0.5, first, {}), MixtureComponent(0.5, second, {})])
        assert mixture._outlier_sensitive_parameters() == {'a': 'positive', 'b': 'negative'}

    def test_components_that_disagree_give_both(self):
        a, b = sympy.symbols('a b')
        positive = Metric(CostMatrix().add_fn_cost(a).add_fp_cost(b).mark_outlier_sensitive(a), Cost())
        negative = Metric(CostMatrix().add_fp_cost(a).add_fn_cost(b).mark_outlier_sensitive(a), Cost())
        for components in ((positive, negative), (negative, positive)):
            mixture = MixtureMetric([MixtureComponent(0.5, metric, {}) for metric in components])
            assert mixture._outlier_sensitive_parameters() == {'a': 'both'}

    def test_empty_when_no_component_marks_anything(self):
        a = sympy.symbols('a')
        metric = Metric(CostMatrix().add_tp_benefit(a), Cost())
        mixture = MixtureMetric([MixtureComponent(1.0, metric, {})])
        assert mixture._outlier_sensitive_parameters() == {}


@pytest.mark.parametrize('method', ['__call__', 'optimal_rate', 'optimal_threshold'])
def test_mixture_forwards_validate_to_components(monkeypatch, empcs_mixture, y_true_and_prediction, method):
    """``validate=False`` is the training loop's promise that the values were checked already.

    The mixture must forward it, or every component re-validates on every call.
    """
    y, y_proba = y_true_and_prediction
    validating_calls = 0
    original = Metric._prepare_parameters

    def counting(self, *, n_samples=None, validate=True, **kwargs):
        nonlocal validating_calls
        validating_calls += validate
        return original(self, n_samples=n_samples, validate=validate, **kwargs)

    monkeypatch.setattr(Metric, '_prepare_parameters', counting)
    getattr(empcs_mixture, method)(y, y_proba, validate=False, **PARAMETER_SETS[0])
    assert validating_calls == 0

    getattr(empcs_mixture, method)(y, y_proba, **PARAMETER_SETS[0])
    assert validating_calls > 0


@pytest.mark.parametrize('strategy', [Cost, LogCost])
@pytest.mark.parametrize('method', ['logit_loss', 'logit_gradient', 'logit_loss_gradient'])
def test_logit_objective_applies_the_penalty_once(strategy, method):
    """The mixture replaces its components' penalties with one of its own, rather than adding it."""
    fp, fn = sympy.symbols('fp fn')
    metric = Metric(CostMatrix().add_tp_benefit(1).add_fp_cost(fp).add_fn_cost(fn), strategy())
    X_raw, y = make_classification(n_samples=80, n_features=3, n_informative=2, n_redundant=0, random_state=4)
    X = _with_intercept(X_raw)
    mixture = MixtureMetric([
        MixtureComponent(0.3, metric, {'fp': 1.0, 'fn': 5.0}),
        MixtureComponent(0.7, metric, {'fp': 2.0, 'fn': 1.0}),
    ])
    arguments = {'features': X, 'y_true': y, 'C': 0.05, 'l1_ratio': 0.5, 'fit_intercept': True}
    objective = mixture._logit_objective(**arguments)
    components = [
        metric._logit_objective(**arguments, fp=1.0, fn=5.0),
        metric._logit_objective(**arguments, fp=2.0, fn=1.0),
    ]
    weights = np.array([0.1, -0.8, 0.5, 1.2])
    assert objective.penalty.value(weights) > 0

    data_loss = 0.3 * components[0].data_loss(weights) + 0.7 * components[1].data_loss(weights)
    data_gradient = 0.3 * components[0].data_gradient(weights) + 0.7 * components[1].data_gradient(weights)
    expected = {
        'logit_loss': data_loss + objective.penalty.value(weights),
        'logit_gradient': data_gradient + objective.penalty.gradient(weights),
    }
    expected['logit_loss_gradient'] = (expected['logit_loss'], expected['logit_gradient'])

    def flatten(value):
        return np.concatenate([np.ravel(part) for part in (value if isinstance(value, tuple) else (value,))])

    np.testing.assert_allclose(flatten(getattr(objective, method)(weights)), flatten(expected[method]), rtol=1e-12)


def test_logit_objective_forwards_n_threads_to_its_components():
    fp, fn = sympy.symbols('fp fn')
    metric = Metric(CostMatrix().add_fp_cost(fp).add_fn_cost(fn), Cost())
    mixture = MixtureMetric([
        MixtureComponent(0.3, metric, {'fp': 1.0, 'fn': 5.0}),
        MixtureComponent(0.7, metric, {'fp': 2.0, 'fn': 1.0}),
    ])
    X = _with_intercept(np.random.default_rng(0).normal(size=(40, 2)))
    y = np.arange(40) % 3 == 0
    objective = mixture._logit_objective(features=X, y_true=y, C=0.1, l1_ratio=0.5, fit_intercept=True)
    assert objective.n_threads == 1
    objective.n_threads = 4
    assert objective.n_threads == 4
    assert [component.n_threads for _, component in objective._weighted_objectives] == [4, 4]
    assert objective.with_indices(np.arange(10)).n_threads == 4


class TestCostOnlyComponents:
    """A mixture of components that decide from their costs alone decides from the combined costs."""

    a, b = sympy.symbols('a b')

    @pytest.fixture
    def mixture(self):
        metric = Metric(CostMatrix().add_fp_cost(self.a).add_fn_cost(self.b), Cost())
        return MixtureMetric([
            MixtureComponent(0.5, metric, {'a': 1.0, 'b': 9.0}),
            MixtureComponent(0.5, metric, {'a': 9.0, 'b': 1.0}),
        ])

    def test_threshold_is_that_of_the_combined_costs(self, mixture, seeded_rng):
        # The combined costs are fp = fn = 5, whose threshold is 0.5 on any scores.
        y_score = seeded_rng.random(1000) ** 3
        y_true = (seeded_rng.random(1000) < y_score).astype(int)
        assert mixture.optimal_threshold(y_true, y_score) == pytest.approx(0.5)
        assert mixture.optimal_threshold(np.array([]), np.array([])) == pytest.approx(0.5)
        assert mixture.optimal_rate(y_true, y_score) == pytest.approx(np.mean(y_score >= 0.5))

    def test_components_of_different_strategies_offer_no_cost_only_decision(self):
        metric = Metric(CostMatrix().add_fp_cost(self.a).add_fn_cost(self.b), Cost())
        log_metric = Metric(CostMatrix().add_fp_cost(self.a).add_fn_cost(self.b), LogCost())
        mixture = MixtureMetric([MixtureComponent(0.5, metric, {}), MixtureComponent(0.5, log_metric, {})])
        assert Capability.COST_ONLY_DECISION not in mixture.capabilities
        assert Capability.OPTIMAL_THRESHOLD not in mixture.capabilities

    @pytest.mark.parametrize('method', ['optimal_threshold', 'optimal_rate'])
    def test_a_decision_it_does_not_support_raises_rather_than_average_the_components(self, method):
        """Averaging the rates of Profit and Savings, which decide by their own costs, is no decision of the mixture."""
        cost_matrix = CostMatrix().add_fp_cost(self.a).add_fn_cost(self.b)
        mixture = MixtureMetric([
            MixtureComponent(0.5, Metric(cost_matrix, Profit()), {'a': 1.0, 'b': 9.0}),
            MixtureComponent(0.5, Metric(cost_matrix, Savings()), {'a': 9.0, 'b': 1.0}),
        ])
        assert Capability.OPTIMAL_RATE not in mixture.capabilities
        with pytest.raises(ValueError, match=f'does not support {method}'):
            getattr(mixture, method)(Y_TRUE, Y_SCORE)


CHEAP_COSTS = {'a': 1.0, 'b': 10.0}
COSTLY_COSTS = {'a': 400.0, 'b': 100.0}


class TestSavingsComponents:
    """
    Savings divide each component's cost by its baseline's cost, so the mixture weighs the costs alike.

    Combining the costs with the mixture weights alone let the component with the largest costs decide.
    """

    a, b = sympy.symbols('a b')

    @pytest.fixture
    def metric(self):
        return Metric(CostMatrix().add_fp_cost(self.a).add_fn_cost(self.b), Savings())

    @pytest.fixture
    def mixture(self, metric):
        return MixtureMetric([MixtureComponent(0.5, metric, CHEAP_COSTS), MixtureComponent(0.5, metric, COSTLY_COSTS)])

    @pytest.fixture
    def calibrated(self, seeded_rng):
        y_score = seeded_rng.beta(2, 3, 20_000)
        return (seeded_rng.random(y_score.size) < y_score).astype(int), y_score

    @staticmethod
    def _baseline_cost(y_true, costs, baseline='zero_one'):
        all_zero, all_one = np.mean(y_true * costs['b']), np.mean((1 - y_true) * costs['a'])
        return {'zero_one': min(all_zero, all_one), 'zero': all_zero, 'one': all_one}[baseline]

    def _threshold(self, y_true, components, baseline='zero_one'):
        """The cost-optimal threshold of the costs weighted by the mixture weight over the baseline's cost."""
        weights = [weight / self._baseline_cost(y_true, costs, baseline) for weight, costs in components]
        fp_cost = sum(weight * costs['a'] for weight, (_, costs) in zip(weights, components, strict=True))
        fn_cost = sum(weight * costs['b'] for weight, (_, costs) in zip(weights, components, strict=True))
        return fp_cost / (fp_cost + fn_cost)

    def test_threshold_weighs_each_component_by_its_baseline(self, mixture, calibrated):
        y_true, y_score = calibrated
        threshold = mixture.optimal_threshold(y_true, y_score)

        assert threshold == pytest.approx(self._threshold(y_true, [(0.5, CHEAP_COSTS), (0.5, COSTLY_COSTS)]))
        weighted_by_the_mixture_weights_alone = (1 + 400) / (1 + 400 + 10 + 100)
        assert mixture(y_true, (y_score >= threshold).astype(float)) > mixture(
            y_true, (y_score >= weighted_by_the_mixture_weights_alone).astype(float)
        )

    @pytest.mark.parametrize('baseline', ['zero', 'one'])
    def test_the_baseline_passed_sets_the_weights(self, mixture, calibrated, baseline):
        y_true, y_score = calibrated
        expected = self._threshold(y_true, [(0.5, CHEAP_COSTS), (0.5, COSTLY_COSTS)], baseline)
        assert mixture.optimal_threshold(y_true, y_score, baseline=baseline) == pytest.approx(expected)

    def test_rate_is_the_share_of_scores_at_or_above_the_threshold(self, mixture, calibrated):
        y_true, y_score = calibrated
        threshold = mixture.optimal_threshold(y_true, y_score)
        assert mixture.optimal_rate(y_true, y_score) == pytest.approx(np.mean(y_score >= threshold))

    def test_instance_dependent_costs_get_a_threshold_per_sample(self, metric, calibrated):
        y_true, y_score = calibrated
        b_values = np.linspace(1.0, 50.0, y_true.size)
        mixture = MixtureMetric([
            MixtureComponent(0.5, metric, {'a': 1.0}),
            MixtureComponent(0.5, metric, {'a': 400.0}),
        ])

        weights = [0.5 / min(np.mean(y_true * b_values), np.mean(1 - y_true) * a) for a in (1.0, 400.0)]
        fp_cost = weights[0] * 1.0 + weights[1] * 400.0
        fn_cost = (weights[0] + weights[1]) * b_values

        np.testing.assert_allclose(
            mixture.optimal_threshold(y_true, y_score, b=b_values), fp_cost / (fp_cost + fn_cost)
        )

    def test_the_decision_needs_the_labels(self, mixture, metric):
        assert Capability.COST_ONLY_DECISION not in mixture.capabilities
        assert Capability.EXPECTED_COST not in mixture.capabilities
        assert {Capability.OPTIMAL_THRESHOLD, Capability.OPTIMAL_RATE} <= mixture.capabilities
        # One component weighs nothing against another, so its decision is the component's own.
        assert Capability.COST_ONLY_DECISION in MixtureMetric([MixtureComponent(1.0, metric, {})]).capabilities

    def test_a_single_component_still_decides_without_labels(self, metric):
        mixture = MixtureMetric([MixtureComponent(1.0, metric, {'a': 1.0})])
        assert mixture.optimal_threshold(np.array([]), np.array([]), b=4.0) == pytest.approx(0.2)
        assert mixture.optimal_rate(np.array([]), np.array([0.1, 0.9]), b=4.0) == pytest.approx(0.5)

    def test_a_baseline_that_costs_nothing_raises(self, mixture):
        with pytest.raises(ValueError, match='the baseline of a Savings component costs nothing'):
            mixture.optimal_threshold(np.zeros(4), np.full(4, 0.5))

    def test_a_nested_mixture_weighs_like_the_flat_one(self, metric, mixture, calibrated):
        y_true, y_score = calibrated
        nested = MixtureMetric([MixtureComponent(0.5, mixture, {}), MixtureComponent(0.5, metric, CHEAP_COSTS)])
        flat = MixtureMetric([
            MixtureComponent(0.75, metric, CHEAP_COSTS),
            MixtureComponent(0.25, metric, COSTLY_COSTS),
        ])
        assert nested.optimal_threshold(y_true, y_score) == pytest.approx(flat.optimal_threshold(y_true, y_score))

    def test_training_objectives_weigh_each_component_by_its_baseline(self, metric, mixture):
        X_raw, y = make_classification(n_samples=200, n_features=4, random_state=0)
        X = _with_intercept(X_raw)
        weights = [0.5 / self._baseline_cost(y, costs) for costs in (CHEAP_COSTS, COSTLY_COSTS)]
        weights = [weight / sum(weights) for weight in weights]

        boost = mixture._prepare_boost_objective(y)
        expected_boost = sum(
            weight * metric._prepare_boost_objective(y, **costs)
            for weight, costs in zip(weights, (CHEAP_COSTS, COSTLY_COSTS), strict=True)
        )
        np.testing.assert_allclose(boost, expected_boost)

        logit_kwargs = {'C': 1.0, 'l1_ratio': 0.0, 'fit_intercept': True}
        coefficients = np.linspace(-0.5, 0.5, X.shape[1])
        loss = mixture._logit_objective(features=X, y_true=y, **logit_kwargs).data_loss(coefficients)
        expected_loss = sum(
            weight * metric._logit_objective(features=X, y_true=y, **logit_kwargs, **costs).data_loss(coefficients)
            for weight, costs in zip(weights, (CHEAP_COSTS, COSTLY_COSTS), strict=True)
        )
        assert loss == pytest.approx(expected_loss)


class TestAliasedOverrides:
    """A component's fixed value wins over the caller's value under any spelling of the symbol."""

    g, v = sympy.symbols('g v')

    @pytest.fixture
    def metric(self):
        return Metric(CostMatrix().add_tp_benefit(self.g).add_fp_cost(self.v).alias('rate', self.g), Cost())

    def test_override_by_symbol_ignores_the_callers_alias(self, metric):
        mixture = MixtureMetric([MixtureComponent(1.0, metric, {'g': 0.0})])
        expected = metric(Y_TRUE, Y_SCORE, g=0.0, v=1.0)
        assert mixture(Y_TRUE, Y_SCORE, rate=0.5, v=1.0) == pytest.approx(expected)

    def test_override_by_alias_is_not_required_from_the_caller(self, metric):
        mixture = MixtureMetric([MixtureComponent(1.0, metric, {'rate': 0.0})])
        assert mixture._missing_parameters({'v'}) == set()
        assert {'g', 'rate'}.isdisjoint(mixture._all_parameters)


def test_components_with_different_parameters_do_not_warn():
    a, b, c = sympy.symbols('a b c')
    first = Metric(CostMatrix().add_fp_cost(a).add_fn_cost(c), Cost())
    second = Metric(CostMatrix().add_fp_cost(b).add_fn_cost(c), Cost())
    mixture = MixtureMetric([MixtureComponent(0.5, first, {}), MixtureComponent(0.5, second, {})])

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        mixture(Y_TRUE, Y_SCORE, a=1.0, b=2.0, c=3.0)
    with pytest.warns(UserWarning, match=r"Unknown parameters passed to metric: \['typo'\]"):
        mixture(Y_TRUE, Y_SCORE, a=1.0, b=2.0, c=3.0, typo=1.0)


@pytest.mark.parametrize('weights', [(0.7, 0.7), (1.5, -0.5)])
def test_constant_weights_must_form_a_mixture(weights):
    metric = Metric(CostMatrix().add_fp_cost('a').add_fn_cost('b'), Cost())
    with pytest.raises(ValueError, match='weights must'):
        MixtureMetric([MixtureComponent(weight, metric, {}) for weight in weights])


class TestWeights:
    """Weights are expressions whose parameters the mixture can see, check and pickle."""

    A = sympy.Symbol('a')
    FIRST = Metric(CostMatrix().add_fp_cost(A).add_fn_cost(1), Cost())
    SECOND = Metric(CostMatrix().add_fp_cost(1).add_fn_cost(A), Cost())

    def mixture(self, first='w', second='1 - w'):
        return MixtureMetric([MixtureComponent(first, self.FIRST, {}), MixtureComponent(second, self.SECOND, {})])

    def test_an_expression_weight_is_the_weighted_sum(self):
        expected = 0.3 * self.FIRST(Y_TRUE, Y_SCORE, a=2.0) + 0.7 * self.SECOND(Y_TRUE, Y_SCORE, a=2.0)
        assert self.mixture()(Y_TRUE, Y_SCORE, a=2.0, w=0.3) == pytest.approx(expected)

    def test_weight_parameters_are_parameters_of_the_mixture(self):
        mixture = self.mixture()
        assert 'w' in mixture._all_parameters
        assert mixture._missing_parameters({'a'}) == {'w'}
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            mixture(Y_TRUE, Y_SCORE, a=2.0, w=0.3)

    def test_a_missing_weight_parameter_is_named(self):
        with pytest.raises(ValueError, match=re.escape("weight parameter(s) ['w']")):
            self.mixture()(Y_TRUE, Y_SCORE, a=2.0)

    @pytest.mark.parametrize(
        ('first', 'second', 'match'),
        [('p', 'q', 'must sum to 1'), ('1.5', '-0.5', 'must not be negative')],
        ids=['not_summing_to_one', 'negative'],
    )
    def test_weights_are_checked_when_called(self, first, second, match):
        parameters = {'p': 0.9, 'q': 0.9}
        if first == 'p':
            mixture = self.mixture(first, second)
        else:
            with pytest.raises(ValueError, match=match):  # constant weights are checked when built
                self.mixture(first, second)
            mixture = self.mixture('w', '1 - w')
            parameters = {'w': 1.5}
            match = 'must not be negative'
        with pytest.raises(ValueError, match=match):
            mixture(Y_TRUE, Y_SCORE, a=2.0, **parameters)

    def test_unvalidated_calls_skip_the_weight_check(self):
        mixture = self.mixture('p', 'q')
        mixture(Y_TRUE, Y_SCORE, validate=False, a=2.0, p=0.9, q=0.9)

    def test_a_callable_weight_is_deprecated_but_still_works(self):
        with pytest.warns(FutureWarning, match='callable MixtureComponent weight is deprecated'):
            mixture = self.mixture(lambda p: p['w'], lambda p: 1 - p['w'])
        assert mixture(Y_TRUE, Y_SCORE, a=2.0, w=0.3) == pytest.approx(self.mixture()(Y_TRUE, Y_SCORE, a=2.0, w=0.3))

    def test_empcs_score_can_be_pickled(self):
        restored = pickle.loads(pickle.dumps(empcs_score))
        assert restored(Y_TRUE, Y_SCORE) == pytest.approx(empcs_score(Y_TRUE, Y_SCORE))


class TestComponentChecks:
    def test_money_and_pure_number_components_are_rejected(self):
        from empulse.metrics import AUEPC

        a = sympy.Symbol('a')
        with pytest.raises(ValueError, match='all be amounts of money, or all pure numbers'):
            MixtureMetric([
                MixtureComponent(0.5, Metric(CostMatrix().add_tp_benefit(a).add_fp_cost(1), AUEPC()), {}),
                MixtureComponent(0.5, Metric(CostMatrix().add_tp_benefit(a).add_fp_cost(1), MaxProfit()), {}),
            ])

    def test_a_parameter_every_component_fixes_is_not_outlier_sensitive(self):
        lam, b = sympy.symbols('lam b')
        metric = Metric(CostMatrix().add_tp_benefit(lam).add_fp_cost(b).mark_outlier_sensitive(lam), Cost())
        mixture = MixtureMetric([
            MixtureComponent(0.5, metric, {'lam': 0.0}),
            MixtureComponent(0.5, metric, {'lam': 1.0}),
        ])
        assert mixture._outlier_sensitive_parameters() == {}


class TestConstrain:
    A = sympy.Symbol('a')

    def mixture(self):
        metric = Metric(CostMatrix().add_fp_cost(self.A).add_fn_cost(1), Cost())
        return MixtureMetric([MixtureComponent('w', metric, {}), MixtureComponent('1 - w', metric, {})])

    def test_lower_above_upper_is_rejected(self):
        with pytest.raises(ValueError, match='greater than upper bound'):
            self.mixture().constrain('a', 5, 1)

    def test_bounding_twice_keeps_the_tighter_bounds(self):
        mixture = self.mixture().constrain('a', 0, 10).constrain('a', 2, 20)
        with pytest.raises(ValueError, match='between 2 and 10'):
            mixture(Y_TRUE, Y_SCORE, a=11.0, w=0.5)

    def test_an_unknown_name_is_rejected(self):
        with pytest.raises(ValueError, match="Cannot constrain 'typo'"):
            self.mixture().constrain('typo', 0, 1)

    def test_a_condition_must_hold_and_survives_pickling(self):
        mixture = self.mixture().constrain('w < 0.9')
        assert isinstance(mixture(Y_TRUE, Y_SCORE, a=1.0, w=0.5), float)
        for candidate in (mixture, pickle.loads(pickle.dumps(mixture))):
            with pytest.raises(ValueError, match=re.escape('w < 0.9')):
                candidate(Y_TRUE, Y_SCORE, a=1.0, w=0.95)

    def test_constraints_see_the_defaults_the_components_share(self):
        """The components score with their default, so the mixture's constraints check it too."""
        metric = Metric(CostMatrix().add_fp_cost(self.A).add_fn_cost('b').set_default(b=2.0), Cost())
        mixture = MixtureMetric([MixtureComponent(0.5, metric, {}), MixtureComponent(0.5, metric, {})])
        with pytest.raises(ValueError, match=re.escape('a > b')):
            mixture.constrain('a > b')(Y_TRUE, Y_SCORE, a=1.0)
        with pytest.raises(ValueError, match='b should be at most 1'):
            mixture.constrain('b', upper=1)(Y_TRUE, Y_SCORE, a=5.0)

    def test_a_default_the_components_disagree_on_is_checked_only_when_passed(self):
        def metric(default):
            return Metric(CostMatrix().add_fp_cost(self.A).add_fn_cost('b').set_default(b=default), Cost())

        mixture = MixtureMetric([MixtureComponent(0.5, metric(2.0), {}), MixtureComponent(0.5, metric(3.0), {})])
        mixture = mixture.constrain('a > b')
        assert isinstance(mixture(Y_TRUE, Y_SCORE, a=1.0), float)
        with pytest.raises(ValueError, match=re.escape('a > b')):
            mixture(Y_TRUE, Y_SCORE, a=1.0, b=2.0)

    def test_a_condition_on_an_unknown_name_is_rejected(self):
        with pytest.raises(ValueError, match=re.escape("Cannot constrain 'typo < 1'")):
            self.mixture().constrain('typo < 1')

    def test_a_random_variable_is_rejected(self):
        """Its values are drawn, not passed, so a bound on it would never be checked."""
        gamma = sympy.stats.Beta('gamma', sympy.Symbol('alpha'), 2)
        metric = Metric(CostMatrix().add_tp_benefit(gamma * sympy.Symbol('clv')).add_fp_cost(1), MaxProfit())
        with pytest.raises(ValueError, match="Cannot constrain 'gamma'"):
            MixtureMetric([MixtureComponent(1.0, metric, {})]).constrain('gamma', 0, 0.1)


class TestAliasedParameters:
    """A mixture resolves a component's aliases before its bounds, predicates and defaults apply."""

    A = sympy.Symbol('a')

    def mixture(self, **kwargs):
        metric = Metric(CostMatrix().add_fp_cost(self.A).add_fn_cost(4).alias({'rate': 'a'}), Cost())
        return MixtureMetric([MixtureComponent(0.5, metric, {}), MixtureComponent(0.5, metric, {})], **kwargs)

    @pytest.mark.parametrize('bounded', ['a', 'rate'])
    @pytest.mark.parametrize('passed', ['a', 'rate'])
    def test_a_bound_holds_under_every_spelling(self, bounded, passed):
        mixture = self.mixture().constrain(bounded, 0, 1)
        with pytest.raises(ValueError, match=f'{passed} should lay between 0 and 1'):
            mixture(Y_TRUE, Y_SCORE, **{passed: 5.0})

    @pytest.mark.parametrize('passed', ['a', 'rate'])
    def test_a_predicate_reads_the_symbol_name(self, passed):
        mixture = self.mixture().constrain(lambda parameters: parameters['a'] <= 1, message='a must be at most 1')
        assert mixture(Y_TRUE, Y_SCORE, **{passed: 0.5}) == pytest.approx(mixture(Y_TRUE, Y_SCORE, a=0.5))
        with pytest.raises(ValueError, match='a must be at most 1'):
            mixture(Y_TRUE, Y_SCORE, **{passed: 5.0})

    @pytest.mark.parametrize('default_under', ['a', 'rate'])
    @pytest.mark.parametrize('passed', ['a', 'rate'])
    def test_a_default_yields_to_a_value_under_another_spelling(self, default_under, passed):
        """The default and the caller's value used to reach the component as two conflicting spellings."""
        mixture = self.mixture(defaults={default_under: 0.5})
        assert mixture(Y_TRUE, Y_SCORE, **{passed: 0.7}) == pytest.approx(self.mixture()(Y_TRUE, Y_SCORE, a=0.7))
