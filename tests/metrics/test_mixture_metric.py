import numpy as np
import pytest
import sympy
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from empulse.metrics import (
    BaseMetric,
    Cost,
    CostMatrix,
    MaxProfit,
    Metric,
    MixtureComponent,
    MixtureMetric,
    classification_threshold,
    empcs_score,
    mpcs_score,
)

from .reference.credit_scoring import empcs, mpcs


@pytest.fixture(scope='module')
def y_true_and_prediction():
    X, y = make_classification(random_state=12)
    lr = LogisticRegression()
    lr.fit(X, y)
    y_proba = lr.predict_proba(X)[:, 1]
    return y, y_proba


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
        MixtureComponent(lambda p: 1 - p['success_rate'] - p['default_rate'], metric_stoch, {}),
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


def test_mixture_missing_weight_parameter_raises(empcs_mixture, y_true_and_prediction):
    y, y_proba = y_true_and_prediction
    with pytest.raises(ValueError, match=r"expected a value for weight parameter 'default_rate'"):
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
    mixture = MixtureMetric([
        MixtureComponent(0.5, max_metric, {}),
        MixtureComponent(0.5, cost_metric, {}),
    ])
    with pytest.raises(ValueError, match='inconsistent optimization directions'):
        _ = mixture.direction


def test_mixture_direction_consistent(empcs_mixture):
    assert empcs_mixture.direction == MaxProfit().direction


def test_mixture_repr(empcs_mixture):
    assert 'MixtureMetric' in repr(empcs_mixture)


def test_mixture_name_property(empcs_mixture):
    assert 'MixtureMetric' in empcs_mixture.__name__


# ---- BaseMetric interface: what makes MixtureMetric usable as a model `loss` ----
# CSLogitClassifier/CSBoostClassifier/CSTreeClassifier/... accept any BaseMetric as their `loss`,
# not just a plain Metric. These tests verify MixtureMetric satisfies that shared interface,
# including the extra methods (`strategy`, `_all_symbols`, `_is_deterministic`, `_evaluate_costs`)
# that model code relies on beyond score/rate/threshold/gradient.


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
    cost_metric = Metric(CostMatrix().add_tp_cost(b), Cost())
    mixture = MixtureMetric([
        MixtureComponent(0.5, max_metric, {}),
        MixtureComponent(0.5, cost_metric, {}),
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
# There is no independent "native EMPCS gradient" to compare against, so these tests check
# that MixtureMetric's combination is *self-consistent*: it must equal manually combining
# each component's own (already-tested-elsewhere) gradient/hessian/loss with the same weights.
#
# These use an all-deterministic 2-component mixture (gamma fixed to 0 and 1), not the full
# 3-component EMPCS mixture with its Uniform(0, 1)-distributed component. That's deliberate:
# MaxProfit's own gradient_boost_objective/logit_objective are currently only implemented for
# deterministic metrics and for stochastic metrics whose distribution derives from
# BasePositiveDistribution (Beta, Gamma, Pareto, etc.) -- Uniform is not one of them, so
# metric_stoch._gradient_boost_objective(...)/._logit_objective(...) themselves raise
# NotImplementedError today, independent of MixtureMetric. That's a pre-existing gap in the
# native MaxProfit machinery, not something introduced by MixtureMetric -- see
# test_full_mixture_gradient_training_not_yet_supported below, which documents it explicitly.


def _two_point_mixture(metric_det, weight_at_0=0.6, weight_at_1=0.4):
    return MixtureMetric([
        MixtureComponent(weight_at_0, metric_det, {'gamma': 0.0}),
        MixtureComponent(weight_at_1, metric_det, {'gamma': 1.0}),
    ])


def _with_intercept(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))


def test_gradient_boost_objective_matches_manual_combination(y_true_and_prediction):
    # alpha_growth=1.0 disables MaxProfit's boosting-epoch alpha annealing: the strategy
    # object shared by both mixture components keeps a running epoch counter across calls,
    # so with annealing left on, two separately-invoked "expected" calls would land on a
    # different epoch (and thus a different smoothing alpha) than the mixture's own calls.
    gamma, roi = sympy.symbols('gamma roi')
    metric_det = Metric(CostMatrix().add_tp_benefit(gamma).add_fp_cost(roi), MaxProfit(alpha_growth=1.0))
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
    # MaxProfit does not implement prepare_boost_objective at all (only Cost/Savings do), so
    # this uses a Cost-strategy mixture instead -- still exercises the same MixtureMetric
    # plumbing, just through a strategy that actually supports the method.
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

    common_kwargs = {'C': 1.0, 'l1_ratio': 0.0, 'soft_threshold': False, 'fit_intercept': True}
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
    common_kwargs = {'C': 1.0, 'l1_ratio': 0.0, 'soft_threshold': False, 'fit_intercept': True}
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
    # alpha_growth=1.0: this calls the objective twice (once directly, once via the
    # steps generator) on purpose, so annealing must be disabled or the second call would
    # legitimately see a different (annealed) alpha than the first -- see the note on
    # test_gradient_boost_objective_matches_manual_combination above.
    gamma, roi_sym = sympy.symbols('gamma roi')
    metric_det = Metric(CostMatrix().add_tp_benefit(gamma).add_fp_cost(roi_sym), MaxProfit(alpha_growth=1.0))
    X_raw, y = make_classification(n_samples=40, n_features=3, n_informative=2, n_redundant=0, random_state=5)
    X = _with_intercept(X_raw)
    roi = 0.2644
    mixture = _two_point_mixture(metric_det, 0.6, 0.4)
    common_kwargs = {'C': 1.0, 'l1_ratio': 0.0, 'soft_threshold': False, 'fit_intercept': True}
    objective = mixture._logit_objective(features=X, y_true=y, roi=roi, **common_kwargs)

    weights = np.zeros(X.shape[1], dtype=np.float64)
    direct_grad = objective.logit_gradient(weights)

    steps = objective.logit_gradient_steps()
    step_grad = steps.send(weights)
    steps.close()

    np.testing.assert_allclose(step_grad, direct_grad)


def test_full_mixture_gradient_training_not_yet_supported(credit_scoring_metrics, y_true_and_prediction):
    """Documents a pre-existing native limitation, not a MixtureMetric bug.

    MaxProfit's gradient_boost_objective/logit_objective only support deterministic metrics
    and stochastic metrics built on a BasePositiveDistribution (Beta, Gamma, Pareto, ...).
    Uniform is not one of those, so the EMPCS mixture's third (Uniform) component cannot be
    used for gradient-based training today. MixtureMetric correctly propagates that
    NotImplementedError rather than silently producing a wrong gradient.
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


# ---- regression test for the ExactMaxProfitRatePiecewise hardcoded-literal-bounds bug ----


@pytest.mark.parametrize('integration_method', ['auto', 'quad'])
def test_hardcoded_uniform_bounds_optimal_rate_no_error(y_true_and_prediction, integration_method):
    """Previously crashed with TypeError: _uniform_params() missing 2 required positional
    arguments 'a' and 'b' when the Uniform distribution's bounds were hardcoded literals
    (no free symbols) and .optimal_rate() (not just __call__) was used.
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
