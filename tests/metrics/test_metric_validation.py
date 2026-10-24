"""
Misuse / defensive-guard tests for the Metric core.

These guard against ways Metric can be misused that don't raise an error but instead silently compute the wrong number.
"""

import warnings

import numpy as np
import pytest
import sympy

import empulse.metrics as empulse_metrics
from empulse.metrics import (
    BaseMetric,
    Cost,
    CostMatrix,
    MaxProfit,
    Metric,
    MixtureComponent,
    MixtureMetric,
    Savings,
    cost_loss,
    expected_cost_loss,
)

Y_TRUE = np.array([1, 0, 1, 0, 1])
Y_SCORE = np.array([0.9, 0.1, 0.8, 0.2, 0.7])


# --- reusing one MetricStrategy instance across two Metrics -------------------------


def test_metric_owns_an_independent_strategy_copy():
    """`Metric` must not share mutable strategy state with the object passed to it."""
    strategy = Cost()
    metric = Metric(CostMatrix().add_fp_cost('a'), strategy)

    assert metric.strategy is not strategy


def test_reusing_one_strategy_instance_keeps_metrics_independent():
    """
    Building a second Metric from the same strategy instance must not corrupt the first.

    `Metric.__init__` must not build the strategy object in place, or the second `Metric`'s `build()`
    call would overwrite the first metric's compiled formula.
    """
    strategy = Cost()
    metric_fp = Metric(CostMatrix().add_fp_cost('a'), strategy)
    # The second Metric shares the strategy instance and an overlapping symbol name, and must not
    # rewrite metric_fp's formula.
    metric_fn = Metric(CostMatrix().add_fn_cost('a'), strategy)

    expected_fp = Metric(CostMatrix().add_fp_cost('a'), Cost())(Y_TRUE, Y_SCORE, a=100.0)
    expected_fn = Metric(CostMatrix().add_fn_cost('a'), Cost())(Y_TRUE, Y_SCORE, a=100.0)

    assert metric_fp(Y_TRUE, Y_SCORE, a=100.0) == pytest.approx(expected_fp)
    assert metric_fn(Y_TRUE, Y_SCORE, a=100.0) == pytest.approx(expected_fn)
    assert expected_fp != expected_fn  # sanity check the two formulas actually differ


def test_reusing_one_strategy_instance_with_disjoint_symbols_still_works():
    """Same scenario as above but with non-overlapping symbol names, for good measure."""
    strategy = Cost()
    metric_a = Metric(CostMatrix().add_fp_cost('a'), strategy)
    metric_b = Metric(CostMatrix().add_fn_cost('b'), strategy)

    assert metric_a(Y_TRUE, Y_SCORE, a=100.0) == pytest.approx(
        Metric(CostMatrix().add_fp_cost('a'), Cost())(Y_TRUE, Y_SCORE, a=100.0)
    )
    assert metric_b(Y_TRUE, Y_SCORE, b=100.0) == pytest.approx(
        Metric(CostMatrix().add_fn_cost('b'), Cost())(Y_TRUE, Y_SCORE, b=100.0)
    )


# --- mutating a CostMatrix after building a Metric ----------------------------------


def test_metric_owns_an_independent_cost_matrix_copy():
    """`Metric` must not share the mutable `CostMatrix` object passed to it."""
    cost_matrix = CostMatrix().add_fp_cost('a')
    metric = Metric(cost_matrix, Cost())

    assert metric.cost_matrix is not cost_matrix


def test_mutating_source_cost_matrix_does_not_affect_the_metric():
    """
    A `Metric` is a snapshot: later mutation of the `CostMatrix` it was built from must not
    change either its computed score or its advertised cost expressions.
    """
    cost_matrix = CostMatrix().add_fp_cost('a')
    metric = Metric(cost_matrix, Cost())

    score_before = metric(Y_TRUE, Y_SCORE, a=100.0)
    fn_cost_before = metric.fn_cost

    cost_matrix.add_fn_cost('b')  # mutate the source matrix after the metric was built

    assert metric.fn_cost == fn_cost_before
    assert metric.fn_cost == sympy.Integer(0)
    assert metric(Y_TRUE, Y_SCORE, a=100.0) == pytest.approx(score_before)


def test_mutating_source_cost_matrix_new_parameter_is_unknown_to_the_metric():
    """
    A parameter added to the source matrix after Metric construction must not be silently
    accepted - it should warn as an unknown parameter, since the metric never learned about it.
    """
    cost_matrix = CostMatrix().add_fp_cost('a')
    metric = Metric(cost_matrix, Cost())
    cost_matrix.add_fn_cost('b')

    with pytest.warns(UserWarning, match=r"Unknown parameters passed to metric: \['b'\]"):
        metric(Y_TRUE, Y_SCORE, a=100.0, b=100.0)


# --- Metric performs no validation of y_true / y_score (regression vs. cost_loss) ----


COST_MATRIX = CostMatrix().add_fp_cost('a').add_fn_cost('b')


@pytest.mark.parametrize(
    'y_true',
    [
        np.array([1, 2, 1, 2, 1]),
        np.array([0, 1, 0, 1, 2]),
    ],
)
def test_non_binary_y_true_raises(y_true):
    """A non-binary y_true must be rejected, matching cost_loss()'s existing behaviour."""
    metric = Metric(COST_MATRIX, Cost())

    with pytest.raises(ValueError, match='should be binary'):
        metric(y_true, Y_SCORE, a=1.0, b=1.0)


def test_boolean_y_true_is_accepted():
    y_score = np.array([0.2, 0.7, 0.4, 0.9])
    y_bool = np.array([False, True, False, True])
    expected = expected_cost_loss(y_bool.astype(int), y_score, fp_cost=1.0, fn_cost=2.0)
    assert expected_cost_loss(y_bool, y_score, fp_cost=1.0, fn_cost=2.0) == pytest.approx(expected)


def test_single_class_y_true_is_accepted():
    """
    A y_true with only one distinct value must be accepted, not rejected.

    Unlike cost_loss()/lift_score(), Metric deliberately does not require variance in y_true: a
    single-class fold or sample-weighted subset is a legitimate input here (e.g. cross-validation,
    or the `check_classifiers_one_label_sample_weights` sklearn compliance check), and rejecting it
    would be needless friction rather than catching a real mistake.
    """
    metric = Metric(COST_MATRIX, Cost())

    score = metric(np.array([1, 1, 1, 1, 1]), Y_SCORE, a=1.0, b=1.0)

    assert isinstance(score, float)


def test_nan_in_y_true_raises():
    metric = Metric(COST_MATRIX, Cost())

    with pytest.raises(ValueError, match='NaN'):
        metric(np.array([1.0, 0.0, np.nan, 0.0, 1.0]), Y_SCORE, a=1.0, b=1.0)


def test_nan_in_y_score_raises():
    metric = Metric(COST_MATRIX, Cost())

    with pytest.raises(ValueError, match='NaN'):
        metric(Y_TRUE, np.array([0.9, 0.1, np.nan, 0.2, 0.7]), a=1.0, b=1.0)


def test_inf_in_y_score_raises():
    metric = Metric(COST_MATRIX, Cost())

    with pytest.raises(ValueError, match='Inf'):
        metric(Y_TRUE, np.array([0.9, 0.1, np.inf, 0.2, 0.7]), a=1.0, b=1.0)


def test_expected_cost_loss_and_cost_loss_raise_the_same_error_on_non_binary_labels():
    """expected_cost_loss (a Metric) and cost_loss (a function) must reject bad input alike."""
    y_bad = np.array([1, 2, 1, 2, 1])

    with pytest.raises(ValueError, match='should be binary') as legacy_exc_info:
        cost_loss(y_bad, Y_SCORE, fp_cost=1.0, fn_cost=1.0)
    with pytest.raises(ValueError, match='should be binary') as metric_exc_info:
        expected_cost_loss(y_bad, Y_SCORE, fp_cost=1.0, fn_cost=1.0)

    assert str(legacy_exc_info.value) == str(metric_exc_info.value)


def test_optimal_threshold_still_accepts_empty_y_true_and_y_score():
    """The documented empty-placeholder path (predict-time threshold, no labels available) must still work."""
    metric = Metric(COST_MATRIX, Cost())

    threshold = metric.optimal_threshold(np.array([]), np.array([]), a=1.0, b=1.0)

    assert threshold == pytest.approx(0.5)


def test_optimal_rate_still_accepts_empty_y_true_with_real_y_score():
    """The documented empty-y_true path (rate needed, but no labels available yet) must still work."""
    metric = Metric(COST_MATRIX, Cost())

    rate = metric.optimal_rate(np.array([]), Y_SCORE, a=1.0, b=1.0)

    assert 0.0 <= rate <= 1.0


def test_optimal_threshold_validates_non_empty_y_true():
    """When y_true is actually provided (non-empty), it must still be validated like everywhere else."""
    metric = Metric(COST_MATRIX, Cost())

    with pytest.raises(ValueError, match='should be binary'):
        metric.optimal_threshold(np.array([1, 2, 1, 2, 1]), Y_SCORE, a=1.0, b=1.0)


def test_call_accepts_a_row_vector_y_score():
    """A shape (1, n_samples) y_score (as catboost's internal metric callback passes) must still work."""
    metric = Metric(COST_MATRIX, Cost())

    score = metric(Y_TRUE, Y_SCORE.reshape(1, -1), a=1.0, b=1.0)

    assert score == pytest.approx(metric(Y_TRUE, Y_SCORE, a=1.0, b=1.0))


# --- wrong-length instance-dependent parameters ---------------------------------------


@pytest.mark.parametrize('value', [np.inf, -np.inf, np.nan, np.array([1.0, 1.0, np.nan, 1.0, 1.0])])
def test_non_finite_parameter_raises(value):
    metric = Metric(COST_MATRIX, Cost())

    with pytest.raises(ValueError, match='a must be finite'):
        metric(Y_TRUE, Y_SCORE, a=value, b=1.0)


def test_non_finite_parameter_error_names_the_alias_the_caller_used():
    cost_matrix = CostMatrix().add_fp_cost('a').add_fn_cost('b').alias({'my_cost': 'a'})
    metric = Metric(cost_matrix, Cost())

    with pytest.raises(ValueError, match='my_cost must be finite'):
        metric(Y_TRUE, Y_SCORE, my_cost=np.inf, b=1.0)


def test_wrong_length_array_parameter_raises():
    metric = Metric(COST_MATRIX, Cost())

    with pytest.raises(ValueError, match=r"Parameter 'a' has length 3, but expected length 5"):
        metric(Y_TRUE, Y_SCORE, a=np.array([1.0, 2.0, 3.0]), b=1.0)


def test_wrong_length_array_parameter_error_names_the_alias_the_caller_used():
    """The error should name what the caller actually typed, not the resolved symbol name."""
    cost_matrix = CostMatrix().add_fp_cost('a').add_fn_cost('b').alias({'my_cost': 'a'})
    metric = Metric(cost_matrix, Cost())

    with pytest.raises(ValueError, match=r"Parameter 'my_cost' has length 2"):
        metric(Y_TRUE, Y_SCORE, my_cost=np.array([1.0, 2.0]), b=1.0)


def test_size_one_array_parameter_is_accepted_as_scalar_broadcast():
    """A length-1 array must be accepted as a scalar-like broadcast, not rejected as 'wrong length'."""
    metric = Metric(COST_MATRIX, Cost())

    broadcast = metric(Y_TRUE, Y_SCORE, a=np.array([2.0]), b=1.0)
    scalar = metric(Y_TRUE, Y_SCORE, a=2.0, b=1.0)

    assert broadcast == pytest.approx(scalar)


def test_correct_length_array_parameter_is_accepted():
    metric = Metric(COST_MATRIX, Cost())

    score = metric(Y_TRUE, Y_SCORE, a=np.full(Y_TRUE.shape, 2.0), b=1.0)

    assert score == pytest.approx(metric(Y_TRUE, Y_SCORE, a=2.0, b=1.0))


def test_optimal_threshold_also_validates_parameter_length():
    metric = Metric(COST_MATRIX, Cost())

    with pytest.raises(ValueError, match=r"Parameter 'a' has length 2, but expected length 5"):
        metric.optimal_threshold(Y_TRUE, Y_SCORE, a=np.array([1.0, 2.0]), b=1.0)


def test_optimal_threshold_predict_time_empty_arrays_skip_length_check():
    """The documented empty-placeholder path (both y_true and y_score empty) must not spuriously
    reject scalar-reduced loss params - there's no sample count to check a length against."""
    metric = Metric(COST_MATRIX, Cost())

    threshold = metric.optimal_threshold(np.array([]), np.array([]), a=1.0, b=1.0)

    assert threshold == pytest.approx(0.5)


# --- optimal_threshold/optimal_rate out of [0, 1] or degenerate cost matrices --------


@pytest.mark.parametrize(
    ('a', 'b', 'expected'),
    [(1.0, -1.0, np.inf), (-1.0, 1.0, 0.0)],
    ids=['targeting_always_costs_more', 'targeting_always_costs_less'],
)
def test_costs_that_do_not_depend_on_the_probability_decide_always_or_never(a, b, expected):
    """Where the threshold's denominator is 0, the cheaper decision holds for every probability."""
    metric = Metric(COST_MATRIX, Cost())

    with pytest.warns(UserWarning, match='every sample or none'):
        assert metric.optimal_threshold(Y_TRUE, Y_SCORE, a=a, b=b) == expected
    with pytest.warns(UserWarning, match='every sample or none'):
        assert metric.optimal_rate(Y_TRUE, Y_SCORE, a=a, b=b) == np.mean(expected <= Y_SCORE)


def test_a_sample_whose_decisions_cost_the_same_is_never_targeted_without_a_warning():
    metric = Metric(COST_MATRIX, Cost())
    a = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    b = np.array([4.0, 0.0, 4.0, 0.0, 4.0])

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        thresholds = metric.optimal_threshold(Y_TRUE, Y_SCORE, a=a, b=b)

    np.testing.assert_allclose(thresholds, [0.2, np.inf, 0.2, np.inf, 0.2])


@pytest.mark.parametrize(
    ('a', 'b', 'expected'),
    [(1.0, -3.0, np.inf), (-1.0, 3.0, 0.0), (3.0, -1.0, np.inf), (-3.0, 1.0, 0.0)],
    ids=['negative_ratio_below_zero', 'positive_ratio_below_zero', 'ratio_above_one', 'negative_ratio_above_one'],
)
def test_threshold_outside_the_probability_range_targets_everyone_or_nobody(a, b, expected):
    """
    A ratio outside [0, 1] decides every sample alike, by the sign of the denominator: clipping it
    instead targets everyone where targeting nobody is optimal when the denominator is negative.
    """
    metric = Metric(COST_MATRIX, Cost())

    with pytest.warns(UserWarning, match='every sample or none'):
        threshold = metric.optimal_threshold(Y_TRUE, Y_SCORE, a=a, b=b)

    assert threshold == expected


@pytest.mark.parametrize('a', [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
@pytest.mark.parametrize('b', [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
def test_optimal_threshold_targets_exactly_where_targeting_costs_less(a, b):
    """Targeting the probabilities at or above the threshold never costs more than the other decision."""
    metric = Metric(COST_MATRIX, Cost())
    probability = np.linspace(0.0, 1.0, 101)
    cost_of_targeting = (1 - probability) * a
    cost_of_not_targeting = probability * b

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        try:
            threshold = metric.optimal_threshold(Y_TRUE, Y_SCORE, a=a, b=b)
        except ValueError:
            # Raised only when no threshold is optimal: targeting strictly pays off at p=0 and strictly
            # loses at p=1, a decreasing rule.
            assert cost_of_targeting[0] < cost_of_not_targeting[0]
            assert cost_of_targeting[-1] > cost_of_not_targeting[-1]
            return

    cost_of_decision = np.where(probability >= threshold, cost_of_targeting, cost_of_not_targeting)
    best_cost = np.minimum(cost_of_targeting, cost_of_not_targeting)
    np.testing.assert_allclose(cost_of_decision, best_cost, atol=1e-12)


def test_costs_tied_at_a_probability_of_one_target_everyone_instead_of_raising():
    """With a negative denominator and a ratio of exactly 1, positives cost the same either way, so target all."""
    metric = Metric(COST_MATRIX, Cost())

    with pytest.warns(UserWarning, match='every sample or none'):
        assert metric.optimal_threshold(Y_TRUE, Y_SCORE, a=-1.0, b=0.0) == 0.0


def test_costs_favouring_the_least_likely_positives_raise():
    """No threshold on the probability is optimal when targeting pays off for the least likely positives."""
    metric = Metric(COST_MATRIX, Cost())

    with pytest.raises(ValueError, match='less likely it is to be positive'):
        metric.optimal_threshold(Y_TRUE, Y_SCORE, a=-1.0, b=-1.0)
    with pytest.raises(ValueError, match='less likely it is to be positive'):
        metric.optimal_rate(Y_TRUE, Y_SCORE, a=-1.0, b=-1.0)


def test_targeting_nobody_excludes_scores_of_exactly_one():
    """The threshold for targeting nobody is not reached by a probability of exactly 1."""
    metric = Metric(COST_MATRIX, Cost())
    y_score = np.array([1.0, 1.0, 0.5, 0.0, 1.0])

    with pytest.warns(UserWarning, match='every sample or none'):
        assert metric.optimal_rate(Y_TRUE, y_score, a=3.0, b=-1.0) == 0.0


def test_in_range_threshold_does_not_warn():
    metric = Metric(COST_MATRIX, Cost())

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        threshold = metric.optimal_threshold(Y_TRUE, Y_SCORE, a=1.0, b=1.0)

    assert 0.0 <= threshold <= 1.0


# --- sample_weight: the cost matrix weighs each sample ------------------------------------


@pytest.mark.parametrize('method', ['__call__', 'optimal_threshold', 'optimal_rate'])
@pytest.mark.parametrize('strategy', [Cost, MaxProfit], ids=['cost', 'max_profit'])
def test_metrics_reject_sample_weight(method, strategy):
    """A sample weight would weigh the samples a second time, on top of their costs."""
    metric = Metric(COST_MATRIX, strategy())

    with pytest.raises(TypeError, match='Metrics do not take sample_weight'):
        getattr(metric, method)(Y_TRUE, Y_SCORE, a=1.0, b=1.0, sample_weight=np.ones(5))


def test_mixture_metric_rejects_sample_weight():
    mixture = MixtureMetric([
        MixtureComponent(0.5, Metric(COST_MATRIX, Cost()), {}),
        MixtureComponent(0.5, Metric(CostMatrix().add_fp_cost('a').add_fn_cost('2 * b'), Cost()), {}),
    ])

    with pytest.raises(TypeError, match='Metrics do not take sample_weight'):
        mixture(Y_TRUE, Y_SCORE, a=1.0, b=1.0, sample_weight=np.ones(5))


def test_a_cost_matrix_parameter_named_sample_weight_is_accepted():
    """The name is only reserved for scikit-learn's sample weights, not for a cost the matrix defines."""
    metric = Metric(CostMatrix().add_fp_cost('sample_weight').add_fn_cost('b'), Cost())

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        score = metric(Y_TRUE, Y_SCORE, sample_weight=2.0, b=1.0)

    assert score == pytest.approx(metric(Y_TRUE, Y_SCORE, sample_weight=np.full(5, 2.0), b=1.0))


def test_a_scorer_requesting_sample_weight_raises():
    """Routing sample weights to an empulse scorer is refused rather than silently ignored."""
    from sklearn import config_context
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import cross_validate

    X = np.random.default_rng(0).normal(size=(40, 3))
    y = np.tile([0, 1], 20)
    metric = Metric(COST_MATRIX, Cost())
    with config_context(enable_metadata_routing=True):
        scorer = make_scorer(metric, response_method='predict_proba', a=1.0, b=1.0)
        scorer.set_score_request(sample_weight=True)
        estimator = LogisticRegression().set_fit_request(sample_weight=False)
        with pytest.raises(TypeError, match='Metrics do not take sample_weight'):
            cross_validate(estimator, X, y, scoring=scorer, params={'sample_weight': np.ones(40)}, error_score='raise')


def test_symbol_used_only_in_the_threshold_numerator_is_still_required():
    """
    A symbol that cancels out of the denominator (fp_cost + tn_benefit + fn_cost + tp_benefit) but
    survives in the numerator (fp_cost + tn_benefit) must still be caught as missing by
    _check_parameters - not left to surface as a cryptic internal TypeError.
    """
    a = sympy.Symbol('a')
    # numerator = fp_cost + tn_benefit = a; denominator = a + 0 + (1 - a) + 0 = 1 (a cancels out).
    cost_matrix = CostMatrix().add_fp_cost(a).add_fn_cost(1 - a)
    metric = Metric(cost_matrix, Cost())

    with pytest.raises(ValueError, match='expected a value for a'):
        metric.optimal_threshold(Y_TRUE, Y_SCORE)


# --- every prebuilt metric reports its own name, not its strategy's --------------------


@pytest.mark.parametrize('attr_name', sorted(empulse_metrics.__all__))
def test_every_public_metric_name_matches_its_attribute_name(attr_name):
    """
    Every public `BaseMetric` instance in `empulse.metrics` must report its own name via
    `__name__`, not e.g. the shared strategy name ('max profit') or (for a MixtureMetric) an
    auto-composed name built from its components. This is what makes make_scorer(...) and
    cross-validation results distinguish two different metrics that happen to share a strategy.
    """
    obj = getattr(empulse_metrics, attr_name)
    if not isinstance(obj, BaseMetric):
        pytest.skip(f'{attr_name} is not a BaseMetric instance')

    assert obj.__name__ == attr_name


def test_metric_name_is_settable():
    metric = Metric(CostMatrix().add_fp_cost('a'), Cost())

    metric.__name__ = 'my_custom_name'

    assert metric.__name__ == 'my_custom_name'


def test_mixture_metric_name_defaults_to_a_composed_name():
    """Sanity check: without an explicit override, MixtureMetric composes a name from its parts."""
    component_metric = Metric(CostMatrix().add_fp_cost('a'), Cost())
    mixture = MixtureMetric([MixtureComponent(1.0, component_metric, {})])

    assert mixture.__name__ == f'MixtureMetric({component_metric.__name__})'


def test_mixture_metric_name_is_settable():
    """A MixtureMetric (e.g. empcs_score) must be able to report a clean name too, not just Metric."""
    component_metric = Metric(CostMatrix().add_fp_cost('a'), Cost())
    mixture = MixtureMetric([MixtureComponent(1.0, component_metric, {})])

    mixture.__name__ = 'my_mixture_metric'

    assert mixture.__name__ == 'my_mixture_metric'


# --- MaxProfit averages instance-dependent parameters (documented, not changed) --------


def test_max_profit_averages_array_like_parameters():
    """
    Pin MaxProfit's documented behaviour: an array-like parameter is reduced to its mean, so it gives
    the exact same result as passing that mean directly - unlike Cost, which genuinely uses
    per-instance values.
    """
    cost_matrix = CostMatrix().add_tp_benefit('a').add_fp_cost('b')
    max_profit_metric = Metric(cost_matrix, MaxProfit())
    cost_metric = Metric(cost_matrix, Cost())

    array_value = np.array([0.0, 0.0, 0.0, 0.0, 500.0])

    max_profit_from_array = max_profit_metric(Y_TRUE, Y_SCORE, a=array_value, b=1.0)
    max_profit_from_mean = max_profit_metric(Y_TRUE, Y_SCORE, a=float(array_value.mean()), b=1.0)
    assert max_profit_from_array == pytest.approx(max_profit_from_mean)

    # Contrast with Cost, which is genuinely instance-dependent and would not collapse like this.
    cost_from_array = cost_metric(Y_TRUE, Y_SCORE, a=array_value, b=1.0)
    cost_from_mean = cost_metric(Y_TRUE, Y_SCORE, a=float(array_value.mean()), b=1.0)
    assert cost_from_array != pytest.approx(cost_from_mean)


# --- Savings reduces stochastic variables to their mean, matching Cost ------------------


def test_savings_accepts_a_stochastic_cost_matrix():
    """Savings accepts a stochastic cost matrix."""
    gamma = sympy.stats.Beta('gamma', sympy.Symbol('alpha'), sympy.Symbol('beta'))
    cost_matrix = CostMatrix().add_tp_benefit(gamma * sympy.Symbol('clv')).add_fp_cost('b')
    metric = Metric(cost_matrix, Savings())

    score = metric(Y_TRUE, Y_SCORE, clv=100.0, b=1.0, alpha=6.0, beta=14.0)

    assert isinstance(score, float)


def test_savings_stochastic_result_matches_mean_substitution():
    """Savings on a stochastic cost matrix must give the exact same result as substituting the
    random variable's mean directly - matching Cost's existing (unchanged) behaviour."""
    gamma = sympy.stats.Beta('gamma', 6, 14)  # numeric alpha/beta: E[gamma] is a plain number
    stochastic_matrix = CostMatrix().add_tp_benefit(gamma * sympy.Symbol('clv')).add_fp_cost('b')
    stochastic_metric = Metric(stochastic_matrix, Savings())

    mean_gamma = float(sympy.stats.E(gamma))
    deterministic_matrix = CostMatrix().add_tp_benefit(mean_gamma * sympy.Symbol('clv')).add_fp_cost('b')
    deterministic_metric = Metric(deterministic_matrix, Savings())

    stochastic_result = stochastic_metric(Y_TRUE, Y_SCORE, clv=100.0, b=1.0)
    deterministic_result = deterministic_metric(Y_TRUE, Y_SCORE, clv=100.0, b=1.0)

    assert stochastic_result == pytest.approx(deterministic_result)


# --- CostMatrix.constrain() ---------------------------------------------------------------------
