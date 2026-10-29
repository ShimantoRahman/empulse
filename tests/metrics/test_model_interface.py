"""ModelInterface, the public view of a metric that models train and decide with."""

import copy
import pickle

import numpy as np
import pytest

from empulse.metrics import (
    AUEPC,
    Capability,
    Cost,
    CostMatrix,
    EmpiricalMaxProfit,
    LogCost,
    MaxProfit,
    Metric,
    MinCost,
    MixtureComponent,
    MixtureMetric,
    ModelInterface,
    Profit,
    Savings,
    empc_score,
)


def _cost_metric(strategy=None):
    return Metric(CostMatrix().add_fp_cost('fp').add_fn_cost('fn'), strategy or Cost())


def _mixture():
    return MixtureMetric([
        MixtureComponent(0.3, _cost_metric(Profit()), {}),
        MixtureComponent(0.7, _cost_metric(Savings()), {}),
    ])


METRICS = {
    'metric': (_cost_metric, {'fp': 1.0, 'fn': 'per-sample'}),
    'mixture': (_mixture, {'fp': 1.0, 'fn': 'per-sample'}),
    'prebuilt': (lambda: empc_score, {'clv': 'per-sample', 'd': 10.0, 'f': 1.0, 'alpha': 6.0, 'beta': 14.0}),
}


@pytest.fixture
def data(seeded_rng):
    y_true = seeded_rng.integers(0, 2, 200)
    y_true[:2] = [0, 1]
    y_score = seeded_rng.random(200)
    per_sample = seeded_rng.uniform(5, 20, 200)
    return y_true, y_score, per_sample


@pytest.fixture(params=list(METRICS))
def metric_and_params(request, data):
    make, template = METRICS[request.param]
    _, _, per_sample = data
    return make(), {name: per_sample if value == 'per-sample' else value for name, value in template.items()}


def test_model_is_a_model_interface(metric_and_params):
    metric, _ = metric_and_params
    assert isinstance(metric.model, ModelInterface)
    assert repr(metric) in repr(metric.model)


def test_parameters_match_the_metric(metric_and_params):
    metric, params = metric_and_params
    interface = metric.model
    assert metric.parameter_names <= interface.accepted_parameters
    assert interface.missing_parameters(params) == set()
    interface.validate_parameters(**params)
    for name in metric.parameter_names:
        assert name in interface.parameter_spellings(name)


def test_values_match_the_hooks(metric_and_params, data):
    metric, params = metric_and_params
    y_true, y_score, _ = data
    interface = metric.model
    assert interface.loss(y_true, y_score, **params) == metric._loss(y_true, y_score, **params)
    assert interface.cost_weight(y_true, **params) == metric._cost_weight(y_true, **params)
    for public, private in zip(
        interface.decision_costs(y_true, **params), metric._decision_costs(y_true, **params), strict=True
    ):
        np.testing.assert_array_equal(public, private)
    assert interface.missing_parameters(()) == metric._missing_parameters(())
    assert interface.is_deterministic == metric._is_deterministic
    assert interface.decides_from_costs == metric._decides_from_costs


def test_loss_is_minimized_whatever_the_direction(data):
    y_true, y_score, fn = data
    for minimized, maximized in ((Cost(), Profit()), (MinCost(), MaxProfit())):
        assert _cost_metric(minimized).model.loss(y_true, y_score, fp=1.0, fn=fn) == pytest.approx(
            _cost_metric(maximized).model.loss(y_true, y_score, fp=1.0, fn=fn)
        )


def test_evaluate_costs_replaces_stochastic_variables_by_their_mean():
    fp_cost, _, tp_cost, _ = empc_score.model.evaluate_costs(
        replace_stochastic=True, clv=100.0, d=10.0, f=1.0, alpha=6.0, beta=14.0
    )
    gamma = 6.0 / (6.0 + 14.0)
    assert fp_cost == pytest.approx(11.0)
    assert tp_cost == pytest.approx(-(gamma * (100.0 - 10.0 - 1.0) - (1 - gamma) * 1.0))


def test_decision_cost_scale_follows_the_cost_units(data):
    y_true, _, fn = data
    interface = _cost_metric().model
    scale = interface.decision_cost_scale(y_true, fp=1.0, fn=fn)
    assert interface.decision_cost_scale(y_true, fp=64.0, fn=64 * fn) == pytest.approx(64 * scale)
    assert interface.loss_scale(y_true, fp=1.0, fn=fn) == pytest.approx(scale)


def test_decision_cost_scale_is_one_for_a_unitless_metric(data):
    y_true, _, fn = data
    interface = _cost_metric(AUEPC()).model
    assert interface.is_unitless
    assert interface.decision_cost_scale(y_true, fp=1.0, fn=fn) == 1.0


def test_require_names_the_requester():
    interface = _cost_metric(MaxProfit()).model
    interface.require(Capability.OPTIMAL_THRESHOLD, requester='Mine')
    with pytest.raises(ValueError, match=r"^Mine only supports .*'class_costs'.* Try this\.$"):
        _cost_metric(LogCost()).model.require(Capability.CLASS_COSTS, requester='Mine', hint='Try this.')


@pytest.mark.parametrize('round_trip', [copy.deepcopy, lambda metric: pickle.loads(pickle.dumps(metric))])
def test_model_survives_copying(round_trip, data):
    y_true, y_score, fn = data
    metric = _cost_metric()
    restored = round_trip(metric)
    assert restored.model.loss(y_true, y_score, fp=1.0, fn=fn) == metric.model.loss(y_true, y_score, fp=1.0, fn=fn)


def _realised_cost(interface, y_true, predicted_positive, **params):
    fp_cost, fn_cost, tp_cost, tn_cost = (
        np.broadcast_to(cost, y_true.shape) for cost in interface.decision_costs(y_true, **params)
    )
    positive = y_true == 1
    return float(
        np.sum(np.where(predicted_positive, np.where(positive, tp_cost, fp_cost), np.where(positive, fn_cost, tn_cost)))
    )


def test_decision_threshold_costs_least_of_every_cut(metric_and_params, data):
    metric, params = metric_and_params
    y_true, y_score, _ = data
    interface = metric.model
    threshold = interface.decision_threshold(y_true, y_score, **params)
    best = _realised_cost(interface, y_true, y_score > threshold, **params)
    for cut in [-np.inf, *np.unique(y_score)]:
        assert best <= _realised_cost(interface, y_true, y_score > cut, **params) + 1e-9


def test_decision_threshold_targets_what_the_empirical_max_profit_targets(data):
    y_true, y_score, fn = data
    cost_matrix = CostMatrix().add_fp_cost('fp').add_fn_cost('fn')
    threshold = Metric(cost_matrix, Cost()).model.decision_threshold(y_true, y_score, fp=1.0, fn=fn)
    rate = Metric(cost_matrix, EmpiricalMaxProfit()).optimal_rate(y_true, y_score, fp=1.0, fn=fn)
    assert np.mean(y_score > threshold) == pytest.approx(rate)


def test_decision_threshold_lies_between_the_scores_it_separates(data):
    y_true, y_score, fn = data
    threshold = _cost_metric().model.decision_threshold(y_true, y_score, fp=12.0, fn=fn)
    assert np.isfinite(threshold)
    above, below = y_score[y_score > threshold].min(), y_score[y_score <= threshold].max()
    assert threshold == pytest.approx((above + below) / 2)


def test_decision_threshold_separates_adjacent_floats():
    y_true = np.array([1, 0])
    y_score = np.array([0.5, np.nextafter(0.5, 0)])
    threshold = _cost_metric().model.decision_threshold(y_true, y_score, fp=1.0, fn=1.0)
    np.testing.assert_array_equal(y_score > threshold, [True, False])


@pytest.mark.parametrize(('fp', 'fn', 'expected'), [(10.0, 0.1, np.inf), (0.1, 10.0, -np.inf)])
def test_decision_threshold_predicts_every_sample_alike_when_that_costs_least(data, fp, fn, expected):
    y_true, y_score, _ = data
    assert _cost_metric().model.decision_threshold(y_true, y_score, fp=fp, fn=fn) == expected


def test_decision_threshold_gives_tied_scores_one_decision():
    y_true = np.array([1, 1, 0, 1])
    y_score = np.array([0.9, 0.4, 0.4, 0.1])
    threshold = _cost_metric().model.decision_threshold(y_true, y_score, fp=1.0, fn=5.0)
    predicted = y_score > threshold
    assert predicted[1] == predicted[2]


def test_decision_threshold_does_not_depend_on_the_cost_units(data):
    y_true, y_score, fn = data
    interface = _cost_metric().model
    threshold = interface.decision_threshold(y_true, y_score, fp=1.0, fn=fn)
    for factor in (1 / 64, 64):
        assert interface.decision_threshold(y_true, y_score, fp=factor, fn=factor * fn) == threshold


@pytest.mark.parametrize('bad_score', [np.nan, np.inf], ids=['nan', 'inf'])
def test_decision_threshold_rejects_scores_that_are_not_finite(data, bad_score):
    y_true, y_score, fn = data
    y_score = y_score.copy()
    y_score[3] = bad_score
    with pytest.raises(ValueError, match='should not contain'):
        _cost_metric().model.decision_threshold(y_true, y_score, fp=1.0, fn=fn)


def test_decision_threshold_rejects_labels_and_scores_of_different_lengths(data):
    y_true, y_score, _ = data
    with pytest.raises(ValueError, match='shapes'):
        _cost_metric().model.decision_threshold(y_true, y_score[:-1], fp=1.0, fn=1.0)
