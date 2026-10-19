"""
The cost and savings metrics, on cases small enough to compute by hand.

``cost_loss`` and ``savings_score`` take decisions; ``expected_cost_loss``,
``expected_log_cost_loss`` and ``expected_savings_score`` take probabilities. These tests call the
public metrics. Their agreement with the reference over many parameter sets is
``test_prebuilt_metric_contract.py``'s job.
"""

import warnings

import numpy as np
import pytest
from sklearn.exceptions import UndefinedMetricWarning

from empulse.metrics import (
    CostMatrix,
    Metric,
    Savings,
    expected_cost_loss,
    expected_log_cost_loss,
    expected_savings_score,
)
from empulse.metrics.savings import (
    _compute_expected_cost,
    cost_loss,
    savings_score,
)


@pytest.mark.parametrize('check_input', [True, False])
@pytest.mark.parametrize(
    'y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, expected',
    [
        ([1, 0, 0, 1], [1, 1, 0, 0], 1.0, 2.0, 3.0, 4.0, 10.0),
        (
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            np.array([1, 2, 3, 4]),
            np.array([1, 2, 3, 4]),
            np.array([1, 2, 3, 4]),
            np.array([1, 2, 3, 4]),
            10.0,
        ),
        (
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            np.array([1, 2, 3, 4]),
            np.array([10, 20, 30, 40]),
            np.array([100, 200, 300, 400]),
            np.array([1000, 2000, 3000, 4000]),
            4321.0,
        ),
        ([1, 0, 0, 1], [1, 1, 0, 0], 1.0, 0.0, 0.0, 0.0, 1.0),
        ([1, 0, 0, 1], [1, 1, 0, 0], 0.0, 2.0, 0.0, 0.0, 2.0),
        ([1, 0, 0, 1], [1, 1, 0, 0], 0.0, 0.0, 3.0, 0.0, 3.0),
        ([1, 0, 0, 1], [1, 1, 0, 0], 0.0, 0.0, 0.0, 4.0, 4.0),
    ],
)
def test_cost_loss(y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, expected, check_input):
    assert (
        cost_loss(
            y_true, y_pred, tp_cost=tp_cost, fp_cost=fp_cost, tn_cost=tn_cost, fn_cost=fn_cost, check_input=check_input
        )
        == expected
    )


@pytest.mark.parametrize('check_input', [True, False])
@pytest.mark.parametrize(
    'y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, expected',
    [
        ([1, 0, 0, 1], [0.51, 0.5, 0.5, 0.51], 0.0, 1.0, 0.0, 1.0, 0.0),  # threshold = 1 / 1 + 1 = 0.5
        ([1, 0, 0, 1], [0.5, 0.51, 0.51, 0.5], 0.0, 1.0, 0.0, 1.0, 4.0),  # threshold = 1 / 1 + 1 = 0.5
        ([1, 0, 0, 1], [0.26, 0.25, 0.25, 0.25], 0.0, 1.0, 0.0, 3.0, 3.0),  # threshold = 1 / 1 + 3 = 0.25
        ([1, 0, 0, 1], [0.25, 0.26, 0.26, 0.25], 0.0, 1.0, 0.0, 3.0, 8.0),  # threshold = 1 / 1 + 3 = 0.25
        (
            [1, 0, 0, 1],
            [0.5, 0.5, 0.5, 0.5],  # TP, FP, TN, FN
            0.0,
            np.array([1, 2, 3, 4]),
            0.0,
            np.array([4, 3, 2, 1]),
            3.0,
        ),  # thresholds = (1/5, 2/5, 3/5, 4/5)
    ],
)
def test_cost_loss_calibrated_probabilities(y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, expected, check_input):
    assert (
        cost_loss(
            y_true, y_pred, tp_cost=tp_cost, fp_cost=fp_cost, tn_cost=tn_cost, fn_cost=fn_cost, check_input=check_input
        )
        == expected
    )


@pytest.mark.parametrize(
    'y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, msg',
    [
        ([1, 0, 0, 1], [1, 1, 0, 0], 0.0, 0.0, 0.0, 0.0, r'All costs are zero.'),
        ([1, 0, 0, 1], [1, 1, 0], 1.0, 0.0, 0.0, 0.0, r'inconsistent numbers of samples'),
        ([1, 0], [1, 1], [1, 2, 3], [1, 2], [1, 2], [1, 2], r'inconsistent numbers of samples'),
        ([1, 0], [1, 1], [1, 2], [1, 2, 3], [1, 2], [1, 2], r'inconsistent numbers of samples'),
        ([1, 0], [1, 1], [1, 2], [1, 2], [1, 2, 3], [1, 2], r'inconsistent numbers of samples'),
        ([1, 0], [1, 1], [1, 2], [1, 2], [1, 2], [1, 2, 3], r'inconsistent numbers of samples'),
        ([1, 0, 0, 1], [1, 1, 0, 0], 0.0, np.inf, 0.0, 1.0, r'fp_cost must be finite'),
        ([1, 0], [1, 1], 0.0, 1.0, 0.0, [1.0, np.nan], r'fn_cost must be finite'),
    ],
)
def test_cost_loss_invalid_input(y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, msg):
    with pytest.raises(ValueError, match=msg):
        assert cost_loss(y_true, y_pred, tp_cost=tp_cost, fp_cost=fp_cost, tn_cost=tn_cost, fn_cost=fn_cost)


@pytest.mark.parametrize(
    'y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, expected',
    [
        ([1, 0, 0, 1], [1, 1, 0, 0], 1.0, 2.0, 3.0, 4.0, 10.0),
        (
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            np.array([1, 2, 3, 4]),
            np.array([1, 2, 3, 4]),
            np.array([1, 2, 3, 4]),
            np.array([1, 2, 3, 4]),
            10.0,
        ),
        (
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            np.array([1, 2, 3, 4]),
            np.array([10, 20, 30, 40]),
            np.array([100, 200, 300, 400]),
            np.array([1000, 2000, 3000, 4000]),
            4321.0,
        ),
        ([1, 0, 0, 1], [1, 1, 0, 0], 1.0, 0.0, 0.0, 0.0, 1.0),
        ([1, 0, 0, 1], [1, 1, 0, 0], 0.0, 2.0, 0.0, 0.0, 2.0),
        ([1, 0, 0, 1], [1, 1, 0, 0], 0.0, 0.0, 3.0, 0.0, 3.0),
        ([1, 0, 0, 1], [1, 1, 0, 0], 0.0, 0.0, 0.0, 4.0, 4.0),
        (
            [0, 1, 1, 0],
            [0.4, 0.8, 0.75, 0.1],
            np.array([0, 0, 0, 0]),
            np.array([4, 1, 2, 2]),
            np.array([0, 0, 0, 0]),
            np.array([1, 3, 3, 1]),
            3.15,
        ),
    ],
)
def test_expected_cost_loss(y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, expected):
    # `expected` is the total cost; the metric reports the average.
    assert expected_cost_loss(
        y_true, y_pred, tp_cost=tp_cost, fp_cost=fp_cost, tn_cost=tn_cost, fn_cost=fn_cost
    ) == pytest.approx(expected / len(y_true))


@pytest.mark.parametrize(
    'y_true, y_pred, tp_costs, tn_costs, fn_costs, fp_costs, expected_cost, expected_avg_cost, expected_log_cost',
    [
        (
            np.array([1, 0, 1, 0]),
            np.array([0.9, 0.1, 0.8, 0.2]),
            np.array([1, 1, 1, 1]),
            np.array([1, 1, 1, 1]),
            np.array([1, 1, 1, 1]),
            np.array([1, 1, 1, 1]),
            np.array([1.0, 1.0, 1.0, 1.0]),
            1.0,
            -2.120263536200091,
        ),
        (
            np.array([1, 0, 1, 0]),
            np.array([0.9, 0.1, 0.8, 0.2]),
            np.array([0, 0, 0, 0]),
            np.array([0, 0, 0, 0]),
            np.array([0, 0, 0, 0]),
            np.array([0, 0, 0, 0]),
            np.array([0.0, 0.0, 0.0, 0.0]),
            0.0,
            0.0,
        ),
        (
            np.array([1, 0, 1, 0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
            np.array([1, 1, 1, 1]),
            np.array([2, 2, 2, 2]),
            np.array([4, 4, 4, 4]),
            np.array([8, 8, 8, 8]),
            np.array([1.0, 2.0, 4.0, 8.0]),
            3.75,
            -135.16370020918933,
        ),
        (
            np.array([1, 0, 1, 0]),
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([1, 1, 1, 1]),
            np.array([2, 2, 2, 2]),
            np.array([4, 4, 4, 4]),
            np.array([8, 8, 8, 8]),
            np.array([2.5, 5.0, 2.5, 5.0]),
            3.75,
            -5.19860385419959,
        ),
        (
            np.array([1, 0, 1, 0]),
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([1, 0, 1, 0]),
            np.array([0, 2, 0, 2]),
            np.array([4, 0, 4, 0]),
            np.array([0, 8, 0, 8]),
            np.array([2.5, 5.0, 2.5, 5.0]),
            3.75,
            -5.19860385419959,
        ),
        (
            np.array([1, 0, 1, 0]),
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([2, 0, 4, 0]),
            np.array([0, 2, 0, 8]),
            np.array([4, 0, 16, 0]),
            np.array([0, 8, 0, 32]),
            np.array([3.0, 5.0, 10.0, 20.0]),
            9.5,
            -13.16979643063896,
        ),
        (
            np.array([1, 0, 1, 0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
            np.array([1, 1, 1, 1]),
            0.0,
            0.0,
            0.0,
            np.array([1.0, 0.0, 0.0, 0.0]),
            0.25,
            -9.010913347279288,
        ),
        (
            np.array([1, 0, 1, 0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
            0.0,
            np.array([1, 1, 1, 1]),
            0.0,
            0.0,
            np.array([0.0, 1.0, 0.0, 0.0]),
            0.25,
            -9.010913347279288,
        ),
        (
            np.array([1, 0, 1, 0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
            0.0,
            0.0,
            np.array([1, 1, 1, 1]),
            0.0,
            np.array([0.0, 0.0, 1.0, 0.0]),
            0.25,
            -9.010913347279288,
        ),
        (
            np.array([1, 0, 1, 0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
            0.0,
            0.0,
            0.0,
            np.array([1, 1, 1, 1]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            0.25,
            -9.010913347279288,
        ),
    ],
)
def test_average_expected_cost(
    y_true,
    y_pred,
    tp_costs,
    tn_costs,
    fn_costs,
    fp_costs,
    expected_cost,
    expected_avg_cost,
    expected_log_cost,
):
    avg_expected_cost = expected_cost_loss(
        y_true,
        y_pred,
        tp_cost=tp_costs,
        tn_cost=tn_costs,
        fn_cost=fn_costs,
        fp_cost=fp_costs,
    )
    log_avg_expected_cost = expected_log_cost_loss(
        y_true,
        y_pred,
        tp_cost=tp_costs,
        tn_cost=tn_costs,
        fn_cost=fn_costs,
        fp_cost=fp_costs,
    )
    assert np.isclose(
        expected_cost, _compute_expected_cost(y_true, y_pred, tp_costs, tn_costs, fn_costs, fp_costs)
    ).all()
    assert avg_expected_cost == pytest.approx(expected_avg_cost)
    assert log_avg_expected_cost == pytest.approx(expected_log_cost)


def test_log_aec_cross_entropy():
    from sklearn.metrics import log_loss

    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    tp_costs = -1
    tn_costs = -1

    assert log_loss(y_true, y_pred) == pytest.approx(
        expected_log_cost_loss(y_true, y_pred, tp_cost=tp_costs, tn_cost=tn_costs)
    )


@pytest.mark.parametrize('check_input', [True, False])
@pytest.mark.parametrize(
    'y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, expected',
    [
        ([1, 0, 0, 1], [1, 1, 0, 0], 1.0, 2.0, 3.0, 4.0, -2 / 3),  # 1 - 10 / min(2*1 + 2*2 = 6, 2*3 + 2*4 = 14)
        ([1, 0, 0, 1], [1, 1, 1, 1], 0.0, 1.0, 0.0, 1.0, 0.0),
        ([1, 0, 0, 1], [0, 0, 0, 0], 0.0, 1.0, 0.0, 1.0, 0.0),
        ([1, 0, 0, 1], [1, 0, 0, 1], 0.0, 1.0, 0.0, 1.0, 1.0),
        ([1, 0, 0, 1], [1, 0, 1, 1], 0.0, 1.0, 0.0, 1.0, 0.5),
        ([1, 0, 0, 1], [1, 1, 0, 1], 0.0, 1.0, 0.0, 1.0, 0.5),
        ([1, 0, 0, 1], [1, 0, 0, 1], 0.0, np.array([1, 2, 3, 4]), 0.0, np.array([1, 2, 3, 4]), 1.0),
        ([1, 0, 0, 1], [1, 1, 1, 1], 0.0, np.array([1, 2, 3, 4]), 0.0, np.array([1, 2, 3, 4]), 0.0),
        ([1, 0, 0, 1], [0, 0, 0, 0], 0.0, np.array([1, 2, 3, 4]), 0.0, np.array([1, 2, 3, 4]), 0.0),
        (
            [1, 0, 0, 1],
            [1, 0, 1, 1],
            0.0,
            np.array([1, 2, 3, 4]),
            0.0,
            np.array([1, 2, 3, 4]),
            0.4,
        ),  # 1 - 3 / min(2+3, 1+4)
        (
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            0.0,
            np.array([1, 2, 3, 4]),
            0.0,
            np.array([1, 2, 3, 4]),
            0.6,
        ),  # 1 - 2 / min(2+3, 1+4)
    ],
)
def test_saving_score(y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, expected, check_input):
    assert savings_score(
        y_true, y_pred, tp_cost=tp_cost, fp_cost=fp_cost, tn_cost=tn_cost, fn_cost=fn_cost, check_input=check_input
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    'y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, msg',
    [
        ([1, 0, 0, 1], [1, 1, 0, 0], 0.0, 0.0, 0.0, 0.0, r'All costs are zero.'),
        ([1, 0, 0, 1], [1, 1, 0], 1.0, 0.0, 0.0, 0.0, r'inconsistent numbers of samples'),
        ([1, 0], [1, 1], [1, 2, 3], [1, 2], [1, 2], [1, 2], r'inconsistent numbers of samples'),
        ([1, 0], [1, 1], [1, 2], [1, 2, 3], [1, 2], [1, 2], r'inconsistent numbers of samples'),
        ([1, 0], [1, 1], [1, 2], [1, 2], [1, 2, 3], [1, 2], r'inconsistent numbers of samples'),
        ([1, 0], [1, 1], [1, 2], [1, 2], [1, 2], [1, 2, 3], r'inconsistent numbers of samples'),
        ([1, 0, 0, 1], [1, 1, 0, 0], 0.0, np.inf, 0.0, 1.0, r'fp_cost must be finite'),
        ([1, 0], [1, 1], 0.0, 1.0, 0.0, [1.0, np.nan], r'fn_cost must be finite'),
    ],
)
def test_saving_score_invalid_input(y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, msg):
    with pytest.raises(ValueError, match=msg):
        savings_score(y_true, y_pred, tp_cost=tp_cost, fp_cost=fp_cost, tn_cost=tn_cost, fn_cost=fn_cost)


@pytest.mark.parametrize(
    'y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, expected',
    [
        ([1, 0, 0, 1], [1, 1, 0, 0], 1.0, 2.0, 3.0, 4.0, -2 / 3),  # 1 - 10 / min(2*1 + 2*2 = 6, 2*3 + 2*4 = 14)
        ([1, 0, 0, 1], [1, 1, 1, 1], 0.0, 1.0, 0.0, 1.0, 0.0),
        ([1, 0, 0, 1], [0, 0, 0, 0], 0.0, 1.0, 0.0, 1.0, 0.0),
        ([1, 0, 0, 1], [1, 0, 0, 1], 0.0, 1.0, 0.0, 1.0, 1.0),
        ([1, 0, 0, 1], [1, 0, 1, 1], 0.0, 1.0, 0.0, 1.0, 0.5),
        ([1, 0, 0, 1], [1, 1, 0, 1], 0.0, 1.0, 0.0, 1.0, 0.5),
        ([1, 0, 0, 1], [1, 0, 0, 1], 0.0, np.array([1, 2, 3, 4]), 0.0, np.array([1, 2, 3, 4]), 1.0),
        ([1, 0, 0, 1], [1, 1, 1, 1], 0.0, np.array([1, 2, 3, 4]), 0.0, np.array([1, 2, 3, 4]), 0.0),
        ([1, 0, 0, 1], [0, 0, 0, 0], 0.0, np.array([1, 2, 3, 4]), 0.0, np.array([1, 2, 3, 4]), 0.0),
        (
            [1, 0, 0, 1],
            [1, 0, 1, 1],
            0.0,
            np.array([1, 2, 3, 4]),
            0.0,
            np.array([1, 2, 3, 4]),
            0.4,
        ),  # 1 - 3 / min(2+3, 1+4)
        (
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            0.0,
            np.array([1, 2, 3, 4]),
            0.0,
            np.array([1, 2, 3, 4]),
            0.6,
        ),  # 1 - 2 / min(2+3, 1+4)
        ([0, 1, 1, 0], [0.4, 0.8, 0.75, 0.1], 0.0, np.array([4, 1, 2, 2]), 0.0, np.array([1, 3, 3, 1]), 0.475),
    ],
)
def test_expected_saving_score(y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, expected):
    assert expected_savings_score(
        y_true, y_pred, tp_cost=tp_cost, fp_cost=fp_cost, tn_cost=tn_cost, fn_cost=fn_cost
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    'y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, baseline, expected',
    [
        # The model's total cost is 1 + 2 + 3 + 4 = 10, and so is the expected cost of predicting the
        # prior 0.5 for every sample.
        ([1, 0, 0, 1], [1, 1, 0, 0], 1.0, 2.0, 3.0, 4.0, 'prior', 0.0),
        ([1, 0, 0, 1], [1, 1, 1, 1], 0.0, 1.0, 0.0, 1.0, 'prior', 0.0),
        ([1, 0, 0, 1], [0, 0, 0, 0], 0.0, 1.0, 0.0, 1.0, 'prior', 0.0),
        ([1, 0, 0, 1], [1, 0, 0, 1], 0.0, 1.0, 0.0, 1.0, 'prior', 1.0),
        ([1, 0, 0, 1], [1, 0, 1, 1], 0.0, 1.0, 0.0, 1.0, 'prior', 0.5),
        ([1, 0, 0, 1], [1, 1, 0, 1], 0.0, 1.0, 0.0, 1.0, 'prior', 0.5),
        ([1, 0, 0, 1], [1, 0, 0, 1], 0.0, np.array([1, 2, 3, 4]), 0.0, np.array([1, 2, 3, 4]), 'prior', 1.0),
        ([1, 0, 0, 1], [1, 1, 1, 1], 0.0, np.array([1, 2, 3, 4]), 0.0, np.array([1, 2, 3, 4]), 'prior', 0.0),
        ([1, 0, 0, 1], [0, 0, 0, 0], 0.0, np.array([1, 2, 3, 4]), 0.0, np.array([1, 2, 3, 4]), 'prior', 0.0),
        ([1, 0, 0, 1], [1, 0, 1, 1], 0.0, np.array([1, 2, 3, 4]), 0.0, np.array([1, 2, 3, 4]), 'prior', 0.4),
        ([1, 0, 0, 1], [1, 1, 0, 1], 0.0, np.array([1, 2, 3, 4]), 0.0, np.array([1, 2, 3, 4]), 'prior', 0.6),
        ([0, 1, 1, 0], [0.4, 0.8, 0.75, 0.1], 0.0, np.array([4, 1, 2, 2]), 0.0, np.array([1, 3, 3, 1]), 'prior', 0.475),
        ([1, 0, 0, 1], [1, 1, 0, 0], 1.0, 2.0, 3.0, 4.0, [0.5, 0.5, 0.5, 0.5], 0.0),
        ([1, 0, 0, 1], [1, 1, 1, 1], 0.0, 1.0, 0.0, 1.0, [0.5, 0.5, 0.5, 0.5], 0.0),
        ([1, 0, 0, 1], [0, 0, 0, 0], 0.0, 1.0, 0.0, 1.0, [0.5, 0.5, 0.5, 0.5], 0.0),
        ([1, 0, 0, 1], [1, 0, 0, 1], 0.0, 1.0, 0.0, 1.0, [0.5, 0.5, 0.5, 0.5], 1.0),
        ([1, 0, 0, 1], [1, 0, 1, 1], 0.0, 1.0, 0.0, 1.0, [0.5, 0.5, 0.5, 0.5], 0.5),
        ([1, 0, 0, 1], [1, 1, 0, 1], 0.0, 1.0, 0.0, 1.0, [0.5, 0.5, 0.5, 0.5], 0.5),
        (
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            0.0,
            np.array([1, 2, 3, 4]),
            0.0,
            np.array([1, 2, 3, 4]),
            [0.5, 0.5, 0.5, 0.5],
            1.0,
        ),
        (
            [1, 0, 0, 1],
            [1, 1, 1, 1],
            0.0,
            np.array([1, 2, 3, 4]),
            0.0,
            np.array([1, 2, 3, 4]),
            [0.5, 0.5, 0.5, 0.5],
            0.0,
        ),
        (
            [1, 0, 0, 1],
            [0, 0, 0, 0],
            0.0,
            np.array([1, 2, 3, 4]),
            0.0,
            np.array([1, 2, 3, 4]),
            [0.5, 0.5, 0.5, 0.5],
            0.0,
        ),
        (
            [1, 0, 0, 1],
            [1, 0, 1, 1],
            0.0,
            np.array([1, 2, 3, 4]),
            0.0,
            np.array([1, 2, 3, 4]),
            [0.5, 0.5, 0.5, 0.5],
            0.4,
        ),
        (
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            0.0,
            np.array([1, 2, 3, 4]),
            0.0,
            np.array([1, 2, 3, 4]),
            [0.5, 0.5, 0.5, 0.5],
            0.6,
        ),
        (
            [0, 1, 1, 0],
            [0.4, 0.8, 0.75, 0.1],
            0.0,
            np.array([4, 1, 2, 2]),
            0.0,
            np.array([1, 3, 3, 1]),
            [0.5, 0.5, 0.5, 0.5],
            0.475,
        ),
    ],
)
def test_expected_saving_score_with_baseline(y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, baseline, expected):
    assert expected_savings_score(
        y_true,
        y_pred,
        tp_cost=tp_cost,
        fp_cost=fp_cost,
        tn_cost=tn_cost,
        fn_cost=fn_cost,
        baseline=baseline,
    ) == pytest.approx(expected)


class TestNegativeBaselineCost:
    """A matrix of benefits makes the baseline cost negative; a cheaper model must still score higher."""

    y_true = np.array([0, 1, 1, 0])

    def test_savings_score(self):
        perfect = savings_score(self.y_true, self.y_true, tp_cost=-10.0, fp_cost=1.0)
        worst = savings_score(self.y_true, 1 - self.y_true, tp_cost=-10.0, fp_cost=1.0)
        assert perfect > 0 > worst

    def test_expected_savings_score(self):
        perfect = expected_savings_score(self.y_true, self.y_true.astype(float), tp_cost=-10.0, fp_cost=1.0)
        uninformed = expected_savings_score(self.y_true, np.full(4, 0.5), tp_cost=-10.0, fp_cost=1.0)
        assert perfect > uninformed

    def test_savings_strategy_ranks_like_cost(self):
        metric = Metric(CostMatrix().add_tp_benefit('b').add_tn_benefit('b'), Savings())
        good = np.array([0.1, 0.9, 0.9, 0.1])
        bad = np.full(4, 0.5)
        assert metric(self.y_true, good, b=1.0) > metric(self.y_true, bad, b=1.0)

    def test_positive_baseline_cost_is_unchanged(self):
        # (base - cost) / |base| equals 1 - cost / base whenever the baseline cost is positive.
        y_pred = np.array([0, 1, 0, 0])
        fp_cost = np.array([4, 1, 2, 2])
        fn_cost = np.array([1, 3, 3, 1])
        base = min(
            cost_loss(self.y_true, np.zeros(4), fp_cost=fp_cost, fn_cost=fn_cost),
            cost_loss(self.y_true, np.ones(4), fp_cost=fp_cost, fn_cost=fn_cost),
        )
        cost = cost_loss(self.y_true, y_pred, fp_cost=fp_cost, fn_cost=fn_cost)
        assert savings_score(self.y_true, y_pred, fp_cost=fp_cost, fn_cost=fn_cost) == pytest.approx(1 - cost / base)


class TestZeroBaselineCost:
    """A baseline that costs nothing leaves the savings undefined."""

    y_true = np.zeros(4, dtype=int)
    y_score = np.array([0.1, 0.2, 0.3, 0.4])

    def test_savings_score(self):
        with pytest.warns(UndefinedMetricWarning, match='zero cost'):
            assert np.isnan(savings_score(self.y_true, self.y_score, fp_cost=1.0, fn_cost=1.0, check_input=False))

    def test_savings_strategy(self):
        metric = Metric(CostMatrix().add_fp_cost('c').add_fn_cost('d'), Savings())
        with pytest.warns(UndefinedMetricWarning, match='zero cost'):
            assert np.isnan(metric(self.y_true, self.y_score, c=1.0, d=1.0))


class TestSavingsStrategyBaselines:
    metric = Metric(CostMatrix().add_fp_cost('c').add_fn_cost('d'), Savings())
    y_true = np.array([1] * 10 + [0] * 90)
    y_score = np.linspace(0.05, 0.95, 100)

    @pytest.mark.parametrize('baseline', [0.3, [0.3] * 100, np.full(100, 0.3)])
    def test_scalar_and_list_baselines_match_an_array(self, baseline):
        expected = self.metric(self.y_true, self.y_score, c=1.0, d=20.0, baseline=np.full(100, 0.3))
        assert self.metric(self.y_true, self.y_score, c=1.0, d=20.0, baseline=baseline) == pytest.approx(expected)

    def test_prior_baseline_predicts_the_positive_class_prior(self):
        prior = np.full(100, self.y_true.mean())
        expected = self.metric(self.y_true, self.y_score, c=1.0, d=20.0, baseline=prior)
        assert self.metric(self.y_true, self.y_score, c=1.0, d=20.0, baseline='prior') == pytest.approx(expected)


def test_cost_loss_with_a_zero_threshold_denominator_does_not_divide_by_zero():
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        cost = cost_loss([0, 1], [0.3, 0.6], fp_cost=np.array([0.0, 1.0]), fn_cost=np.array([0.0, 1.0]))
    assert cost == 0.0
