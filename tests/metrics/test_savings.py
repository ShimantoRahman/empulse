import numpy as np
import pytest

from empulse.metrics.savings import (
    _compute_expected_cost,
    cost_loss,
    expected_cost_loss,
    expected_log_cost_loss,
    expected_savings_score,
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
    ],
)
def test_cost_loss_invalid_input(y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, msg):
    with pytest.raises(ValueError, match=msg):
        assert cost_loss(y_true, y_pred, tp_cost=tp_cost, fp_cost=fp_cost, tn_cost=tn_cost, fn_cost=fn_cost)


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
def test_expected_cost_loss(y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, expected, check_input):
    assert expected_cost_loss(
        y_true, y_pred, tp_cost=tp_cost, fp_cost=fp_cost, tn_cost=tn_cost, fn_cost=fn_cost, check_input=check_input
    ) == pytest.approx(expected)


@pytest.mark.parametrize('check_input', [True, False])
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
    check_input,
):
    avg_expected_cost = expected_cost_loss(
        y_true,
        y_pred,
        tp_cost=tp_costs,
        tn_cost=tn_costs,
        fn_cost=fn_costs,
        fp_cost=fp_costs,
        check_input=check_input,
        normalize=True,
    )
    log_avg_expected_cost = expected_log_cost_loss(
        y_true,
        y_pred,
        tp_cost=tp_costs,
        tn_cost=tn_costs,
        fn_cost=fn_costs,
        fp_cost=fp_costs,
        check_input=check_input,
        normalize=True,
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
        expected_log_cost_loss(y_true, y_pred, tp_cost=tp_costs, tn_cost=tn_costs, normalize=True)
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
    ],
)
def test_expected_cost_loss_invalid_input(y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, msg):
    with pytest.raises(ValueError, match=msg):
        assert expected_cost_loss(y_true, y_pred, tp_cost=tp_cost, fp_cost=fp_cost, tn_cost=tn_cost, fn_cost=fn_cost)


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
    ],
)
def test_saving_score_invalid_input(y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, msg):
    with pytest.raises(ValueError, match=msg):
        savings_score(y_true, y_pred, tp_cost=tp_cost, fp_cost=fp_cost, tn_cost=tn_cost, fn_cost=fn_cost)


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
        ([0, 1, 1, 0], [0.4, 0.8, 0.75, 0.1], 0.0, np.array([4, 1, 2, 2]), 0.0, np.array([1, 3, 3, 1]), 0.475),
    ],
)
def test_expected_saving_score(y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, expected, check_input):
    assert expected_savings_score(
        y_true, y_pred, tp_cost=tp_cost, fp_cost=fp_cost, tn_cost=tn_cost, fn_cost=fn_cost, check_input=check_input
    ) == pytest.approx(expected)


@pytest.mark.parametrize('check_input', [True, False])
@pytest.mark.parametrize(
    'y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, baseline, expected',
    [
        ([1, 0, 0, 1], [1, 1, 0, 0], 1.0, 2.0, 3.0, 4.0, 'prior', 0.2857142857142857),
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
def test_expected_saving_score_with_baseline(
    y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, baseline, expected, check_input
):
    assert expected_savings_score(
        y_true,
        y_pred,
        tp_cost=tp_cost,
        fp_cost=fp_cost,
        tn_cost=tn_cost,
        fn_cost=fn_cost,
        baseline=baseline,
        check_input=check_input,
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
    ],
)
def test_expected_saving_score_invalid_input(y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, msg):
    with pytest.raises(ValueError, match=msg):
        expected_savings_score(y_true, y_pred, tp_cost=tp_cost, fp_cost=fp_cost, tn_cost=tn_cost, fn_cost=fn_cost)
