import numpy as np
import pytest

from empulse.metrics.savings import cost_loss, savings_score


@pytest.mark.parametrize("check_input", [True, False])
@pytest.mark.parametrize("y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, expected", [
    ([1, 0, 0, 1], [1, 1, 0, 0], 1.0, 2.0, 3.0, 4.0, 10.0),
    ([1, 0, 0, 1], [1, 1, 0, 0], np.array([1, 2, 3, 4]),
     np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]), 10.0),
    ([1, 0, 0, 1], [1, 1, 0, 0], np.array([1, 2, 3, 4]),
     np.array([10, 20, 30, 40]), np.array([100, 200, 300, 400]), np.array([1000, 2000, 3000, 4000]), 4321.0),
    ([1, 0, 0, 1], [1, 1, 0, 0], 1.0, 0.0, 0.0, 0.0, 1.0),
    ([1, 0, 0, 1], [1, 1, 0, 0], 0.0, 2.0, 0.0, 0.0, 2.0),
    ([1, 0, 0, 1], [1, 1, 0, 0], 0.0, 0.0, 3.0, 0.0, 3.0),
    ([1, 0, 0, 1], [1, 1, 0, 0], 0.0, 0.0, 0.0, 4.0, 4.0),
])
def test_cost_loss(y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, expected, check_input):
    assert cost_loss(y_true, y_pred, tp_cost=tp_cost, fp_cost=fp_cost,
                     tn_cost=tn_cost, fn_cost=fn_cost, check_input=check_input) == expected


@pytest.mark.parametrize("check_input", [True, False])
@pytest.mark.parametrize("y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, expected", [
    ([1, 0, 0, 1], [0.51, 0.5, 0.5, 0.51], 0.0, 1.0, 0.0, 1.0, 0.0),  # threshold = 1 / 1 + 1 = 0.5
    ([1, 0, 0, 1], [0.5, 0.51, 0.51, 0.5], 0.0, 1.0, 0.0, 1.0, 4.0),  # threshold = 1 / 1 + 1 = 0.5
    ([1, 0, 0, 1], [0.26, 0.25, 0.25, 0.25], 0.0, 1.0, 0.0, 3.0, 3.0),  # threshold = 1 / 1 + 3 = 0.25
    ([1, 0, 0, 1], [0.25, 0.26, 0.26, 0.25], 0.0, 1.0, 0.0, 3.0, 8.0),  # threshold = 1 / 1 + 3 = 0.25
    ([1, 0, 0, 1], [0.5, 0.5, 0.5, 0.5],  # TP, FP, TN, FN
     0.0, np.array([1, 2, 3, 4]), 0.0, np.array([4, 3, 2, 1]), 3.0),  # thresholds = (1/5, 2/5, 3/5, 4/5)
])
def test_cost_loss_calibrated_probabilities(y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, expected, check_input):
    assert cost_loss(y_true, y_pred, tp_cost=tp_cost, fp_cost=fp_cost,
                     tn_cost=tn_cost, fn_cost=fn_cost, check_input=check_input) == expected


@pytest.mark.parametrize("y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, msg", [
    ([1, 0, 0, 1], [1, 1, 0, 0], 0.0, 0.0, 0.0, 0.0, r"All costs are zero."),
    ([1, 0, 0, 1], [1, 1, 0], 1.0, 0.0, 0.0, 0.0, r"inconsistent numbers of samples"),
    ([1, 0], [1, 1], [1, 2, 3], [1, 2], [1, 2], [1, 2], r"inconsistent numbers of samples"),
    ([1, 0], [1, 1], [1, 2], [1, 2, 3], [1, 2], [1, 2], r"inconsistent numbers of samples"),
    ([1, 0], [1, 1], [1, 2], [1, 2], [1, 2, 3], [1, 2], r"inconsistent numbers of samples"),
    ([1, 0], [1, 1], [1, 2], [1, 2], [1, 2], [1, 2, 3], r"inconsistent numbers of samples"),
])
def test_cost_loss_invalid_input(y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, msg):
    with pytest.raises(ValueError, match=msg):
        assert cost_loss(y_true, y_pred, tp_cost=tp_cost, fp_cost=fp_cost, tn_cost=tn_cost, fn_cost=fn_cost)


@pytest.mark.parametrize("check_input", [True, False])
@pytest.mark.parametrize("y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, expected", [
    ([1, 0, 0, 1], [1, 1, 0, 0], 1.0, 2.0, 3.0, 4.0, -2 / 3),  # 1 - 10 / min(2*1 + 2*2 = 6, 2*3 + 2*4 = 14)
    ([1, 0, 0, 1], [1, 1, 1, 1], 0.0, 1.0, 0.0, 1.0, 0.0),
    ([1, 0, 0, 1], [0, 0, 0, 0], 0.0, 1.0, 0.0, 1.0, 0.0),
    ([1, 0, 0, 1], [1, 0, 0, 1], 0.0, 1.0, 0.0, 1.0, 1.0),
    ([1, 0, 0, 1], [1, 0, 1, 1], 0.0, 1.0, 0.0, 1.0, 0.5),
    ([1, 0, 0, 1], [1, 1, 0, 1], 0.0, 1.0, 0.0, 1.0, 0.5),
    ([1, 0, 0, 1], [1, 0, 0, 1],
     0.0, np.array([1, 2, 3, 4]),
     0.0, np.array([1, 2, 3, 4]),
     1.0),
    ([1, 0, 0, 1], [1, 1, 1, 1],
     0.0, np.array([1, 2, 3, 4]),
     0.0, np.array([1, 2, 3, 4]),
     0.0),
    ([1, 0, 0, 1], [0, 0, 0, 0],
     0.0, np.array([1, 2, 3, 4]),
     0.0, np.array([1, 2, 3, 4]),
     0.0),
    ([1, 0, 0, 1], [1, 0, 1, 1],
     0.0, np.array([1, 2, 3, 4]),
     0.0, np.array([1, 2, 3, 4]),
     0.4),  # 1 - 3 / min(2+3, 1+4)
    ([1, 0, 0, 1], [1, 1, 0, 1],
     0.0, np.array([1, 2, 3, 4]),
     0.0, np.array([1, 2, 3, 4]),
     0.6),  # 1 - 2 / min(2+3, 1+4)

])
def test_saving_score(y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, expected, check_input):
    assert savings_score(y_true, y_pred, tp_cost=tp_cost, fp_cost=fp_cost,
                         tn_cost=tn_cost, fn_cost=fn_cost, check_input=check_input) == pytest.approx(expected)


@pytest.mark.parametrize("y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, msg", [
    ([1, 0, 0, 1], [1, 1, 0, 0], 0.0, 0.0, 0.0, 0.0, r"All costs are zero."),
    ([1, 0, 0, 1], [1, 1, 0], 1.0, 0.0, 0.0, 0.0, r"inconsistent numbers of samples"),
    ([1, 0], [1, 1], [1, 2, 3], [1, 2], [1, 2], [1, 2], r"inconsistent numbers of samples"),
    ([1, 0], [1, 1], [1, 2], [1, 2, 3], [1, 2], [1, 2], r"inconsistent numbers of samples"),
    ([1, 0], [1, 1], [1, 2], [1, 2], [1, 2, 3], [1, 2], r"inconsistent numbers of samples"),
    ([1, 0], [1, 1], [1, 2], [1, 2], [1, 2], [1, 2, 3], r"inconsistent numbers of samples"),
])
def test_saving_score_invalid_input(y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, msg):
    with pytest.raises(ValueError, match=msg):
        savings_score(y_true, y_pred, tp_cost=tp_cost, fp_cost=fp_cost, tn_cost=tn_cost, fn_cost=fn_cost)
