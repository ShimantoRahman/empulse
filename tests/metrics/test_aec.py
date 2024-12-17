import numpy as np
import pytest

from empulse.metrics.aec import aec_loss, log_aec_loss, _compute_expected_cost


@pytest.mark.parametrize("check_input", [True, False])
@pytest.mark.parametrize(
    "y_true, y_pred, tp_costs, tn_costs, fn_costs, fp_costs, expected_cost, expected_avg_cost, expected_log_cost",
    [
        (np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2]),
         np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]),
         np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]),
         np.array([1.0, 1.0, 1.0, 1.0]), 1.0, 0.0),

        (np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2]),
         np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]),
         np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]),
         np.array([0.0, 0.0, 0.0, 0.0]), 0.0, -36.04365338911715),

        (np.array([1, 0, 1, 0]), np.array([1.0, 0.0, 0.0, 1.0]),
         np.array([1, 1, 1, 1]), np.array([2, 2, 2, 2]),
         np.array([4, 4, 4, 4]), np.array([8, 8, 8, 8]),
         np.array([1.0, 2.0, 4.0, 8.0]), 3.75, 1.0397207708399179),

        (np.array([1, 0, 1, 0]), np.array([0.5, 0.5, 0.5, 0.5]),
         np.array([1, 1, 1, 1]), np.array([2, 2, 2, 2]),
         np.array([4, 4, 4, 4]), np.array([8, 8, 8, 8]),
         np.array([2.5, 5.0, 2.5, 5.0]), 3.75, 1.2628643221541278),

        (np.array([1, 0, 1, 0]), np.array([1.0, 0.0, 0.0, 1.0]),
         np.exp(np.array([1, 1, 1, 1])), np.exp(np.array([2, 2, 2, 2])),
         np.exp(np.array([4, 4, 4, 4])), np.exp(np.array([8, 8, 8, 8])),
         np.exp(np.array([1.0, 2.0, 4.0, 8.0])), 761.4158687505656, 3.75),

        (np.array([1, 0, 1, 0]), np.array([0.5, 0.5, 0.5, 0.5]),
         np.array([1, 0, 1, 0]), np.array([0, 2, 0, 2]),
         np.array([4, 0, 4, 0]), np.array([0, 8, 0, 8]),
         np.array([2.5, 5.0, 2.5, 5.0]), 3.75, 1.2628643221541278),

        (np.array([1, 0, 1, 0]), np.array([0.5, 0.5, 0.5, 0.5]),
         np.array([2, 0, 4, 0]), np.array([0, 2, 0, 8]),
         np.array([4, 0, 16, 0]), np.array([0, 8, 0, 32]),
         np.array([3.0, 5.0, 10.0, 20.0]), 9.5, 2.0015918919125615),

        (np.array([1, 0, 1, 0]), np.array([1.0, 0.0, 0.0, 1.0]),
         np.array([1, 1, 1, 1]), 0.0,
         0.0, 0.0,
         np.array([1.0, 0.0, 0.0, 0.0]), 0.25, -27.032740041837865),

        (np.array([1, 0, 1, 0]), np.array([1.0, 0.0, 0.0, 1.0]),
         0.0, np.array([1, 1, 1, 1]),
         0.0, 0.0,
         np.array([0.0, 1.0, 0.0, 0.0]), 0.25, -27.032740041837865),

        (np.array([1, 0, 1, 0]), np.array([1.0, 0.0, 0.0, 1.0]),
         0.0, 0.0,
         np.array([1, 1, 1, 1]), 0.0,
         np.array([0.0, 0.0, 1.0, 0.0]), 0.25, -27.032740041837865),

        (np.array([1, 0, 1, 0]), np.array([1.0, 0.0, 0.0, 1.0]),
         0.0, 0.0,
         0.0, np.array([1, 1, 1, 1]),
         np.array([0.0, 0.0, 0.0, 1.0]), 0.25, -27.032740041837865),

    ])
def test_aec(y_true, y_pred, tp_costs, tn_costs, fn_costs, fp_costs, expected_cost, expected_avg_cost,
             expected_log_cost, check_input):
    avg_expected_cost = aec_loss(y_true, y_pred, tp_cost=tp_costs, tn_cost=tn_costs,
                                 fn_cost=fn_costs, fp_cost=fp_costs, check_input=check_input)
    log_avg_expected_cost = log_aec_loss(y_true, y_pred, tp_cost=tp_costs, tn_cost=tn_costs,
                                         fn_cost=fn_costs, fp_cost=fp_costs, check_input=check_input)
    assert np.isclose(expected_cost,
                      _compute_expected_cost(y_true, y_pred, tp_costs, tn_costs, fn_costs, fp_costs, check_input)).all()
    assert avg_expected_cost == pytest.approx(expected_avg_cost)
    assert log_avg_expected_cost == pytest.approx(expected_log_cost)


def test_unequal_array_lengths():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1.0, 0.0, 0.0, 1.0])
    tp_costs = np.array([1, 1, 1])

    with pytest.raises(ValueError):
        _compute_expected_cost(y_true, y_pred, tp_cost=tp_costs)
    with pytest.raises(ValueError):
        aec_loss(y_true, y_pred, tp_cost=tp_costs)
    with pytest.raises(ValueError):
        log_aec_loss(y_true, y_pred, tp_cost=tp_costs)
    with pytest.raises(ValueError):
        _compute_expected_cost(y_true, y_pred, tn_cost=tp_costs)
    with pytest.raises(ValueError):
        aec_loss(y_true, y_pred, tn_cost=tp_costs)
    with pytest.raises(ValueError):
        log_aec_loss(y_true, y_pred, tn_cost=tp_costs)
    with pytest.raises(ValueError):
        _compute_expected_cost(y_true, y_pred, fn_cost=tp_costs)
    with pytest.raises(ValueError):
        aec_loss(y_true, y_pred, fn_cost=tp_costs)
    with pytest.raises(ValueError):
        log_aec_loss(y_true, y_pred, fn_cost=tp_costs)
    with pytest.raises(ValueError):
        _compute_expected_cost(y_true, y_pred, fp_cost=tp_costs)
    with pytest.raises(ValueError):
        aec_loss(y_true, y_pred, fp_cost=tp_costs)
    with pytest.raises(ValueError):
        log_aec_loss(y_true, y_pred, fp_cost=tp_costs)
