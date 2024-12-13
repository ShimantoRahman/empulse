import numpy as np
import pytest

from empulse.metrics.aec import aec_score, log_aec_score, _compute_expected_cost


def test_equal_costs():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    tp_costs = np.array([1, 1, 1, 1])
    tn_costs = np.array([1, 1, 1, 1])
    fn_costs = np.array([1, 1, 1, 1])
    fp_costs = np.array([1, 1, 1, 1])
    expected_cost = _compute_expected_cost(y_true, y_pred, tp_costs, tn_costs, fn_costs, fp_costs)
    avg_expected_cost = aec_score(y_true, y_pred, tp_costs=tp_costs,
                                  tn_costs=tn_costs, fn_costs=fn_costs, fp_costs=fp_costs)
    log_avg_expected_cost = log_aec_score(y_true, y_pred, tp_costs=tp_costs,
                                          tn_costs=tn_costs, fn_costs=fn_costs, fp_costs=fp_costs)
    assert np.isclose(expected_cost, np.array([1.0, 1.0, 1.0, 1.0])).all()
    assert avg_expected_cost == pytest.approx(1.0)
    assert log_avg_expected_cost == pytest.approx(0.0)


def test_equal_zero_costs():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    tp_costs = np.array([0, 0, 0, 0])
    tn_costs = np.array([0, 0, 0, 0])
    fn_costs = np.array([0, 0, 0, 0])
    fp_costs = np.array([0, 0, 0, 0])
    expected_cost = _compute_expected_cost(y_true, y_pred, tp_costs, tn_costs, fn_costs, fp_costs)
    avg_expected_cost = aec_score(y_true, y_pred, tp_costs=tp_costs,
                                  tn_costs=tn_costs, fn_costs=fn_costs, fp_costs=fp_costs)
    assert np.isclose(expected_cost, np.array([0.0, 0.0, 0.0, 0.0])).all()
    assert avg_expected_cost == pytest.approx(0.0)


def test_equal_combination_costs():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1.0, 0.0, 0.0, 1.0])
    tp_costs = np.array([1, 1, 1, 1])
    tn_costs = np.array([2, 2, 2, 2])
    fn_costs = np.array([4, 4, 4, 4])
    fp_costs = np.array([8, 8, 8, 8])
    expected_cost = _compute_expected_cost(y_true, y_pred, tp_costs, tn_costs, fn_costs, fp_costs)
    avg_expected_cost = aec_score(y_true, y_pred, tp_costs=tp_costs,
                                  tn_costs=tn_costs, fn_costs=fn_costs, fp_costs=fp_costs)
    log_avg_expected_cost = log_aec_score(y_true, y_pred, tp_costs=tp_costs,
                                          tn_costs=tn_costs, fn_costs=fn_costs, fp_costs=fp_costs)
    assert np.isclose(expected_cost, np.array([1.0, 2.0, 4.0, 8.0])).all()
    assert avg_expected_cost == pytest.approx(3.75)
    assert log_avg_expected_cost == pytest.approx(1.0397207708399179)


def test_equal_combination_costs_two():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([.5, .5, .5, .5])
    tp_costs = np.array([1, 1, 1, 1])
    tn_costs = np.array([2, 2, 2, 2])
    fn_costs = np.array([4, 4, 4, 4])
    fp_costs = np.array([8, 8, 8, 8])
    expected_cost = _compute_expected_cost(y_true, y_pred, tp_costs, tn_costs, fn_costs, fp_costs)
    avg_expected_cost = aec_score(y_true, y_pred, tp_costs=tp_costs,
                                  tn_costs=tn_costs, fn_costs=fn_costs, fp_costs=fp_costs)
    log_avg_expected_cost = log_aec_score(y_true, y_pred, tp_costs=tp_costs,
                                          tn_costs=tn_costs, fn_costs=fn_costs, fp_costs=fp_costs)
    assert np.isclose(expected_cost, np.array([2.5, 5.0, 2.5, 5.0])).all()
    assert avg_expected_cost == pytest.approx(3.75)
    assert log_avg_expected_cost == pytest.approx(1.2628643221541278)


def test_equal_combination_costs_three():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1.0, 0.0, 0.0, 1.0])
    tp_costs = np.exp(np.array([1, 1, 1, 1]))
    tn_costs = np.exp(np.array([2, 2, 2, 2]))
    fn_costs = np.exp(np.array([4, 4, 4, 4]))
    fp_costs = np.exp(np.array([8, 8, 8, 8]))
    expected_cost = _compute_expected_cost(y_true, y_pred, tp_costs, tn_costs, fn_costs, fp_costs)
    avg_expected_cost = aec_score(y_true, y_pred, tp_costs=tp_costs,
                                  tn_costs=tn_costs, fn_costs=fn_costs, fp_costs=fp_costs)
    log_avg_expected_cost = log_aec_score(y_true, y_pred, tp_costs=tp_costs,
                                          tn_costs=tn_costs, fn_costs=fn_costs, fp_costs=fp_costs)
    assert np.isclose(expected_cost, np.exp(np.array([1.0, 2.0, 4.0, 8.0]))).all()
    assert avg_expected_cost == pytest.approx(761.4158687505656)
    assert log_avg_expected_cost == pytest.approx(3.75)


def test_equal_sparse_costs():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([.5, .5, .5, .5])
    tp_costs = np.array([1, 0, 1, 0])
    tn_costs = np.array([0, 2, 0, 2])
    fn_costs = np.array([4, 0, 4, 0])
    fp_costs = np.array([0, 8, 0, 8])
    expected_cost = _compute_expected_cost(y_true, y_pred, tp_costs, tn_costs, fn_costs, fp_costs)
    avg_expected_cost = aec_score(y_true, y_pred, tp_costs=tp_costs,
                                  tn_costs=tn_costs, fn_costs=fn_costs, fp_costs=fp_costs)
    log_avg_expected_cost = log_aec_score(y_true, y_pred, tp_costs=tp_costs,
                                          tn_costs=tn_costs, fn_costs=fn_costs, fp_costs=fp_costs)
    assert np.isclose(expected_cost, np.array([2.5, 5.0, 2.5, 5.0])).all()
    assert avg_expected_cost == pytest.approx(3.75)
    assert log_avg_expected_cost == pytest.approx(1.2628643221541278)


def test_equal_sparse_costs_two():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([.5, .5, .5, .5])
    tp_costs = np.array([2, 0, 4, 0])
    tn_costs = np.array([0, 2, 0, 8])
    fn_costs = np.array([4, 0, 16, 0])
    fp_costs = np.array([0, 8, 0, 32])
    expected_cost = _compute_expected_cost(y_true, y_pred, tp_costs, tn_costs, fn_costs, fp_costs)
    avg_expected_cost = aec_score(y_true, y_pred, tp_costs=tp_costs,
                                  tn_costs=tn_costs, fn_costs=fn_costs, fp_costs=fp_costs)
    log_avg_expected_cost = log_aec_score(y_true, y_pred, tp_costs=tp_costs,
                                          tn_costs=tn_costs, fn_costs=fn_costs, fp_costs=fp_costs)
    assert np.isclose(expected_cost, np.array([3.0, 5.0, 10.0, 20.0])).all()
    assert avg_expected_cost == pytest.approx(9.5)
    assert log_avg_expected_cost == pytest.approx(2.0015918919125615)


def test_no_costs_raises_value_error():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    with pytest.raises(ValueError):
        _compute_expected_cost(y_true, y_pred)
    with pytest.raises(ValueError):
        aec_score(y_true, y_pred)


def test_only_tp_costs():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1.0, 0.0, 0.0, 1.0])
    tp_costs = np.array([1, 1, 1, 1])

    expected_cost = _compute_expected_cost(y_true, y_pred, tp_costs=tp_costs)
    avg_expected_cost = aec_score(y_true, y_pred, tp_costs=tp_costs)
    log_avg_expected_cost = log_aec_score(y_true, y_pred, tp_costs=tp_costs)

    assert np.isclose(expected_cost, np.array([1.0, 0.0, 0.0, 0.0])).all()
    assert avg_expected_cost == pytest.approx(0.25)
    assert log_avg_expected_cost == pytest.approx(-27.032740041837865)


def test_only_fp_costs():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1.0, 0.0, 0.0, 1.0])
    fp_costs = np.array([1, 1, 1, 1])

    expected_cost = _compute_expected_cost(y_true, y_pred, fp_costs=fp_costs)
    avg_expected_cost = aec_score(y_true, y_pred, fp_costs=fp_costs)
    log_avg_expected_cost = log_aec_score(y_true, y_pred, fp_costs=fp_costs)

    assert np.isclose(expected_cost, np.array([0.0, 0.0, 0.0, 1.0])).all()
    assert avg_expected_cost == pytest.approx(0.25)
    assert log_avg_expected_cost == pytest.approx(-27.032740041837865)


def test_only_fn_costs():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1.0, 0.0, 0.0, 1.0])
    fn_costs = np.array([1, 1, 1, 1])

    expected_cost = _compute_expected_cost(y_true, y_pred, fn_costs=fn_costs)
    avg_expected_cost = aec_score(y_true, y_pred, fn_costs=fn_costs)
    log_avg_expected_cost = log_aec_score(y_true, y_pred, fn_costs=fn_costs)

    assert np.isclose(expected_cost, np.array([0.0, 0.0, 1.0, 0.0])).all()
    assert avg_expected_cost == pytest.approx(0.25)
    assert log_avg_expected_cost == pytest.approx(-27.032740041837865)


def test_only_tn_costs():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1.0, 0.0, 0.0, 1.0])
    tn_costs = np.array([1, 1, 1, 1])

    expected_cost = _compute_expected_cost(y_true, y_pred, tn_costs=tn_costs)
    avg_expected_cost = aec_score(y_true, y_pred, tn_costs=tn_costs)
    log_avg_expected_cost = log_aec_score(y_true, y_pred, tn_costs=tn_costs)

    assert np.isclose(expected_cost, np.array([0.0, 1.0, 0.0, 0.0])).all()
    assert avg_expected_cost == pytest.approx(0.25)
    assert log_avg_expected_cost == pytest.approx(-27.032740041837865)


def test_unequal_array_lengths():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1.0, 0.0, 0.0, 1.0])
    tp_costs = np.array([1, 1, 1])


    with pytest.raises(ValueError):
        _compute_expected_cost(y_true, y_pred, tp_costs=tp_costs)
    with pytest.raises(ValueError):
        aec_score(y_true, y_pred, tp_costs=tp_costs)
    with pytest.raises(ValueError):
        log_aec_score(y_true, y_pred, tp_costs=tp_costs)
    with pytest.raises(ValueError):
        _compute_expected_cost(y_true, y_pred, tn_costs=tp_costs)
    with pytest.raises(ValueError):
        aec_score(y_true, y_pred, tn_costs=tp_costs)
    with pytest.raises(ValueError):
        log_aec_score(y_true, y_pred, tn_costs=tp_costs)
    with pytest.raises(ValueError):
        _compute_expected_cost(y_true, y_pred, fn_costs=tp_costs)
    with pytest.raises(ValueError):
        aec_score(y_true, y_pred, fn_costs=tp_costs)
    with pytest.raises(ValueError):
        log_aec_score(y_true, y_pred, fn_costs=tp_costs)
    with pytest.raises(ValueError):
        _compute_expected_cost(y_true, y_pred, fp_costs=tp_costs)
    with pytest.raises(ValueError):
        aec_score(y_true, y_pred, fp_costs=tp_costs)
    with pytest.raises(ValueError):
        log_aec_score(y_true, y_pred, fp_costs=tp_costs)
