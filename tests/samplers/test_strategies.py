import numpy as np
import pytest

from empulse.samplers._strategies import _independent_weights


def test_independent_weights():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 1, 0, 1])
    protected_attr = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    weights = _independent_weights(y_true, protected_attr)
    unpriviledged_neg, priviledged_neg, unpriviledged_pos, priviledged_pos = weights.flatten()
    assert pytest.approx(unpriviledged_neg, abs=1e-3) == 0.666666
    assert pytest.approx(priviledged_neg, abs=1e-3) == 2
    assert pytest.approx(unpriviledged_pos, abs=1e-3) == 1.5
    assert pytest.approx(priviledged_pos, abs=1e-3) == 0.75
