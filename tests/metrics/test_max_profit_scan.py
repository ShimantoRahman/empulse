"""
The compiled deterministic maximum profit (`max_profit_scan`) against the profit evaluated at every threshold.

The reference is the NumPy path the deterministic MaxProfit score falls back on: the profit function
evaluated at every point of the ROC curve, maximized with ``np.argmax``, and the threshold of that
point found by :func:`~empulse.metrics.classification_threshold`.
"""

import numpy as np
import pytest
import sympy

from empulse.metrics import classification_threshold
from empulse.metrics._cy_convex_hull import max_profit_scan
from empulse.metrics.metric._compile import _safe_lambdify
from empulse.metrics.metric.strategies.max_profit_strategy.deterministic import _calculate_profits_deterministic
from empulse.metrics.metric.strategies.max_profit_strategy.max_profit_strategy import _build_profit_function

_CLASS_SYMBOLS = sympy.symbols('tp tn fp fn')
_PROFIT_FUNCTION = _build_profit_function(*_CLASS_SYMBOLS)
_CALCULATE_PROFIT = _safe_lambdify(_PROFIT_FUNCTION)


def _reference(y_true, y_score, class_values):
    """Profit, rate and threshold, from the profit evaluated at every point of the ROC curve."""
    parameters = dict(zip(('tp', 'tn', 'fp', 'fn'), class_values, strict=True))
    profits, tprs, fprs, pi0, pi1 = _calculate_profits_deterministic(
        y_true, y_score, _CALCULATE_PROFIT, _PROFIT_FUNCTION, **parameters
    )
    best = int(np.argmax(profits))
    rate = float(tprs[best] * pi0 + fprs[best] * pi1)
    return float(profits[best]), rate, float(classification_threshold(y_true, y_score, rate))


# Below and above the number of samples the convex hull sorts in C++, and many-way ties.
@pytest.mark.parametrize('n_samples', [1, 2, 7, 256, 257, 3000])
@pytest.mark.parametrize('positive_fraction', [0.0, 0.3, 1.0])
@pytest.mark.parametrize('tied_scores', [False, True], ids=['distinct', 'tied'])
def test_matches_the_profit_at_every_threshold(seeded_rng, n_samples, positive_fraction, tied_scores):
    for _ in range(20):
        y_true = (seeded_rng.random(n_samples) < positive_fraction).astype(np.int32)
        y_score = seeded_rng.integers(0, 5, n_samples) / 4.0 if tied_scores else seeded_rng.random(n_samples)
        # Continuous values of either sign, so no two thresholds tie on profit.
        class_values = seeded_rng.normal(0, 10, 4)

        profit, rate, threshold = max_profit_scan(y_true, y_score, *class_values)
        expected_profit, expected_rate, expected_threshold = _reference(y_true, y_score, class_values)

        assert profit == pytest.approx(expected_profit, rel=1e-12, abs=1e-12)
        assert rate == pytest.approx(expected_rate, abs=1e-12)
        assert threshold == expected_threshold


def test_equal_profits_resolve_to_the_fewest_samples_targeted():
    # tp=3, fn=-3: targeting a positive gains 3 and loses the 3 of missing it, so with only
    # positives every threshold has the same profit.
    y_true = np.ones(5, dtype=np.int32)
    y_score = np.array([0.1, 0.4, 0.2, 0.9, 0.5])

    profit, rate, threshold = max_profit_scan(y_true, y_score, 3.0, 1.0, 18.0, -3.0)

    assert profit == 3.0
    assert rate == 0.0
    assert threshold == 1.0


def test_profit_is_the_average_over_samples():
    y_true = np.array([1, 0, 1, 0], dtype=np.int32)
    y_score = np.array([0.9, 0.8, 0.7, 0.1])

    # Targeting the top sample (a positive) gains tp=10 and misses one positive (fn=4), while both
    # negatives are correctly rejected (tn=0): (10 - 4) / 4.
    profit, rate, threshold = max_profit_scan(y_true, y_score, 10.0, 0.0, 20.0, 4.0)

    assert profit == pytest.approx(1.5)
    assert rate == pytest.approx(0.25)
    assert threshold == 0.9


@pytest.mark.parametrize(
    ('y_true', 'y_score', 'match'),
    [
        (np.array([1, 0], dtype=np.int32), np.array([0.5]), 'same length'),
        (np.array([], dtype=np.int32), np.array([]), 'at least one sample'),
    ],
)
def test_rejects_invalid_samples(y_true, y_score, match):
    with pytest.raises(ValueError, match=match):
        max_profit_scan(y_true, y_score, 1.0, 0.0, 1.0, 1.0)
