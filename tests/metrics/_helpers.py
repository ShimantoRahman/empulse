"""
Helpers shared by several metric test modules.

``RANKING_*`` and :func:`logit_and_count_evaluations` were duplicated verbatim between
``test_assumption_symbols.py`` and ``test_expression_distribution_arguments.py``;
:func:`brute_force_emp` is the reference both the piecewise and the sampling tests compare against.
"""

import numpy as np
from scipy.integrate import trapezoid

from empulse.metrics.metric.strategies.max_profit_strategy.common import _convex_hull

# A 200-sample ranking with a weak signal, and a design matrix to evaluate logit objectives on.
_RNG = np.random.default_rng(0)
RANKING_Y_TRUE = (_RNG.random(200) < 0.3).astype(int)
RANKING_Y_SCORE = _RNG.random(200) + 0.4 * RANKING_Y_TRUE * _RNG.random(200)
RANKING_FEATURES = np.hstack((np.ones((200, 1)), _RNG.normal(size=(200, 3))))


def logit_and_count_evaluations(metric, parameters):
    """Everything a MaxProfit metric computes from a ranking: score, rate, count loss and logit loss."""
    unique_scores, groups = np.unique(RANKING_Y_SCORE, return_inverse=True)
    n_positive = np.bincount(groups, weights=RANKING_Y_TRUE).astype(np.int64)
    n_negative = np.bincount(groups, weights=1 - RANKING_Y_TRUE).astype(np.int64)
    objective = metric._logit_value_objective(
        features=RANKING_FEATURES, y_true=RANKING_Y_TRUE, C=1.0, l1_ratio=1.0, fit_intercept=True, **parameters
    )
    return [
        metric(RANKING_Y_TRUE, RANKING_Y_SCORE, **parameters),
        metric.optimal_rate(RANKING_Y_TRUE, RANKING_Y_SCORE, **parameters),
        metric._prepare_count_loss(**parameters)(unique_scores, n_positive, n_negative),
        objective.logit_loss(np.full(RANKING_FEATURES.shape[1], 0.1)),
    ]


def brute_force_emp(y_true, y_score, benefit_of, cost, pdf, lower, upper, n_points=400_001):
    """
    Independent reference: integrate ``max_t P(t, x) * h(x)`` on a dense grid.

    Deliberately shares nothing with the package beyond the convex hull -- no piecewise regions, no
    partial moments, no root finding -- so it can only agree with the implementation by both being
    right.
    """
    positive_class_prior = float(np.mean(y_true))
    negative_class_prior = 1.0 - positive_class_prior
    tprs, fprs = _convex_hull(y_true, y_score)

    x = np.linspace(lower, upper, n_points)
    profit = positive_class_prior * np.outer(tprs, benefit_of(x)) - negative_class_prior * np.outer(
        fprs, np.full_like(x, cost)
    )
    return float(trapezoid(profit.max(axis=0) * pdf(x), x))
