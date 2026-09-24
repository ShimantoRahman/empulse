import itertools

import numpy as np
import pytest
from scipy.spatial import ConvexHull, QhullError
from sklearn.metrics import roc_curve

from empulse._types import FloatNDArray

try:
    from empulse.metrics._cy_convex_hull import convex_hull as cy_convex_hull
except ImportError:
    cy_convex_hull = None

pytestmark = pytest.mark.skipif(
    cy_convex_hull is None,
    reason='Implementations not importable in this environment',
)


def py_convex_hull(y_true: FloatNDArray, y_pred: FloatNDArray) -> tuple[FloatNDArray, FloatNDArray]:
    """
    Compute the convex hull points of the ROC curve.

    Parameters
    ----------
    y_true : 1D np.ndarray, shape=(n_samples,)
        Binary target values.

    y_pred : 1D np.ndarray, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Convex Hull points of the ROC curve (TPR, FPR)
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1, drop_intermediate=True)
    if fpr[0] != 0 or tpr[0] != 0:
        fpr = np.concatenate([[0], fpr])
        tpr = np.concatenate([[0], tpr])
    if fpr[-1] != 1 or tpr[-1] != 1:
        fpr = np.concatenate([fpr, [1]])
        tpr = np.concatenate([tpr, [1]])

    is_finite = np.isfinite(fpr) & np.isfinite(tpr)
    fpr = fpr[is_finite]
    tpr = tpr[is_finite]
    if fpr.shape[0] < 2:
        raise ValueError('Too few distinct predictions for ROCCH')

    points = np.c_[fpr, tpr]  # concatenate into matrix with two columns
    try:
        ind = ConvexHull(points).vertices  # indices of the points on the convex hull
    except QhullError:
        return np.array([0, 1]), np.array([0, 1])

    convex_hull_fpr = fpr[ind]
    convex_hull_tpr = tpr[ind]
    ind_upper_triangle = convex_hull_fpr < convex_hull_tpr  # only consider points above the 45° line
    convex_hull_fpr = np.concatenate([[0], convex_hull_fpr[ind_upper_triangle], [1]])
    convex_hull_tpr = np.concatenate([[0], convex_hull_tpr[ind_upper_triangle], [1]])
    ind = np.argsort(convex_hull_fpr)  # sort along the x-axis
    convex_hull_fpr = convex_hull_fpr[ind]
    convex_hull_tpr = convex_hull_tpr[ind]

    return convex_hull_tpr, convex_hull_fpr


def _make_case(rs, n=100, pos_ratio=0.3, tie_blocks=None):
    """Generate (y_true, y_score) with optional tie blocks for scores."""
    n_pos = int(n * pos_ratio)
    y_true = np.zeros(n, dtype=np.int32)
    if n_pos > 0:
        pos_idx = rs.choice(n, size=n_pos, replace=False)
        y_true[pos_idx] = 1

    y_score = rs.random(n)
    if tie_blocks:
        # Create some repeated-score segments to stress tie handling
        for start, end, val in tie_blocks:
            start = max(0, start)
            end = min(n, end)
            if start < end:
                y_score[start:end] = val
    return y_true.astype(np.int32), y_score.astype(np.float64)


@pytest.mark.parametrize(
    'y_true,y_score',
    [
        # Random continuous scores, balanced-ish
        _make_case(np.random.default_rng(0), n=200, pos_ratio=0.5),
        # Imbalanced, few positives
        _make_case(np.random.default_rng(1), n=300, pos_ratio=0.05),
        # Imbalanced, few negatives
        _make_case(np.random.default_rng(2), n=300, pos_ratio=0.95),
        # With many ties in the middle
        _make_case(
            np.random.default_rng(3),
            n=250,
            pos_ratio=0.4,
            tie_blocks=[(50, 120, 0.5), (120, 180, 0.7)],
        ),
        # All scores identical (forces minimal distinct thresholds)
        _make_case(
            np.random.default_rng(4),
            n=150,
            pos_ratio=0.3,
            tie_blocks=[(0, 150, 0.42)],
        ),
        # Alternating labels with random scores
        (
            np.array([i % 2 for i in range(200)], dtype=np.int32),
            np.random.default_rng(5).random(200).astype(np.float64),
        ),
    ],
)
def test_convex_hull_equivalence(y_true, y_score):
    # Ensure required dtypes for the Cython function
    y_true_c = np.ascontiguousarray(y_true, dtype=np.int32)
    y_score_c = np.ascontiguousarray(y_score, dtype=np.float64)

    # Cython implementation returns (tpr, fpr)
    tpr_cy, fpr_cy = cy_convex_hull(y_true_c, y_score_c)

    # Python implementation returns (tpr, fpr)
    tpr_py, fpr_py = py_convex_hull(y_true_c, y_score_c)

    # Basic sanity: shapes match
    assert tpr_cy.shape == tpr_py.shape
    assert fpr_cy.shape == fpr_py.shape

    # Values match within tolerance (floating-point ops, sorting, hull construction)
    assert np.allclose(tpr_cy, tpr_py, rtol=1e-10, atol=1e-12)
    assert np.allclose(fpr_cy, fpr_py, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('seed', range(20))
def test_convex_hull_from_counts_matches_convex_hull_of_the_samples(seed):
    """Samples grouped by score give exactly the hull of the samples, however the groups are split up."""
    from empulse.metrics._cy_convex_hull import convex_hull_from_counts

    rng = np.random.default_rng(seed)
    n_samples = int(rng.integers(20, 500))
    y_true = rng.integers(0, 2, n_samples).astype(np.int32)
    y_true[:2] = [0, 1]
    n_distinct = int(rng.integers(1, 40))
    y_score = rng.random(n_distinct)[rng.integers(0, n_distinct, n_samples)]  # many ties

    scores, group = np.unique(y_score, return_inverse=True)
    n_positive = np.bincount(group, weights=y_true).astype(np.int64)
    n_negative = np.bincount(group).astype(np.int64) - n_positive
    # The same score may also be spread over several groups, in any order.
    scores = np.concatenate([scores, scores])
    n_positive = np.concatenate([n_positive // 2, n_positive - n_positive // 2])
    n_negative = np.concatenate([n_negative - n_negative // 3, n_negative // 3])
    order = rng.permutation(scores.size)

    expected = cy_convex_hull(y_true, y_score)
    grouped = convex_hull_from_counts(scores[order], n_positive[order], n_negative[order])
    np.testing.assert_array_equal(grouped[0], expected[0])
    np.testing.assert_array_equal(grouped[1], expected[1])


def _samples_from_groups(n_positive, n_negative):
    """Expand groups into samples, the i-th group sharing the i-th highest score."""
    n_groups = len(n_positive)
    y_score = np.repeat(np.linspace(1.0, 0.0, n_groups), np.add(n_positive, n_negative))
    y_true = np.concatenate([
        np.r_[np.ones(p, dtype=np.int32), np.zeros(q, dtype=np.int32)]
        for p, q in zip(n_positive, n_negative, strict=True)
    ])
    return y_true, y_score


@pytest.mark.parametrize(
    'n_positive,n_negative,expected_tpr,expected_fpr',
    [
        # Perfect ranking: the hull climbs straight to (0, 1).
        ([1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 1], [0, 0, 1]),
        # Inverted ranking: the whole curve lies below the diagonal, which is all that remains.
        ([0, 0, 1, 1], [1, 1, 0, 0], [0, 1], [0, 1]),
        # One score for every sample: a single step from (0, 0) to (1, 1).
        ([3], [5], [0, 1], [0, 1]),
        # Curve (0, 1/3), (1/3, 1/3), (1/3, 2/3), (1/3, 1), (2/3, 1), (1, 1): the dip at (1/3, 1/3) and
        # (1/3, 2/3) lie below the hull, and (2/3, 1) lies on its top edge.
        ([1, 0, 1, 1, 0, 0], [0, 1, 0, 0, 1, 1], [0, 1 / 3, 1, 1], [0, 0, 1 / 3, 1]),
        # (1/4, 1/2) lies on the straight line from (0, 0) to (1/2, 1), so it is no vertex.
        ([2, 2, 0], [1, 1, 2], [0, 1, 1], [0, 1 / 2, 1]),
        # Points on the diagonal add nothing either.
        ([1, 1, 1], [2, 2, 2], [0, 1], [0, 1]),
    ],
)
def test_convex_hull_of_a_known_curve(n_positive, n_negative, expected_tpr, expected_fpr):
    from empulse.metrics._cy_convex_hull import convex_hull_from_counts

    y_true, y_score = _samples_from_groups(n_positive, n_negative)
    tpr, fpr = cy_convex_hull(y_true, y_score)
    np.testing.assert_allclose(tpr, expected_tpr, rtol=0, atol=1e-15)
    np.testing.assert_allclose(fpr, expected_fpr, rtol=0, atol=1e-15)

    scores = np.linspace(1.0, 0.0, len(n_positive))
    tpr, fpr = convex_hull_from_counts(
        scores, np.asarray(n_positive, dtype=np.int64), np.asarray(n_negative, dtype=np.int64)
    )
    np.testing.assert_allclose(tpr, expected_tpr, rtol=0, atol=1e-15)
    np.testing.assert_allclose(fpr, expected_fpr, rtol=0, atol=1e-15)


def test_convex_hull_keeps_every_corner_of_a_finely_sampled_curve():
    """
    Every point of a strictly concave curve is a vertex of its hull, however close together they lie.

    The k-th of m groups holds m - k positives and one negative, so the slope of the curve drops by
    the same amount at every point. With m = 300 the curve turns by about 7e-8 at each point, which a
    fixed tolerance on the turn (as the hull used to have) mistakes for a straight line.
    """
    from empulse.metrics._cy_convex_hull import convex_hull_from_counts

    m = 300
    n_positive = np.arange(m, 0, -1, dtype=np.int64)
    n_negative = np.ones(m, dtype=np.int64)
    expected_tpr = np.r_[0, np.cumsum(n_positive)] / n_positive.sum()
    expected_fpr = np.arange(m + 1) / m

    y_true, y_score = _samples_from_groups(n_positive, n_negative)
    for tpr, fpr in (
        cy_convex_hull(y_true, y_score),
        convex_hull_from_counts(np.linspace(1.0, 0.0, m), n_positive, n_negative),
    ):
        np.testing.assert_array_equal(tpr, expected_tpr)
        np.testing.assert_array_equal(fpr, expected_fpr)


def _roc_counts(y_true, y_score):
    """The ROC curve as the negatives and positives ranked at or above each distinct score, from (0, 0)."""
    order = np.argsort(-y_score, kind='stable')
    ranked_true, ranked_score = y_true[order], y_score[order]
    ends = np.r_[np.nonzero(ranked_score[1:] != ranked_score[:-1])[0], y_true.size - 1]
    n_positive = np.cumsum(ranked_true)[ends]
    return np.r_[0, ends + 1 - n_positive], np.r_[0, n_positive]


@pytest.mark.parametrize('seed', range(12))
def test_convex_hull_is_exactly_the_upper_hull_of_the_roc_curve(seed):
    """
    The hull's vertices are points of the ROC curve, no point of the curve lies above it, and it turns at every vertex.

    Checked with exact integer arithmetic on the counts, on samples large enough that the points of
    the curve lie very close together.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(rng.choice([50, 5_000, 50_000]))
    y_true = rng.integers(0, 2, n_samples).astype(np.int32)
    y_true[:2] = [0, 1]
    y_score = rng.normal(rng.uniform(0, 2) * y_true, 1)
    if seed % 3 == 1:
        y_score = np.round(y_score, 1)

    tpr, fpr = cy_convex_hull(y_true, y_score)
    n_negative, n_positive = _roc_counts(y_true, y_score)
    total_negative, total_positive = n_negative[-1], n_positive[-1]

    # Each vertex is a point of the curve, in order from (0, 0) to (1, 1).
    vertices = np.array([
        np.flatnonzero((n_positive / total_positive == t) & (n_negative / total_negative == f))[0]
        for t, f in zip(tpr, fpr, strict=True)
    ])
    assert vertices[0] == 0
    assert vertices[-1] == n_negative.size - 1
    assert np.all(np.diff(vertices) > 0)

    def cross(a, b, c):
        return (n_negative[b] - n_negative[a]) * (n_positive[c] - n_positive[a]) - (n_positive[b] - n_positive[a]) * (
            n_negative[c] - n_negative[a]
        )

    for start, end in itertools.pairwise(vertices):
        assert np.all(cross(start, end, np.arange(start, end + 1)) <= 0), 'a point of the curve lies above the hull'
    for previous, vertex, following in zip(vertices[:-2], vertices[1:-1], vertices[2:], strict=True):
        assert cross(previous, vertex, following) < 0, 'the hull does not turn at a vertex'


@pytest.mark.parametrize(
    ('y_true', 'y_score', 'match'),
    [
        (np.array([], dtype=np.int32), np.array([], dtype=np.float64), 'at least one sample'),
        (np.array([0, 1], dtype=np.int32), np.array([0.5], dtype=np.float64), 'same length'),
        (np.array([0], dtype=np.int32), np.array([0.2, 0.5], dtype=np.float64), 'same length'),
    ],
)
def test_convex_hull_rejects_invalid_samples(y_true, y_score, match):
    with pytest.raises(ValueError, match=match):
        cy_convex_hull(y_true, y_score)


@pytest.mark.parametrize(
    ('y_score', 'n_positive', 'n_negative', 'match'),
    [
        ([], [], [], 'at least one sample'),
        ([0.2, 0.5], [0, 0], [0, 0], 'at least one sample'),
        ([0.2, 0.5], [1], [1], 'same length'),
        ([0.2, 0.5], [1, 2], [1, 2, 3], 'same length'),
        ([0.2, 0.5], [1, -1], [1, 2], 'cannot be negative'),
    ],
)
def test_convex_hull_from_counts_rejects_invalid_groups(y_score, n_positive, n_negative, match):
    from empulse.metrics._cy_convex_hull import convex_hull_from_counts

    with pytest.raises(ValueError, match=match):
        convex_hull_from_counts(
            np.array(y_score, dtype=np.float64),
            np.array(n_positive, dtype=np.int64),
            np.array(n_negative, dtype=np.int64),
        )
