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
    tpr_py, fpr_py = py_convex_hull(y_true_c, y_score_c, expand_dims=False)

    # Basic sanity: shapes match
    assert tpr_cy.shape == tpr_py.shape
    assert fpr_cy.shape == fpr_py.shape

    # Values match within tolerance (floating-point ops, sorting, hull construction)
    assert np.allclose(tpr_cy, tpr_py, rtol=1e-10, atol=1e-12)
    assert np.allclose(fpr_cy, fpr_py, rtol=1e-10, atol=1e-12)
