import numpy as np
from scipy.spatial import ConvexHull, QhullError

from sklearn.metrics import roc_curve


def _compute_convex_hull(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        expand_dims: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the convex hull points of the ROC curve.

    Parameters
    ----------
    y_true : 1D np.ndarray, shape=(n_samples,)
        Binary target values.

    y_pred : 1D np.ndarray, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    expand_dims : bool, default=False
        Whether to expand the dimensions of the convex hull points to (n_points, 1).

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
        raise ValueError("Too few distinct predictions for ROCCH")

    points = np.c_[fpr, tpr]  # concatenate into matrix with two columns
    try:
        ind = ConvexHull(points).vertices  # indices of the points on the convex hull
    except QhullError:
        return np.array([0, 1]), np.array([0, 1])

    convex_hull_fpr = fpr[ind]
    convex_hull_tpr = tpr[ind]
    ind_upper_triangle = convex_hull_fpr < convex_hull_tpr  # only consider points above the 45° line
    convex_hull_fpr = np.concatenate([[0], convex_hull_fpr[ind_upper_triangle], [1]])  # type: ignore
    convex_hull_tpr = np.concatenate([[0], convex_hull_tpr[ind_upper_triangle], [1]])  # type: ignore
    ind = np.argsort(convex_hull_fpr)  # sort along the x-axis
    convex_hull_fpr = convex_hull_fpr[ind]
    convex_hull_tpr = convex_hull_tpr[ind]

    if expand_dims:
        convex_hull_tpr = np.expand_dims(convex_hull_tpr, axis=1)
        convex_hull_fpr = np.expand_dims(convex_hull_fpr, axis=1)

    return convex_hull_tpr, convex_hull_fpr
