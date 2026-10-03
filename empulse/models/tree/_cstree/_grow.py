"""Resolve a tree's hyperparameters once per fit, and grow trees with them."""

import math
import numbers
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.utils._param_validation import Interval, RealNotInt, StrOptions

from ...._types import FloatNDArray
from ._costs import criterion_kind
from ._splitter import Splitter
from ._tree import CostTree, build_tree, prune_tree

# The largest value of scikit-learn's tree seeds, 2^31 - 1.
RAND_R_MAX = np.iinfo(np.int32).max

TREE_PARAM_CONSTRAINTS: dict[str, list[Any]] = {
    'criterion': [StrOptions({'cost', 'gini', 'entropy', 'log_loss'})],
    'max_depth': [Interval(Integral, 1, None, closed='left'), None],
    'min_samples_split': [
        Interval(Integral, 2, None, closed='left'),
        Interval(RealNotInt, 0.0, 1.0, closed='right'),
    ],
    'min_samples_leaf': [
        Interval(Integral, 1, None, closed='left'),
        Interval(RealNotInt, 0.0, 1.0, closed='neither'),
    ],
    'min_weight_fraction_leaf': [Interval(Real, 0.0, 0.5, closed='both')],
    'max_features': [
        Interval(Integral, 1, None, closed='left'),
        Interval(RealNotInt, 0.0, 1.0, closed='right'),
        StrOptions({'sqrt', 'log2'}),
        None,
    ],
    'random_state': ['random_state'],
    'max_leaf_nodes': [Interval(Integral, 2, None, closed='left'), None],
    'min_impurity_decrease': [Interval(Real, 0.0, None, closed='left'), None],
    'ccp_alpha': [Interval(Real, 0.0, None, closed='left')],
}


@dataclass(frozen=True)
class TreeParams:
    """The hyperparameters of a tree, resolved against the training data."""

    criterion: int
    random_splits: bool
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int
    min_weight_fraction_leaf: float
    max_features: int
    max_leaf_nodes: int
    min_impurity_decrease: float
    ccp_alpha: float

    @property
    def cost_bound(self) -> bool:
        """Whether nodes no split could improve enough become leaves without a split search."""
        return self.criterion == criterion_kind('cost') and self.min_impurity_decrease > 0

    def min_weight_leaf(self, n_samples: int, sample_weight: FloatNDArray | None) -> float:
        """Return the least total weight of a leaf."""
        if sample_weight is None:
            return self.min_weight_fraction_leaf * n_samples
        return self.min_weight_fraction_leaf * float(np.sum(sample_weight))


def resolve_tree_params(
    *,
    n_samples: int,
    n_features: int,
    criterion: str,
    splitter: str,
    max_depth: int | None,
    min_samples_split: float,
    min_samples_leaf: float,
    min_weight_fraction_leaf: float,
    max_features: str | float | None,
    max_leaf_nodes: int | None,
    min_impurity_decrease: float,
    ccp_alpha: float,
) -> TreeParams:
    """Resolve fractions, presets and ``None`` the way scikit-learn's ``BaseDecisionTree`` does."""
    if isinstance(min_samples_leaf, numbers.Integral):
        min_samples_leaf_ = int(min_samples_leaf)
    else:
        min_samples_leaf_ = math.ceil(min_samples_leaf * n_samples)

    if isinstance(min_samples_split, numbers.Integral):
        min_samples_split_ = int(min_samples_split)
    else:
        min_samples_split_ = max(2, math.ceil(min_samples_split * n_samples))
    min_samples_split_ = max(min_samples_split_, 2 * min_samples_leaf_)

    if max_features == 'sqrt':
        max_features_ = max(1, int(np.sqrt(n_features)))
    elif max_features == 'log2':
        max_features_ = max(1, int(np.log2(n_features)))
    elif max_features is None:
        max_features_ = n_features
    elif isinstance(max_features, numbers.Integral):
        max_features_ = int(max_features)
    else:
        fraction = float(max_features)
        max_features_ = max(1, int(fraction * n_features)) if fraction > 0.0 else 0

    return TreeParams(
        criterion=criterion_kind(criterion),
        random_splits=splitter == 'random',
        max_depth=np.iinfo(np.int32).max if max_depth is None else int(max_depth),
        min_samples_split=min_samples_split_,
        min_samples_leaf=min_samples_leaf_,
        min_weight_fraction_leaf=float(min_weight_fraction_leaf),
        max_features=max_features_,
        max_leaf_nodes=-1 if max_leaf_nodes is None else int(max_leaf_nodes),
        min_impurity_decrease=float(min_impurity_decrease),
        ccp_alpha=float(ccp_alpha),
    )


def grow_tree(
    params: TreeParams,
    X: FloatNDArray,
    records: FloatNDArray,
    sample_weight: FloatNDArray | None,
    random_state: np.random.RandomState,
    missing_mask: NDArray[np.uint8] | None,
) -> CostTree:
    """
    Grow one tree.

    Parameters
    ----------
    params : TreeParams
        The resolved hyperparameters.
    X : ndarray of shape (n_samples, n_features), dtype float32, Fortran-ordered
        The training samples. Forests share one copy between all their trees.
    records : ndarray of shape (n_samples, 4)
        The per-sample cost records of :func:`cost_records`, also shared.
    sample_weight : ndarray of shape (n_samples,) or None
        The weight of each sample in this tree. Samples weighing zero take no part.
    random_state : RandomState
        Draws the seed of the tree's feature and threshold draws.
    missing_mask : ndarray of shape (n_features,), dtype uint8, or None
        The :func:`missing_feature_mask` of ``X``.
    """
    n_samples, n_features = X.shape
    splitter = Splitter(
        X,
        records,
        sample_weight,
        params.criterion,
        params.random_splits,
        params.max_features,
        params.min_samples_leaf,
        params.min_weight_leaf(n_samples, sample_weight),
        random_state.randint(0, RAND_R_MAX),
        params.cost_bound,
        missing_mask,
    )
    tree = CostTree(n_features)
    build_tree(
        tree,
        splitter,
        params.min_samples_split,
        params.min_samples_leaf,
        params.min_weight_leaf(n_samples, sample_weight),
        params.max_depth,
        params.max_leaf_nodes,
        params.min_impurity_decrease,
        params.cost_bound,
        cost_scale(records),
    )
    if params.ccp_alpha > 0.0:
        tree = prune_tree(tree, params.ccp_alpha)
    return tree


def cost_scale(records: FloatNDArray) -> float:
    """Return the mean over the samples of the costs of predicting them positive and negative, or 1 if zero."""
    scale = float(np.mean(np.abs(records[:, 0]) + np.abs(records[:, 1])))
    return scale if scale > 0.0 else 1.0


def as_float32(X: Any, *, fortran: bool = False) -> FloatNDArray:
    """Return ``X`` as float32: Fortran-ordered for growing a tree, C-ordered for traversing one."""
    if fortran:
        return np.asfortranarray(X, dtype=np.float32)
    return np.ascontiguousarray(X, dtype=np.float32)


def missing_feature_mask(X: FloatNDArray) -> NDArray[np.uint8] | None:
    """Return whether each feature of ``X`` has a missing (NaN) value, or ``None`` when none has."""
    with np.errstate(over='ignore', invalid='ignore'):
        # A sum overflowing to +inf and -inf also gives NaN; the exact check below sorts that out.
        if not np.isnan(np.sum(X)):
            return None
    mask = np.asarray(np.isnan(X).any(axis=0), dtype=np.uint8)
    if not mask.any():
        return None
    return mask
