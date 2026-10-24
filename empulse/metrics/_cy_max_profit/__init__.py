try:  # ruff: ignore[non-empty-init-module]
    from .piecewise import (
        Distribution,
        _expected_max_profit_of_groups_address,
        expected_max_profit,
        expected_max_profit_from_counts,
        partial_moments,
    )
except ImportError:  # pragma: no cover - exercised only when the extension is unbuilt or cannot link
    # Unlike the convex hull, this extension has a fallback: without it, the piecewise score
    # integrates every distribution numerically, and the analytic gradients are not available. It
    # links against SciPy's cython_special at import time, so a SciPy release that changed one of the
    # functions it uses would otherwise make empulse.metrics unimportable rather than only slower.
    Distribution = None
    expected_max_profit = None
    expected_max_profit_from_counts = None
    partial_moments = None
    _expected_max_profit_of_groups_address = None

__all__ = [
    'Distribution',
    '_expected_max_profit_of_groups_address',
    'expected_max_profit',
    'expected_max_profit_from_counts',
    'partial_moments',
]
