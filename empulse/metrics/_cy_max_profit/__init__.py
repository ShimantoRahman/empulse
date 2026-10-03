try:  # ruff: ignore[non-empty-init-module]
    from .piecewise import (
        Distribution,
        _expected_max_profit_of_groups_address,
        expected_max_profit,
        expected_max_profit_from_counts,
    )
except ImportError:  # pragma: no cover - exercised only when the extension is unbuilt or cannot link
    # Unlike the convex hull, this extension has a pure-Python fallback (the piecewise score
    # classes' own integration), which is used when it is missing. It links against SciPy's
    # cython_special at import time, so a SciPy release that changed one of the functions it uses
    # would otherwise make empulse.metrics unimportable rather than only slower.
    Distribution = None
    expected_max_profit = None
    expected_max_profit_from_counts = None
    _expected_max_profit_of_groups_address = None

__all__ = [
    'Distribution',
    '_expected_max_profit_of_groups_address',
    'expected_max_profit',
    'expected_max_profit_from_counts',
]
