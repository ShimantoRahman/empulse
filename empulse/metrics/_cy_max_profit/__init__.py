try:  # ruff: ignore[non-empty-init-module]
    from .piecewise import Distribution, expected_max_profit
except ImportError:  # pragma: no cover - exercised only when the extension is unbuilt or cannot link
    # Unlike the convex hull, this extension has a pure-Python fallback (the piecewise score
    # classes' own integration), which is used when it is missing. It links against SciPy's
    # cython_special at import time, so a SciPy release that changed one of the functions it uses
    # would otherwise make empulse.metrics unimportable rather than only slower.
    Distribution = None
    expected_max_profit = None

__all__ = ['Distribution', 'expected_max_profit']
