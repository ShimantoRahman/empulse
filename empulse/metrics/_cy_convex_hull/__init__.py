try:  # ruff: ignore[non-empty-init-module]
    from .convex_hull import convex_hull
except ImportError as exc:  # pragma: no cover - exercised only when the extension is unbuilt
    raise ImportError(
        'empulse.metrics._cy_convex_hull.convex_hull could not be imported. This compiled Cython '
        'extension is required -- empulse.metrics has no pure-Python fallback for it. Build it with '
        '`just compile` (requires MSVC on PATH on Windows).'
    ) from exc

__all__ = ['convex_hull']
