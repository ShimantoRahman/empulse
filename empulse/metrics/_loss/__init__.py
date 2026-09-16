try:  # ruff: ignore[non-empty-init-module]
    from .loss import cy_boost_grad_hess, cy_logit_gradient, cy_logit_loss, cy_logit_loss_gradient
except ImportError as exc:  # pragma: no cover - exercised only when the extension is unbuilt
    raise ImportError(
        'empulse.metrics._loss.loss could not be imported. This compiled Cython extension is '
        'required -- empulse.metrics has no pure-Python fallback for it. Build it with '
        '`just compile` (requires MSVC on PATH on Windows).'
    ) from exc

__all__ = [
    'cy_boost_grad_hess',
    'cy_logit_gradient',
    'cy_logit_loss',
    'cy_logit_loss_gradient',
]
