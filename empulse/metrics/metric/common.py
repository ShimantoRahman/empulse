from collections.abc import Callable
from typing import Any, Protocol

import sympy
from numpy.typing import NDArray


class MetricFn(Protocol):  # noqa: D101
    def __call__(self, y_true: NDArray, y_score: NDArray, **kwargs: Any) -> float: ...  # noqa: D102


def _check_parameters(*parameters: str | sympy.Expr) -> Callable[[MetricFn], MetricFn]:
    """
    Check if all parameters are provided.

    In particular:
        - deterministic parameters
        - distribution parameters of stochastic variables
    """

    def decorator(func: MetricFn) -> MetricFn:
        def wrapper(y_true: NDArray, y_score: NDArray, **kwargs: Any) -> float:
            for value in parameters:
                if str(value) not in kwargs:
                    raise ValueError(f'Metric expected a value for {value}, did not receive it.')
            return func(y_true, y_score, **kwargs)

        return wrapper

    return decorator
