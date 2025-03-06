from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import sympy
from numpy.typing import NDArray


class MetricFn(Protocol):  # noqa: D101
    def __call__(self, y_true: NDArray, y_score: NDArray, **kwargs: Any) -> float: ...  # noqa: D102


class LogitObjective(Protocol):  # noqa: D101
    def __call__(self, x: NDArray, y_true: NDArray, y_score: NDArray, **kwargs: Any) -> NDArray[np.floating]: ...  # noqa: D102


class BoostObjective(Protocol):  # noqa: D101
    def __call__(  # noqa: D102
        self, y_true: NDArray, y_score: NDArray, **kwargs: Any
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]: ...


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
