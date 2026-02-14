from collections.abc import Iterable
from typing import Any

import numpy as np
import sympy

from ....._types import FloatNDArray, IntNDArray
from ...._cy_convex_hull import convex_hull


def _convex_hull(y_true: IntNDArray, y_score: FloatNDArray) -> tuple[IntNDArray, FloatNDArray]:
    return convex_hull(y_true.astype(np.int32), y_score.astype(np.float64))  # type: ignore[no-any-return]


def extract_distribution_parameters(
    parameters: dict[str, Any], distribution_args: Iterable[sympy.Symbol]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract the distribution parameters from the other parameters."""
    distribution_parameters = {
        str(key): parameters.pop(str(key)) for key in distribution_args if str(key) in parameters
    }
    return distribution_parameters, parameters
