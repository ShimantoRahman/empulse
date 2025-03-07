from collections.abc import Callable
from typing import Any

import numpy as np
from numpy._typing import _ArrayLikeFloat_co, _ArrayLikeInt_co
from numpy.typing import NDArray
from sklearn.utils._param_validation import _Constraint

FloatNDArray = NDArray[np.bool_ | np.integer[Any] | np.floating[Any]]
IntNDArray = NDArray[np.bool_ | np.integer[Any]]
FloatArrayLike = _ArrayLikeFloat_co
IntArrayLike = _ArrayLikeInt_co
ParameterConstraint = dict[str, list[_Constraint | str | type[Callable[[Any], Any]] | None]]
