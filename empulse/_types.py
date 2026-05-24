from collections.abc import Callable
from typing import Any

import numpy as np
from numpy._typing import _ArrayLikeFloat_co, _ArrayLikeInt_co
from numpy.typing import NDArray
from sklearn.utils._param_validation import _Constraint

FloatNDArray = NDArray[np.bool_ | np.integer[Any] | np.floating[Any]]
Float64Array = NDArray[np.float64]
Float32Array = NDArray[np.float32]
IntNDArray = NDArray[np.bool_ | np.integer[Any]]
Int64Array = NDArray[np.int64]
Int32Array = NDArray[np.int32]
FloatArrayLike = _ArrayLikeFloat_co
IntArrayLike = _ArrayLikeInt_co
ParameterConstraint = dict[str, list[_Constraint | str | type[Callable[[Any], Any]] | None]]
