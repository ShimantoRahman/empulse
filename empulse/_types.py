from typing import Any

import numpy as np
from numpy._typing import _ArrayLikeFloat_co
from numpy.typing import NDArray

FloatNDArray = NDArray[np.bool_ | np.integer[Any] | np.floating[Any]]
FloatArrayLike = _ArrayLikeFloat_co
