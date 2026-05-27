import numpy as np
import numpy.typing as npt

def cy_logit_loss_gradient(
    weights: npt.NDArray[np.float64],
    features: npt.NDArray[np.float64],
    grad_const: npt.NDArray[np.float64],
    loss_const1: npt.NDArray[np.float64],
    loss_const2: npt.NDArray[np.float64],
    C: float = ...,
    l1_ratio: float = ...,
    fit_intercept: bool = ...,
    soft_threshold: bool = ...,
) -> tuple[float, npt.NDArray[np.float64]]: ...
def cy_logit_loss(
    weights: npt.NDArray[np.float64],
    features: npt.NDArray[np.float64],
    loss_const1: npt.NDArray[np.float64],
    loss_const2: npt.NDArray[np.float64],
    C: float = ...,
    l1_ratio: float = ...,
    fit_intercept: bool = ...,
    soft_threshold: bool = ...,
) -> float: ...
def cy_logit_gradient(
    weights: npt.NDArray[np.float64],
    features: npt.NDArray[np.float64],
    grad_const: npt.NDArray[np.float64],
    C: float = ...,
    l1_ratio: float = ...,
    fit_intercept: bool = ...,
    soft_threshold: bool = ...,
) -> npt.NDArray[np.float64]: ...
def cy_boost_grad_hess(
    y_true: npt.NDArray[np.float64],
    y_score: npt.NDArray[np.float64],
    grad_const: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
