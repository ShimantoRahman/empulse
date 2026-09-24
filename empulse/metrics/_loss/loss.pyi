import numpy as np
import numpy.typing as npt

def cy_logit_loss_gradient(
    weights: npt.NDArray[np.float64],
    features: npt.NDArray[np.float64],
    grad_const: npt.NDArray[np.float64],
    loss_const1: npt.NDArray[np.float64],
    loss_const2: npt.NDArray[np.float64],
    l1_weight: float = ...,
    l2_weight: float = ...,
    start_coef: int = ...,
) -> tuple[float, npt.NDArray[np.float64]]: ...
def cy_logit_loss(
    weights: npt.NDArray[np.float64],
    features: npt.NDArray[np.float64],
    loss_const1: npt.NDArray[np.float64],
    loss_const2: npt.NDArray[np.float64],
    l1_weight: float = ...,
    l2_weight: float = ...,
    start_coef: int = ...,
) -> float: ...
def cy_logit_gradient(
    weights: npt.NDArray[np.float64],
    features: npt.NDArray[np.float64],
    grad_const: npt.NDArray[np.float64],
    l1_weight: float = ...,
    l2_weight: float = ...,
    start_coef: int = ...,
) -> npt.NDArray[np.float64]: ...
def cy_boost_grad_hess(
    y_true: npt.NDArray[np.float64],
    y_score: npt.NDArray[np.float64],
    grad_const: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def cy_log_cost_loss_gradient(
    weights: npt.NDArray[np.float64],
    features: npt.NDArray[np.float64],
    loss_const1: npt.NDArray[np.float64],
    loss_const2: npt.NDArray[np.float64],
    l1_weight: float = ...,
    l2_weight: float = ...,
    start_coef: int = ...,
) -> tuple[float, npt.NDArray[np.float64]]: ...
def cy_log_cost_loss(
    weights: npt.NDArray[np.float64],
    features: npt.NDArray[np.float64],
    loss_const1: npt.NDArray[np.float64],
    loss_const2: npt.NDArray[np.float64],
    l1_weight: float = ...,
    l2_weight: float = ...,
    start_coef: int = ...,
) -> float: ...
def cy_log_cost_gradient(
    weights: npt.NDArray[np.float64],
    features: npt.NDArray[np.float64],
    loss_const1: npt.NDArray[np.float64],
    loss_const2: npt.NDArray[np.float64],
    l1_weight: float = ...,
    l2_weight: float = ...,
    start_coef: int = ...,
) -> npt.NDArray[np.float64]: ...
def cy_log_cost_boost_grad_hess(
    y_score: npt.NDArray[np.float64],
    loss_const1: npt.NDArray[np.float64],
    loss_const2: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
