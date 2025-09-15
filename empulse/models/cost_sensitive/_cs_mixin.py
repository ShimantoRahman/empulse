import warnings
from numbers import Real
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._metadata_requests import RequestMethod

from ..._common import Parameter
from ..._types import FloatArrayLike, FloatNDArray
from ...metrics import Metric


class CostSensitiveMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.__post_init__()
        super().__init__(*args, **kwargs)

    def __post_init__(self) -> None:
        # Allow passing costs accepted by the metric loss through metadata routing
        if isinstance(self._get_metric_loss(), Metric):
            self.__class__.set_fit_request = RequestMethod(  # type: ignore[attr-defined]
                'fit',
                sorted(self.__class__._get_default_requests().fit.requests.keys() | self.loss._all_symbols),  # type: ignore[attr-defined]
            )

    @overload
    def _check_costs(
        self,
        *,
        tp_cost: FloatArrayLike | float | Parameter,
        tn_cost: FloatArrayLike | float | Parameter,
        fn_cost: FloatArrayLike | float | Parameter,
        fp_cost: FloatArrayLike | float | Parameter,
        caller: str = 'fit',
        force_array: Literal[True] = True,
        n_samples: int,
    ) -> tuple[
        FloatNDArray,
        FloatNDArray,
        FloatNDArray,
        FloatNDArray,
    ]: ...

    @overload
    def _check_costs(
        self,
        *,
        tp_cost: FloatArrayLike | float | Parameter,
        tn_cost: FloatArrayLike | float | Parameter,
        fn_cost: FloatArrayLike | float | Parameter,
        fp_cost: FloatArrayLike | float | Parameter,
        caller: str = 'fit',
        force_array: Literal[False] = False,
        n_samples: int | None = None,
    ) -> tuple[
        FloatNDArray | float,
        FloatNDArray | float,
        FloatNDArray | float,
        FloatNDArray | float,
    ]: ...

    def _check_costs(
        self,
        *,
        tp_cost: FloatArrayLike | float | Parameter,
        tn_cost: FloatArrayLike | float | Parameter,
        fn_cost: FloatArrayLike | float | Parameter,
        fp_cost: FloatArrayLike | float | Parameter,
        caller: str = 'fit',
        force_array: bool = False,
        n_samples: int | None = None,
    ) -> tuple[
        FloatNDArray | float,
        FloatNDArray | float,
        FloatNDArray | float,
        FloatNDArray | float,
    ]:
        """
        Check if costs are set and return them.

        Also convert them to numpy arrays if they are array-like.
        Overwrite costs set in constructor if they are set in the fit/predict method.
        """
        if tp_cost is Parameter.UNCHANGED:
            tp_cost = self.tp_cost  # type: ignore[attr-defined]
        if tn_cost is Parameter.UNCHANGED:
            tn_cost = self.tn_cost  # type: ignore[attr-defined]
        if fn_cost is Parameter.UNCHANGED:
            fn_cost = self.fn_cost  # type: ignore[attr-defined]
        if fp_cost is Parameter.UNCHANGED:
            fp_cost = self.fp_cost  # type: ignore[attr-defined]

        if all_costs_zero(tp_cost, tn_cost, fn_cost, fp_cost):
            warnings.warn(
                'All costs are zero. Setting fp_cost=1 and fn_cost=1. '
                f'To avoid this warning, set costs explicitly in the {self.__class__.__name__}.{caller}() method.',
                UserWarning,
                stacklevel=2,
            )
            fp_cost = 1
            fn_cost = 1

        if force_array:
            if n_samples is None:
                raise ValueError('n_samples should be set when force_array is True.')
            tp_cost = np.asarray(tp_cost) if not isinstance(tp_cost, Real) else np.full(n_samples, tp_cost)
            tn_cost = np.asarray(tn_cost) if not isinstance(tn_cost, Real) else np.full(n_samples, tn_cost)
            fn_cost = np.asarray(fn_cost) if not isinstance(fn_cost, Real) else np.full(n_samples, fn_cost)
            fp_cost = np.asarray(fp_cost) if not isinstance(fp_cost, Real) else np.full(n_samples, fp_cost)
        else:
            if not isinstance(tp_cost, Real):
                tp_cost = np.asarray(tp_cost)
            if not isinstance(tn_cost, Real):
                tn_cost = np.asarray(tn_cost)
            if not isinstance(fn_cost, Real):
                fn_cost = np.asarray(fn_cost)
            if not isinstance(fp_cost, Real):
                fp_cost = np.asarray(fp_cost)

        # Normalize shapes to (1, N)
        tp_cost = tp_cost.reshape(-1) if isinstance(tp_cost, np.ndarray) else tp_cost
        tn_cost = tn_cost.reshape(-1) if isinstance(tn_cost, np.ndarray) else tn_cost
        fn_cost = fn_cost.reshape(-1) if isinstance(fn_cost, np.ndarray) else fn_cost
        fp_cost = fp_cost.reshape(-1) if isinstance(fp_cost, np.ndarray) else fp_cost

        return tp_cost, tn_cost, fn_cost, fp_cost

    def _get_metric_loss(self) -> Metric | None:
        """Get the metric loss function if available."""
        return None


def all_float(*arrays: ArrayLike | float | Parameter) -> bool:
    return all(isinstance(array, Real) and not isinstance(array, Parameter) for array in arrays)


def all_costs_zero(
    tp_cost: FloatArrayLike | float | Parameter,
    tn_cost: FloatArrayLike | float | Parameter,
    fn_cost: FloatArrayLike | float | Parameter,
    fp_cost: FloatArrayLike | float | Parameter,
) -> bool:
    return (
        all_float(tp_cost, tn_cost, fn_cost, fp_cost)
        and sum(abs(cost) for cost in (tp_cost, tn_cost, fn_cost, fp_cost)) == 0.0  # type: ignore[misc, arg-type]
    )
