import warnings
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..._common import Parameter


class CostSensitiveMixin:
    def _check_costs(
            self,
            *,
            tp_cost: ArrayLike | float | Parameter,
            tn_cost: ArrayLike | float | Parameter,
            fn_cost: ArrayLike | float | Parameter,
            fp_cost: ArrayLike | float | Parameter,
            caller: str = 'fit'
    ) -> tuple[
        NDArray | float,
        NDArray | float,
        NDArray | float,
        NDArray | float,
    ]:
        """
        Check if costs are set and return them.
        Also convert them to numpy arrays if they are array-like.
        Overwrite costs set in constructor if they are set in the fit/predict method.
        """

        tp_cost, tn_cost, fn_cost, fp_cost = self._set_costs(
            tp_cost=tp_cost,
            tn_cost=tn_cost,
            fn_cost=fn_cost,
            fp_cost=fp_cost
        )

        if not isinstance(tp_cost, Real):
            tp_cost = np.asarray(tp_cost)
        if not isinstance(tn_cost, Real):
            tn_cost = np.asarray(tn_cost)
        if not isinstance(fn_cost, Real):
            fn_cost = np.asarray(fn_cost)
        if not isinstance(fp_cost, Real):
            fp_cost = np.asarray(fp_cost)

        all_numbers = all(isinstance(cost, Real) for cost in (tp_cost, tn_cost, fn_cost, fp_cost))
        all_zeros = all(abs(cost) == 0 for cost in (tp_cost, tn_cost, fn_cost, fp_cost))

        if all_numbers and all_zeros:
            warnings.warn(
                "All costs are zero. Setting fp_cost=1 and fn_cost=1. "
                f"To avoid this warning, set costs explicitly in the {self.__class__.__name__}.{caller}() method.",
                UserWarning)
            fp_cost = 1
            fn_cost = 1

        # return tp_cost
        return tp_cost, tn_cost, fn_cost, fp_cost

    def _set_costs(
            self,
            *,
            tp_cost: ArrayLike | float | Parameter,
            tn_cost: ArrayLike | float | Parameter,
            fn_cost: ArrayLike | float | Parameter,
            fp_cost: ArrayLike | float | Parameter,
    ) -> tuple[ArrayLike | float, ArrayLike | float, ArrayLike | float, ArrayLike | float]:
        if tp_cost is Parameter.UNCHANGED:
            tp_cost = self.tp_cost
        if tn_cost is Parameter.UNCHANGED:
            tn_cost = self.tn_cost
        if fn_cost is Parameter.UNCHANGED:
            fn_cost = self.fn_cost
        if fp_cost is Parameter.UNCHANGED:
            fp_cost = self.fp_cost

        return tp_cost, tn_cost, fn_cost, fp_cost
