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
        caller: str = 'fit',
        force_array: bool = False,
        n_samples: int | None = None,
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
        if tp_cost is Parameter.UNCHANGED:
            tp_cost = self.tp_cost
        if tn_cost is Parameter.UNCHANGED:
            tn_cost = self.tn_cost
        if fn_cost is Parameter.UNCHANGED:
            fn_cost = self.fn_cost
        if fp_cost is Parameter.UNCHANGED:
            fp_cost = self.fp_cost

        if (
            all(isinstance(cost, Real) for cost in (tp_cost, tn_cost, fn_cost, fp_cost))
            and sum(abs(cost) for cost in (tp_cost, tn_cost, fn_cost, fp_cost)) == 0.0
        ):
            warnings.warn(
                'All costs are zero. Setting fp_cost=1 and fn_cost=1. '
                f'To avoid this warning, set costs explicitly in the {self.__class__.__name__}.{caller}() method.',
                UserWarning,
                stacklevel=2,
            )
            fp_cost = 1
            fn_cost = 1

        if force_array:
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

        return tp_cost, tn_cost, fn_cost, fp_cost
