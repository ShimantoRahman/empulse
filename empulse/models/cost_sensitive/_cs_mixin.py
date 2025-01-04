import warnings
from numbers import Real

import numpy as np

from ..._common import Parameter


class CostSensitiveMixin:

    def _check_costs(self, *, tp_cost, tn_cost, fn_cost, fp_cost, caller='fit'):
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

        if (all(isinstance(cost, Real) for cost in (tp_cost, tn_cost, fn_cost, fp_cost)) and
                sum(abs(cost) for cost in (tp_cost, tn_cost, fn_cost, fp_cost)) == 0.0):
            warnings.warn(
                "All costs are zero. Setting fp_cost=1 and fn_cost=1. "
                f"To avoid this warning, set costs explicitly in the {self.__class__.__name__}.{caller}() method.",
                UserWarning)
            fp_cost = 1
            fn_cost = 1

        if not isinstance(tp_cost, Real):
            tp_cost = np.asarray(tp_cost)
        if not isinstance(tn_cost, Real):
            tn_cost = np.asarray(tn_cost)
        if not isinstance(fn_cost, Real):
            fn_cost = np.asarray(fn_cost)
        if not isinstance(fp_cost, Real):
            fp_cost = np.asarray(fp_cost)

        return tp_cost, tn_cost, fn_cost, fp_cost
