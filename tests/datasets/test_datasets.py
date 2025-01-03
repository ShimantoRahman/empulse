import pytest
import numpy as np
import pandas as pd
from empulse.datasets import (
    load_churn_tv_subscriptions,
    load_upsell_bank_telemarketing,
    load_give_me_some_credit,
    load_credit_scoring_pakdd
)

@pytest.mark.parametrize("loader", [
    load_churn_tv_subscriptions,
    load_upsell_bank_telemarketing,
    load_give_me_some_credit,
    load_credit_scoring_pakdd
])
@pytest.mark.parametrize("as_frame", [True, False])
@pytest.mark.parametrize("return_X_y_costs", [True, False])
def test_dataset_loading(loader, as_frame, return_X_y_costs):
    result = loader(as_frame=as_frame, return_X_y_costs=return_X_y_costs)

    if return_X_y_costs:
        if as_frame:
            data, target, tp_cost, fp_cost, tn_cost, fn_cost = result
            assert isinstance(data, pd.DataFrame)
            assert isinstance(target, pd.Series)
            assert isinstance(tp_cost, pd.Series) or isinstance(tp_cost, (int, float))
            assert isinstance(fp_cost, pd.Series) or isinstance(fp_cost, (int, float))
            assert isinstance(tn_cost, pd.Series) or isinstance(tn_cost, (int, float))
            assert isinstance(fn_cost, pd.Series) or isinstance(fn_cost, (int, float))
        else:
            data, target, tp_cost, fp_cost, tn_cost, fn_cost = result
            assert isinstance(data, np.ndarray)
            assert isinstance(target, np.ndarray)
            assert isinstance(tp_cost, np.ndarray) or isinstance(tp_cost, (int, float))
            assert isinstance(fp_cost, np.ndarray) or isinstance(fp_cost, (int, float))
            assert isinstance(tn_cost, np.ndarray) or isinstance(tn_cost, (int, float))
            assert isinstance(fn_cost, np.ndarray) or isinstance(fn_cost, (int, float))
    else:
        dataset = result
        if as_frame:
            assert isinstance(dataset.data, pd.DataFrame)
            assert isinstance(dataset.target, pd.Series)
            assert isinstance(dataset.tp_cost, pd.Series) or isinstance(dataset.tp_cost, (int, float))
            assert isinstance(dataset.fp_cost, pd.Series) or isinstance(dataset.fp_cost, (int, float))
            assert isinstance(dataset.tn_cost, pd.Series) or isinstance(dataset.tn_cost, (int, float))
            assert isinstance(dataset.fn_cost, pd.Series) or isinstance(dataset.fn_cost, (int, float))
        else:
            assert isinstance(dataset.data, np.ndarray)
            assert isinstance(dataset.target, np.ndarray)
            assert isinstance(dataset.tp_cost, np.ndarray) or isinstance(dataset.tp_cost, (int, float))
            assert isinstance(dataset.fp_cost, np.ndarray) or isinstance(dataset.fp_cost, (int, float))
            assert isinstance(dataset.tn_cost, np.ndarray) or isinstance(dataset.tn_cost, (int, float))
            assert isinstance(dataset.fn_cost, np.ndarray) or isinstance(dataset.fn_cost, (int, float))

    # Check dimensions
    if return_X_y_costs:
        assert data.shape[0] == target.shape[0]
        if isinstance(tp_cost, (pd.Series, np.ndarray)):
            assert tp_cost.shape[0] == target.shape[0]
        if isinstance(fp_cost, (pd.Series, np.ndarray)):
            assert fp_cost.shape[0] == target.shape[0]
        if isinstance(tn_cost, (pd.Series, np.ndarray)):
            assert tn_cost.shape[0] == target.shape[0]
        if isinstance(fn_cost, (pd.Series, np.ndarray)):
            assert fn_cost.shape[0] == target.shape[0]
    else:
        assert dataset.data.shape[0] == dataset.target.shape[0]
        if isinstance(dataset.tp_cost, (pd.Series, np.ndarray)):
            assert dataset.tp_cost.shape[0] == dataset.target.shape[0]
        if isinstance(dataset.fp_cost, (pd.Series, np.ndarray)):
            assert dataset.fp_cost.shape[0] == dataset.target.shape[0]
        if isinstance(dataset.tn_cost, (pd.Series, np.ndarray)):
            assert dataset.tn_cost.shape[0] == dataset.target.shape[0]
        if isinstance(dataset.fn_cost, (pd.Series, np.ndarray)):
            assert dataset.fn_cost.shape[0] == dataset.target.shape[0]
