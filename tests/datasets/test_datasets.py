import numpy as np
import pandas as pd
import pytest

from empulse.datasets import (
    load_churn_tv_subscriptions,
    load_credit_scoring_pakdd,
    load_give_me_some_credit,
    load_upsell_bank_telemarketing,
)


@pytest.mark.parametrize(
    'loader',
    [load_churn_tv_subscriptions, load_upsell_bank_telemarketing, load_give_me_some_credit, load_credit_scoring_pakdd],
)
@pytest.mark.parametrize('as_frame', [True, False])
@pytest.mark.parametrize('return_X_y_costs', [True, False])
def test_dataset_loading(loader, as_frame, return_X_y_costs):
    result = loader(as_frame=as_frame, return_X_y_costs=return_X_y_costs)

    if return_X_y_costs:
        if as_frame:
            data, target, tp_cost, fp_cost, tn_cost, fn_cost = result
            assert isinstance(data, pd.DataFrame)
            assert isinstance(target, pd.Series)
            assert isinstance(tp_cost, pd.Series | int | float)
            assert isinstance(fp_cost, pd.Series | int | float)
            assert isinstance(tn_cost, pd.Series | int | float)
            assert isinstance(fn_cost, pd.Series | int | float)
        else:
            data, target, tp_cost, fp_cost, tn_cost, fn_cost = result
            assert isinstance(data, np.ndarray)
            assert isinstance(target, np.ndarray)
            assert isinstance(tp_cost, np.ndarray | int | float)
            assert isinstance(fp_cost, np.ndarray | int | float)
            assert isinstance(tn_cost, np.ndarray | int | float)
            assert isinstance(fn_cost, np.ndarray | int | float)
        assert data.shape[0] == target.shape[0]
        if isinstance(tp_cost, pd.Series | np.ndarray):
            assert tp_cost.shape[0] == target.shape[0]
        if isinstance(fp_cost, pd.Series | np.ndarray):
            assert fp_cost.shape[0] == target.shape[0]
        if isinstance(tn_cost, pd.Series | np.ndarray):
            assert tn_cost.shape[0] == target.shape[0]
        if isinstance(fn_cost, pd.Series | np.ndarray):
            assert fn_cost.shape[0] == target.shape[0]
    else:
        dataset = result
        if as_frame:
            assert isinstance(dataset.data, pd.DataFrame)
            assert isinstance(dataset.target, pd.Series)
            assert isinstance(dataset.tp_cost, pd.Series | int | float)
            assert isinstance(dataset.fp_cost, pd.Series | int | float)
            assert isinstance(dataset.tn_cost, pd.Series | int | float)
            assert isinstance(dataset.fn_cost, pd.Series | int | float)
        else:
            assert isinstance(dataset.data, np.ndarray)
            assert isinstance(dataset.target, np.ndarray)
            assert isinstance(dataset.tp_cost, np.ndarray | int | float)
            assert isinstance(dataset.fp_cost, np.ndarray | int | float)
            assert isinstance(dataset.tn_cost, np.ndarray | int | float)
            assert isinstance(dataset.fn_cost, np.ndarray | int | float)
        assert dataset.data.shape[0] == dataset.target.shape[0]
        if isinstance(dataset.tp_cost, pd.Series | np.ndarray):
            assert dataset.tp_cost.shape[0] == dataset.target.shape[0]
        if isinstance(dataset.fp_cost, pd.Series | np.ndarray):
            assert dataset.fp_cost.shape[0] == dataset.target.shape[0]
        if isinstance(dataset.tn_cost, pd.Series | np.ndarray):
            assert dataset.tn_cost.shape[0] == dataset.target.shape[0]
        if isinstance(dataset.fn_cost, pd.Series | np.ndarray):
            assert dataset.fn_cost.shape[0] == dataset.target.shape[0]
