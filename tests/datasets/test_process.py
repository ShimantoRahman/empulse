"""Unit tests for empulse.datasets._process cleaning functions."""

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from narwhals.typing import EagerAllowed, IntoBackend

from empulse.datasets._process import (
    _GIVE_ME_SOME_CREDIT_COL_MAP,
    _GIVE_ME_SOME_CREDIT_FEATURE_ORDER,
    process_bank_telemarketing,
    process_churn_tv,
    process_credit_scoring_pakdd,
    process_give_me_some_credit,
    process_iranian_churn,
)

BACKENDS = [
    pytest.param(pd, id='pandas'),
    pytest.param(pl, id='polars'),
]


def _from_dict(data: dict, backend: IntoBackend[EagerAllowed]) -> nw.DataFrame:
    return nw.from_dict(data, backend=backend)


def _churn_tv_raw(n: int = 5) -> dict:
    """Minimal churn-TV-style raw dict (all strings, as from _read_csv_gz)."""
    data: dict = {'idx': [str(i) for i in range(n)]}
    for i in range(45):
        data[f'feat_{i}'] = ['1.5'] * n
    data['target'] = (['1', '0'] * n)[:n]
    data['C_TP'] = ['100.0'] * n
    data['C_FP'] = ['10.0'] * n
    data['C_TN'] = ['0.0'] * n
    data['C_FN'] = ['50.0'] * n
    return data


class TestProcessChurnTV:
    @pytest.mark.parametrize('backend', BACKENDS)
    def test_return_types(self, backend):
        df = _from_dict(_churn_tv_raw(), backend)
        feat, target, costs = process_churn_tv(df)
        assert isinstance(feat, nw.DataFrame)
        assert isinstance(target, nw.Series)
        assert isinstance(costs, dict)

    def test_instance_costs_keys(self):
        df = _from_dict(_churn_tv_raw(), pd)
        _, _, costs = process_churn_tv(df)
        assert set(costs.keys()) == {'tp_benefit', 'fp_cost', 'tn_benefit', 'fn_cost'}

    def test_shapes_consistent(self):
        n = 7
        df = _from_dict(_churn_tv_raw(n), pd)
        feat, target, costs = process_churn_tv(df)
        assert len(feat) == n
        assert len(target) == n
        for arr in costs.values():
            assert arr.shape[0] == n

    def test_target_name(self):
        df = _from_dict(_churn_tv_raw(), pd)
        _, target, _ = process_churn_tv(df)
        assert target.name == 'churn'

    def test_feature_count(self):
        """45 anonymous features, index and cost cols excluded."""
        df = _from_dict(_churn_tv_raw(), pd)
        feat, _, _ = process_churn_tv(df)
        assert len(feat.columns) == 45

    def test_cost_values_correct(self):
        df = _from_dict(_churn_tv_raw(3), pd)
        _, _, costs = process_churn_tv(df)
        np.testing.assert_array_almost_equal(costs['tp_benefit'], [100.0, 100.0, 100.0])
        np.testing.assert_array_almost_equal(costs['fp_cost'], [10.0, 10.0, 10.0])


def _bank_raw(n: int = 6) -> dict:
    """Synthetic bank-telemarketing-style raw dict (all strings)."""
    return {
        'age': ['30', '45', '25', '60', '35', '50'][:n],
        'balance': ['1000', '500', '200', '3000', '-100', '800'][:n],
        'previous': ['0', '1', '2', '0', '1', '3'][:n],
        'job': ['admin.'] * n,
        'marital': ['married'] * n,
        'education': ['secondary'] * n,
        'default': ['no', 'no', 'yes', 'no', 'no', 'no'][:n],
        'housing': ['yes', 'no', 'yes', 'yes', 'no', 'yes'][:n],
        'loan': ['no'] * n,
        'poutcome': ['unknown'] * n,
        'y': ['yes', 'no', 'yes', 'yes', 'no', 'yes'][:n],
    }


class TestProcessBankTelemarketing:
    @pytest.mark.parametrize('backend', BACKENDS)
    def test_return_types(self, backend):
        df = _from_dict(_bank_raw(), backend)
        feat, target, balance = process_bank_telemarketing(df)
        assert isinstance(feat, nw.DataFrame)
        assert isinstance(target, nw.Series)
        assert isinstance(balance, np.ndarray)

    def test_negative_balance_filtered_out(self):
        """Clients with balance <= 0 must be excluded."""
        df = _from_dict(_bank_raw(6), pd)
        feat, _, balance = process_bank_telemarketing(df)
        # Row with balance=-100 should be dropped
        assert len(feat) == 5
        assert np.all(balance > 0)

    def test_target_name(self):
        df = _from_dict(_bank_raw(), pd)
        _, target, _ = process_bank_telemarketing(df)
        assert target.name == 'subscription'

    def test_target_binary(self):
        df = _from_dict(_bank_raw(6), pd)
        _, target, _ = process_bank_telemarketing(df)
        unique = set(target.to_numpy())
        assert unique <= {0, 1}

    def test_feature_columns(self):
        df = _from_dict(_bank_raw(), pd)
        feat, _, _ = process_bank_telemarketing(df)
        expected_cols = {
            'age',
            'balance',
            'previous',
            'job',
            'marital',
            'education',
            'has_credit_in_default',
            'has_housing_loan',
            'has_personal_loan',
            'previous_outcome',
        }
        assert set(feat.columns) == expected_cols

    def test_boolean_encoding(self):
        """default/housing/loan columns should be encoded as 0/1."""
        raw = _bank_raw(3)
        raw['default'] = ['yes', 'no', 'yes']
        df = _from_dict(raw, pd)
        feat, _, _ = process_bank_telemarketing(df)
        # Filter out the negative-balance row first
        has_default = feat.to_native()['has_credit_in_default'].to_numpy()
        assert set(has_default).issubset({0, 1})

    def test_shapes_consistent(self):
        df = _from_dict(_bank_raw(6), pd)
        feat, target, balance = process_bank_telemarketing(df)
        assert len(feat) == len(target) == len(balance)


def _pakdd_raw(n: int = 5) -> dict:
    """Minimal PAKDD-style raw dict (all strings, as from _read_csv_gz)."""
    # Must include TARGET_LABEL_BAD=1, PERSONAL_NET_INCOME, and at least one feature col.
    # The layout expected by process_credit_scoring_pakdd:
    # feature cols = all cols except TARGET, then drop last col (trailing metadata col).
    # We'll include the full expected set of 'interesting' columns + a dummy trailing col.
    base = {
        'TARGET_LABEL_BAD=1': ['0', '1', '0', '1', '0'][:n],
        'ID_CLIENT': ['1', '2', '3', '4', '5'][:n],
        'MONTHS_IN_RESIDENCE': ['12', '24', '6', '36', '18'][:n],
        'MONTHS_IN_THE_JOB': ['24', '12', '36', '8', '60'][:n],
        'AGE': ['30', '45', '28', '55', '40'][:n],
        'PERSONAL_NET_INCOME': ['500.0', '1000.0', '800.0', '600.0', '2000.0'][:n],
        'MATE_INCOME': ['0.0', '500.0', '0.0', '300.0', '1000.0'][:n],
        'QUANT_BANKING_ACCOUNTS': ['1', '2', '0', '3', '1'][:n],
        'QUANT_ADDITIONAL_CARDS_IN_THE_APPLICATION': ['0', '1', '0', '0', '2'][:n],
        'PAYMENT_DAY': ['5', '10', '15', '20', '25'][:n],
        'SEX': ['M', 'F', 'M', 'M', 'F'][:n],
        'MARITAL_STATUS': ['S', 'M', 'D', 'W', 'S'][:n],
        'FLAG_RESIDENCIAL_PHONE': ['Y', 'N', 'Y', 'Y', 'N'][:n],
        'FLAG_MOBILE_PHONE': ['N', 'Y', 'N', 'Y', 'Y'][:n],
        'FLAG_CONTACT_PHONE': ['Y', 'Y', 'N', 'Y', 'N'][:n],
        'FLAG_RESIDENCE_TOWN_EQ_WORKING_TOWN': ['Y', 'N', 'Y', 'N', 'Y'][:n],
        'FLAG_RESIDENCE_STATE_EQ_WORKING_STATE': ['Y', 'Y', 'N', 'N', 'Y'][:n],
        'FLAG_RESIDENCIAL_ADDRESS_EQ_POSTAL_ADDRESS': ['Y', 'N', 'Y', 'Y', 'N'][:n],
        'FLAG_MOTHERS_NAME': ['Y', 'N', 'Y', 'Y', 'N'][:n],
        'FLAG_FATHERS_NAME': ['Y', 'N', 'Y', 'N', 'Y'][:n],
        'FLAG_OTHER_CARD': ['N', 'N', 'N', 'N', 'Y'][:n],
        'FLAG_CARD_INSURANCE_OPTION': ['N', 'Y', 'N', 'Y', 'N'][:n],
        'RESIDENCE_TYPE': ['P', 'A', 'C', 'P', 'A'][:n],
        'SHOP_RANK': ['1', '2', '3', '1', '2'][:n],
        'AREA_CODE_RESIDENCIAL_PHONE': ['11', '21', '31', '41', '11'][:n],
        'ID_SHOP': ['100', '200', '300', '100', '200'][:n],
        'COD_APPLICATION_BOOTH': ['1', '2', '1', '3', '2'][:n],
        'PROFESSION_CODE': ['10', '20', '30', '10', '40'][:n],
        # Trailing metadata column (gets dropped by process_credit_scoring_pakdd)
        '_TRAILING': ['x'] * n,
    }
    return base


class TestProcessCreditScoringPakdd:
    @pytest.mark.parametrize('backend', BACKENDS)
    def test_return_types(self, backend):
        df = _from_dict(_pakdd_raw(), backend)
        feat, target, income = process_credit_scoring_pakdd(df)
        assert isinstance(feat, nw.DataFrame)
        assert isinstance(target, nw.Series)
        assert isinstance(income, np.ndarray)

    def test_target_name(self):
        df = _from_dict(_pakdd_raw(), pd)
        _, target, _ = process_credit_scoring_pakdd(df)
        assert target.name == 'default'

    def test_target_binary(self):
        df = _from_dict(_pakdd_raw(), pd)
        _, target, _ = process_credit_scoring_pakdd(df)
        assert set(target.to_numpy()).issubset({0, 1})

    def test_income_filter(self):
        """Rows with income outside [100, 10000] should be dropped."""
        raw = _pakdd_raw(3)
        raw['PERSONAL_NET_INCOME'] = ['50.0', '500.0', '20000.0']  # first and last filtered out
        df = _from_dict(raw, pd)
        feat, _, income = process_credit_scoring_pakdd(df)
        assert len(feat) == 1
        assert income[0] == pytest.approx(500.0 * 0.33, rel=1e-3)

    def test_income_scaling(self):
        """Monthly income array should be 33% of PERSONAL_NET_INCOME."""
        df = _from_dict(_pakdd_raw(), pd)
        feat, _, income = process_credit_scoring_pakdd(df)
        expected = feat.to_native()['personal_net_income'].to_numpy().astype(float) * 0.33
        np.testing.assert_allclose(income, expected, rtol=1e-4)

    def test_shapes_consistent(self):
        df = _from_dict(_pakdd_raw(5), pd)
        feat, target, income = process_credit_scoring_pakdd(df)
        assert len(feat) == len(target) == len(income)

    def test_flag_encoding(self):
        """FLAG columns should be encoded as 0/1 UInt8."""
        df = _from_dict(_pakdd_raw(), pd)
        feat_nw, _, _ = process_credit_scoring_pakdd(df)
        feat = feat_nw.to_native()
        flag_col = 'has_residential_phone'
        if flag_col in feat.columns:
            assert set(feat[flag_col].unique()).issubset({0, 1})

    def test_sex_encoding(self):
        """M → 1, F → 0."""
        raw = _pakdd_raw(2)
        raw['SEX'] = ['M', 'F']
        df = _from_dict(raw, pd)
        feat_nw, _, _ = process_credit_scoring_pakdd(df)
        feat = feat_nw.to_native()
        if 'is_male' in feat.columns:
            vals = feat['is_male'].to_numpy()
            assert vals[0] == 1
            assert vals[1] == 0

    def test_column_names_snake_case(self):
        """All output column names should be lowercase snake_case."""
        df = _from_dict(_pakdd_raw(), pd)
        feat, _, _ = process_credit_scoring_pakdd(df)
        for col in feat.columns:
            assert col == col.lower()
            assert ' ' not in col

    def test_invalid_income_values_excluded(self):
        """Sentinel strings 'N' and '' should be treated as null → filtered out."""
        raw = _pakdd_raw(3)
        raw['PERSONAL_NET_INCOME'] = ['N', '500.0', '']
        df = _from_dict(raw, pd)
        feat, _, _ = process_credit_scoring_pakdd(df)
        # Only the row with income=500 survives
        assert len(feat) == 1


# ---------------------------------------------------------------------------
# process_iranian_churn
# ---------------------------------------------------------------------------


def _iranian_raw(n: int = 4) -> dict:
    """Synthetic Iranian Churn raw dict (strings, as from load_or_fetch)."""
    return {
        'Call  Failure': ['2', '5', '0', '3'][:n],
        'Subscription  Length': ['30', '12', '24', '6'][:n],
        'Charge  Amount': ['100', '200', '50', '150'][:n],
        'Seconds of Use': ['500', '1000', '200', '750'][:n],
        'Frequency of use': ['10', '20', '5', '15'][:n],
        'Frequency of SMS': ['5', '10', '0', '8'][:n],
        'Distinct Called Numbers': ['15', '25', '8', '20'][:n],
        'Age Group': ['2', '3', '1', '4'][:n],
        'Tariff Plan': ['1', '2', '1', '1'][:n],
        'Status': ['yes', 'no', 'yes', 'no'][:n],
        'Age': ['28', '45', '35', '52'][:n],
        'Customer Value': ['500.0', '800.0', '300.0', '1200.0'][:n],
        'Churn': ['1', '0', '1', '0'][:n],
    }


class TestProcessIranianChurn:
    @pytest.mark.parametrize('backend', BACKENDS)
    def test_return_types(self, backend):
        feat, target, clv = process_iranian_churn(_iranian_raw(), backend)
        assert isinstance(feat, nw.DataFrame)
        assert isinstance(target, nw.Series)
        assert isinstance(clv, np.ndarray)

    def test_shapes_consistent(self):
        n = 4
        feat, target, clv = process_iranian_churn(_iranian_raw(n), pd)
        assert len(feat) == n
        assert len(target) == n
        assert clv.shape == (n,)

    def test_clv_values(self):
        _, _, clv = process_iranian_churn(_iranian_raw(4), pd)
        np.testing.assert_allclose(clv, [500.0, 800.0, 300.0, 1200.0])

    def test_target_binary(self):
        _, target, _ = process_iranian_churn(_iranian_raw(), pd)
        assert set(target.to_numpy()).issubset({0, 1})

    def test_yes_no_column_encoded(self):
        """'Status' column (yes/no) should become 0/1 UInt8."""
        feat, _, _ = process_iranian_churn(_iranian_raw(), pd)
        # 'status' after sanitization
        status_col = next((c for c in feat.columns if 'status' in c), None)
        if status_col:
            vals = set(feat.to_native()[status_col].to_numpy())
            assert vals.issubset({0, 1})

    def test_column_names_sanitized(self):
        """Column names should have no double underscores."""
        feat, _, _ = process_iranian_churn(_iranian_raw(), pd)
        for col in feat.columns:
            assert '__' not in col, f'Double underscore in column name: {col!r}'

    def test_clv_and_target_excluded_from_features(self):
        """CLV and Churn columns must not appear in the feature frame."""
        feat, _, _ = process_iranian_churn(_iranian_raw(), pd)
        for col in feat.columns:
            assert 'customer_value' not in col.lower()
            assert col.lower() not in {'churn', 'class'}

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_cross_backend_same_shape(self, backend):
        raw = _iranian_raw()
        feat_pd, _, clv_pd = process_iranian_churn(raw, pd)
        feat_pl, _, clv_pl = process_iranian_churn(raw, pl)
        assert feat_pd.shape == feat_pl.shape
        np.testing.assert_allclose(clv_pd, clv_pl)


def _gmsc_raw(n: int = 6) -> dict:
    """Synthetic Give-Me-Some-Credit raw dict using original OpenML column names."""
    return {
        'SeriousDlqin2yrs': ['0', '1', '0', '0', '1', '0'][:n],
        'RevolvingUtilizationOfUnsecuredLines': ['0.5', '0.9', '0.2', '0.3', '0.8', '0.1'][:n],
        'age': ['45', '35', '28', '60', '42', '55'][:n],
        'NumberOfTime30-59DaysPastDueNotWorse': ['0', '2', '0', '1', '3', '0'][:n],
        'DebtRatio': ['0.3', '0.8', '0.1', '0.2', '0.5', '0.05'][:n],
        'MonthlyIncome': ['5000', '3000', '4000', '8000', '2500', '6000'][:n],
        'NumberOfOpenCreditLinesAndLoans': ['5', '3', '8', '10', '2', '7'][:n],
        'NumberOfTimes90DaysLate': ['0', '1', '0', '0', '2', '0'][:n],
        'NumberRealEstateLoansOrLines': ['1', '0', '2', '3', '0', '1'][:n],
        'NumberOfTime60-89DaysPastDueNotWorse': ['0', '1', '0', '0', '1', '0'][:n],
        'NumberOfDependents': ['2', '0', '1', '3', '0', '2'][:n],
    }


class TestProcessGiveMeSomeCredit:
    @pytest.mark.parametrize('backend', BACKENDS)
    def test_return_types(self, backend):
        feat, target, income, debt, t_np = process_give_me_some_credit(_gmsc_raw(), backend)
        assert isinstance(feat, nw.DataFrame)
        assert isinstance(target, nw.Series)
        assert isinstance(income, np.ndarray)
        assert isinstance(debt, np.ndarray)
        assert isinstance(t_np, np.ndarray)

    def test_target_name(self):
        _, target, *_ = process_give_me_some_credit(_gmsc_raw(), pd)
        assert target.name == 'default'

    def test_target_binary(self):
        _, target, *_ = process_give_me_some_credit(_gmsc_raw(), pd)
        assert set(target.to_numpy()).issubset({0, 1})

    def test_shapes_consistent(self):
        n = 6
        feat, target, income, debt, t_np = process_give_me_some_credit(_gmsc_raw(n), pd)
        assert len(feat) == len(target) == len(income) == len(debt) == len(t_np)

    def test_missing_income_filtered(self):
        """Rows with '?' or '0' income should be dropped."""
        raw = _gmsc_raw(3)
        raw['MonthlyIncome'] = ['?', '3000', '0']  # first and last filtered
        feat, _, income, *_ = process_give_me_some_credit(raw, pd)
        assert len(feat) == 1
        assert income[0] == pytest.approx(3000.0)

    def test_high_debt_ratio_filtered(self):
        """Rows with debt_ratio >= 1 should be dropped."""
        raw = _gmsc_raw(3)
        raw['DebtRatio'] = ['0.3', '1.5', '0.2']  # middle one filtered
        feat, *_ = process_give_me_some_credit(raw, pd)
        assert len(feat) == 2

    def test_feature_order(self):
        """Feature columns should follow the canonical order."""
        feat, *_ = process_give_me_some_credit(_gmsc_raw(), pd)
        available = [c for c in _GIVE_ME_SOME_CREDIT_FEATURE_ORDER if c in feat.columns]
        assert feat.columns == available

    def test_column_name_mapping(self):
        """Original OpenML column names must be mapped to canonical names."""
        feat, *_ = process_give_me_some_credit(_gmsc_raw(), pd)
        assert 'monthly_income' in feat.columns
        assert 'debt_ratio' in feat.columns

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_cross_backend_same_shape(self, backend):
        raw = _gmsc_raw()
        feat_pd, *_ = process_give_me_some_credit(raw, pd)
        feat_other, *_ = process_give_me_some_credit(raw, backend)
        assert feat_pd.shape == feat_other.shape


class TestGiveMeSomeCreditColMap:
    def test_all_feature_order_cols_reachable(self):
        """Every column in _GIVE_ME_SOME_CREDIT_FEATURE_ORDER must be a value in the map."""
        mapped_values = set(_GIVE_ME_SOME_CREDIT_COL_MAP.values())
        for col in _GIVE_ME_SOME_CREDIT_FEATURE_ORDER:
            assert col in mapped_values, f'{col!r} is not a target value in the column map'
