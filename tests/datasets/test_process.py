"""Unit tests for empulse.datasets._process cleaning functions."""

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from narwhals.typing import EagerAllowed, IntoBackend

from empulse.datasets._cost_matrices import churn_retention_monthly_cost_matrix
from empulse.datasets._process import (
    _GIVE_ME_SOME_CREDIT_COL_MAP,
    _GIVE_ME_SOME_CREDIT_FEATURE_ORDER,
    KDD98_ATTRIBUTES,
    SOUTH_GERMAN_CREDIT_COLUMNS,
    process_bank_telemarketing,
    process_cell2cell,
    process_churn_tv,
    process_credit_card_fraud,
    process_credit_scoring_pakdd,
    process_default_credit_card_clients,
    process_give_me_some_credit,
    process_home_equity,
    process_ieee_fraud_detection,
    process_iranian_churn,
    process_kdd98,
    process_kddcup09_churn,
    process_south_german_credit,
    process_telco_customer_churn,
    process_vub_credit_scoring,
)
from empulse.metrics import Cost, Metric

from ._helpers import BACKENDS, MOCK_RAW_CELL2CELL


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
        assert set(costs.keys()) == {'tp_cost', 'fp_cost', 'tn_cost', 'fn_cost'}

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
        np.testing.assert_array_almost_equal(costs['tp_cost'], [100.0, 100.0, 100.0])
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
        assert set(feat['has_residential_phone'].unique()).issubset({0, 1})

    def test_sex_encoding(self):
        """M → 1, F → 0."""
        raw = _pakdd_raw(2)
        raw['SEX'] = ['M', 'F']
        df = _from_dict(raw, pd)
        feat_nw, _, _ = process_credit_scoring_pakdd(df)
        vals = feat_nw.to_native()['is_male'].to_numpy()
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
        assert set(feat.to_native()['status'].to_numpy()).issubset({0, 1})

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

    def test_cross_backend_same_shape(self):
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

    def test_cross_backend_same_shape(self):
        raw = _gmsc_raw()
        feat_pd, *_ = process_give_me_some_credit(raw, pd)
        feat_pl, *_ = process_give_me_some_credit(raw, pl)
        assert feat_pd.shape == feat_pl.shape


class TestGiveMeSomeCreditColMap:
    def test_all_feature_order_cols_reachable(self):
        """Every column in _GIVE_ME_SOME_CREDIT_FEATURE_ORDER must be a value in the map."""
        mapped_values = set(_GIVE_ME_SOME_CREDIT_COL_MAP.values())
        for col in _GIVE_ME_SOME_CREDIT_FEATURE_ORDER:
            assert col in mapped_values, f'{col!r} is not a target value in the column map'


class TestProcessLiteratureDatasets:
    """The processors of the datasets whose cost matrices come from the literature."""

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_process_telco_customer_churn(self, backend):
        raw = {
            'customerID': ['1', '2', '3', '4'],
            'gender': ['Female', 'Male', 'Male', 'Female'],
            'SeniorCitizen': ['0', '0', '1', '0'],
            'MonthlyCharges': ['29.85', '56.95', '53.85', '42.30'],
            'TotalCharges': ['29.85', '1889.50', ' ', '108.15'],  # 3rd row has blank TotalCharges
            'Churn': ['No', 'No', 'Yes', 'Yes'],
        }
        feat, target, charges = process_telco_customer_churn(raw, backend)
        # Blank row filtered out
        assert len(feat) == 3
        assert feat.columns == ['gender', 'senior_citizen', 'monthly_charges', 'total_charges']
        assert 'customer_id' not in feat.columns and 'customerID' not in feat.columns
        assert 'churn' not in feat.columns
        np.testing.assert_array_equal(charges, [29.85, 56.95, 42.30])
        np.testing.assert_array_equal(target.to_numpy(), [0, 0, 1])

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_process_default_credit_card_clients(self, backend):
        raw = {
            'id': ['1', '2', '3'],
            'x1': ['20000', '120000', '90000'],
            'x2': ['2', '2', '2'],
            'x3': ['2', '2', '2'],
            'x4': ['1', '2', '2'],
            'x5': ['24', '26', '34'],
            'y': ['1', '0', '0'],
        }
        feat, _target, cl, target_np = process_default_credit_card_clients(raw, backend)
        assert len(feat) == 3
        assert 'id' not in feat.columns
        assert 'target' not in feat.columns
        assert 'limit_bal' in feat.columns
        np.testing.assert_array_equal(cl, [20000.0, 120000.0, 90000.0])
        np.testing.assert_array_equal(target_np, [1, 0, 0])

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_process_ieee_fraud_detection(self, backend):
        raw = {
            'TransactionID': ['1001', '1002', '1003'],
            'TransactionDT': ['86400', '86401', '86402'],
            'TransactionAmt': ['36.5', '117.0', '280.0'],
            'ProductCD': ['W', 'W', 'H'],
            'id_01': ['-5.0', '?', '0.0'],
            'id-01': ['?', '?', '?'],  # the Kaggle test set's naming, empty for training rows
            'isFraud': ['0', '0', '1'],
        }
        feat, target, amount = process_ieee_fraud_detection(raw, backend)
        assert feat.columns == ['transaction_amt', 'product_cd', 'id_01']
        # numeric attributes are numbers, categorical ones stay strings; '?' is missing in both
        assert feat.schema['transaction_amt'] == nw.Float64
        assert feat.schema['id_01'] == nw.Float64
        assert feat['id_01'].is_null().to_list() == [False, True, False]
        assert feat['product_cd'].to_list() == ['W', 'W', 'H']
        np.testing.assert_array_equal(amount, [36.5, 117.0, 280.0])
        np.testing.assert_array_equal(target.to_numpy(), [0, 0, 1])

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_process_credit_card_fraud(self, backend):
        raw = {
            'Time': ['0.0', '1.0', '2.0', '3.0'],
            'V1': ['-1.35', '1.19', '-0.43', '0.50'],
            'V2': ['-0.07', '0.26', '-0.17', '0.10'],
            'Amount': ['149.62', '0.0', '2.69', '300.0'],  # 2nd row has Amount == 0.0, filtered out
            'Class': ['0', '1', '0', '1'],
        }
        feat, target, amount = process_credit_card_fraud(raw, backend)
        assert feat.columns == ['v1', 'v2', 'amount']
        np.testing.assert_array_equal(amount, [149.62, 2.69, 300.0])
        # the zero-amount fraud is dropped with its row
        np.testing.assert_array_equal(target.to_numpy(), [0, 0, 1])

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_process_kdd98(self, backend):
        raw = {col: ['1', '2', ' '] for col in KDD98_ATTRIBUTES}
        raw['GENDER'] = ['F', 'M', ' ']
        raw['AVGGIFT'] = ['12.5', '25.0', '15.0']
        raw['TARGET_B'] = ['0', '1', '1']
        raw['TARGET_D'] = ['0', '25', '7.5']

        feat, target, amount = process_kdd98(raw, backend)
        assert feat.columns == list(KDD98_ATTRIBUTES.values())
        # the donation amount is the false-negative cost, so it must not leak into the features
        assert 'target_d' not in feat.columns
        assert 'target_b' not in feat.columns
        np.testing.assert_array_equal(amount, [0.0, 25.0, 7.5])
        np.testing.assert_array_equal(target.to_numpy(), [0, 1, 1])

        # blanks are missing values, in both categorical and numeric attributes
        assert feat['gender'].is_null().to_list() == [False, False, True]
        assert feat['gender'].to_list()[:2] == ['F', 'M']
        assert feat['age'].is_null().to_list() == [False, False, True]
        assert feat['age'].to_list()[:2] == [1.0, 2.0]

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_process_vub_credit_scoring(self, backend):
        raw = {
            'ID': ['1', '2', '3'],
            'Default_45': ['0', '1', '0'],
            'Loan_amount': ['-1.0', '0.5', '2.0'],
            'FICO_Score': ['0.3', '', '-0.7'],
            'Days_late': ['0', '45', '0'],
            'Expected_loss': ['-1.0', '0.5', '2.0'],
            'Expected_profit': ['-1.0', '0.5', '2.0'],
            'Test_set1': ['0', '1', '0'],
            'v1': ['0.1', '0.2', '0.3'],
        }
        df = nw.from_dict(raw, backend=backend)
        feat, target, amounts = process_vub_credit_scoring(df)
        assert feat.columns == ['loan_amount', 'fico_score', 'v1']
        # standardised loan amounts are shifted to be strictly positive, keeping their differences
        np.testing.assert_allclose(amounts, [1e-9, 1.5 + 1e-9, 3.0 + 1e-9])
        np.testing.assert_array_equal(target.to_numpy(), [0, 1, 0])

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_process_home_equity(self, backend):
        raw = {
            'BAD': ['1', '0', '1'],
            'LOAN': ['1100', '1700', '1800'],
            'MORTDUE': ['25860.0', '97800.0', '48649.0'],
            'REASON': ['HomeImp', 'HomeImp', 'DebtCon'],
            'JOB': ['Other', 'Office', 'Other'],
            'DEBTINC': ['?', '37.11', '36.88'],
        }
        feat, _target, amounts, target_np = process_home_equity(raw, backend)
        assert feat.columns == ['loan_amount', 'mortgage_due', 'reason', 'job', 'debt_to_income']
        np.testing.assert_array_equal(amounts, [1100.0, 1700.0, 1800.0])
        np.testing.assert_array_equal(target_np, [1, 0, 1])

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_process_south_german_credit(self, backend):
        rows = [
            # laufkont laufzeit moral verw hoehe ... kredit
            '1 18 4 2 1049 1 2 4 2 1 4 2 21 3 1 1 3 2 1 2 1',
            '1 9 4 0 2799 1 3 2 3 1 2 1 36 3 1 2 3 1 1 2 0',
        ]
        german_names = list(SOUTH_GERMAN_CREDIT_COLUMNS)
        raw = {col: [row.split()[i] for row in rows] for i, col in enumerate(german_names)}

        feat, target, amounts, target_np = process_south_german_credit(raw, backend)
        assert feat.columns == [SOUTH_GERMAN_CREDIT_COLUMNS[col] for col in german_names[:-1]]
        np.testing.assert_array_equal(amounts, [1049.0, 2799.0])
        # `kredit` is 1 = good; the positive class is the bad credit risk
        np.testing.assert_array_equal(target_np, [0, 1])
        np.testing.assert_array_equal(target.to_numpy(), [0, 1])

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_process_kddcup09_churn(self, backend):
        raw = {
            'Var1': ['10.5', '?'],
            'Var191': ['cat_a', '?'],
            'CHURN': ['-1', '1'],
        }
        feat, target = process_kddcup09_churn(raw, backend)
        assert feat.columns == ['var1', 'var191']
        np.testing.assert_array_equal(target.to_numpy(), [0, 1])


class TestProcessCell2Cell:
    @pytest.mark.parametrize('backend', BACKENDS)
    def test_process_cell2cell_filtering_and_types(self, backend):
        feat, target, monthly_revenue = process_cell2cell(MOCK_RAW_CELL2CELL, backend)

        # Row with missing MonthlyRevenue should be filtered out: 4 -> 3
        assert len(target) == 3
        assert len(monthly_revenue) == 3
        df = nw.to_native(feat)
        assert len(df) == 3

        # Target checks
        np.testing.assert_array_equal(target.to_numpy(), [1, 0, 0])

        # Monthly revenue checks
        np.testing.assert_allclose(monthly_revenue, [50.0, 100.0, 75.5])

        # Feature columns: CustomerID and Churn dropped
        assert 'customerid' not in feat.columns
        assert 'customer_id' not in feat.columns
        assert 'churn' not in feat.columns
        assert 'monthly_revenue' in feat.columns
        assert 'service_area' in feat.columns
        assert 'credit_rating' in feat.columns
        assert 'handset_price' in feat.columns
        assert 'age_hh1' in feat.columns
        assert 'non_us_travel' in feat.columns

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_process_cell2cell_cost_evaluation(self, backend):
        _feat, _target, monthly_revenue = process_cell2cell(MOCK_RAW_CELL2CELL, backend)

        cm, costs = churn_retention_monthly_cost_matrix(
            monthly_revenue, clv_months=12, incentive_fraction=0.05, contact_cost=1, accept_rate=0.3
        )
        metric = Metric(cm, Cost())
        y_true = np.array([1, 0, 0])
        y_score = np.array([0.9, 0.1, 0.2])
        score = metric(y_true, y_score, **costs)
        assert np.isfinite(score)
