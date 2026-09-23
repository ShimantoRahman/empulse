import re

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from empulse.datasets import (
    Dataset,
    fetch_cell2cell,
    fetch_credit_card_fraud,
    fetch_default_credit_card_clients,
    fetch_home_equity,
    fetch_ieee_fraud_detection,
    fetch_kdd98,
    fetch_kddcup09_churn,
    fetch_south_german_credit,
    fetch_telco_customer_churn,
    load_vub_credit_scoring,
)
from empulse.datasets._cost_matrices import (
    churn_monthly_charges_cost_matrix,
    credit_scoring_known_cl_cost_matrix,
    direct_marketing_cost_matrix,
    fraud_detection_cost_matrix,
)
from empulse.datasets._process import (
    KDD98_ATTRIBUTES,
    SOUTH_GERMAN_CREDIT_COLUMNS,
    process_credit_card_fraud,
    process_default_credit_card_clients,
    process_home_equity,
    process_ieee_fraud_detection,
    process_kdd98,
    process_kddcup09_churn,
    process_south_german_credit,
    process_telco_customer_churn,
    process_vub_credit_scoring,
)
from empulse.metrics import Cost, CostMatrix, Metric

_BACKENDS = [
    pytest.param(pd, id='pandas'),
    pytest.param(pl, id='polars'),
]


def _bahnsen_fp_cost(cl, pi_1, *, interest_rate=0.0479, fund_cost=0.0294, lgd=0.75, n=24):
    """Independent restatement of the Bahnsen et al. (2014) false positive cost."""
    int_r, int_cf = interest_rate / 12, fund_cost / 12

    def lost_profit(credit_line):
        installment = credit_line * int_r * (1 + int_r) ** n / ((1 + int_r) ** n - 1)
        present_value = installment / int_cf * (1 - (1 + int_cf) ** -n)
        return present_value - credit_line

    cl_avg = np.mean(cl)
    return np.maximum(0, lost_profit(cl) - (1 - pi_1) * lost_profit(cl_avg) + pi_1 * cl_avg * lgd)


class TestCostMatrices:
    def test_churn_monthly_charges_cost_matrix(self):
        charges = np.array([29.85, 56.95])
        cm, costs = churn_monthly_charges_cost_matrix(charges, fn_months=12.0, fp_months=2.0)
        assert isinstance(cm, CostMatrix)
        np.testing.assert_array_equal(costs['monthly_charges'], charges)

        # Predict nobody churns: the churner costs 12 months of charges, the other nothing.
        score = Metric(cm, Cost())(np.array([1, 0]), np.zeros(2), **costs)
        np.testing.assert_allclose(score, 12 * charges[0] / 2)

    def test_credit_scoring_known_cl_cost_matrix(self):
        cl = np.array([20000.0, 120000.0, 90000.0])
        target = np.array([1, 0, 0])
        cm, costs = credit_scoring_known_cl_cost_matrix(cl, target)
        assert isinstance(cm, CostMatrix)
        np.testing.assert_array_equal(costs['cl'], cl)
        np.testing.assert_allclose(costs['fp_cost'], _bahnsen_fp_cost(cl, pi_1=1 / 3))

        # Reject everybody: each non-defaulter costs its fp_cost.
        score = Metric(cm, Cost())(target, np.ones(3), **costs)
        np.testing.assert_allclose(score, costs['fp_cost'][1:].sum() / 3)

    def test_fraud_detection_cost_matrix(self):
        amounts = np.array([50.0, 120.5])
        cm, costs = fraud_detection_cost_matrix(amounts, investigation_cost=10.0)
        np.testing.assert_array_equal(costs['amount'], amounts)

        metric = Metric(cm, Cost())
        # Investigating everything costs c_f per transaction; investigating nothing loses the fraud.
        np.testing.assert_allclose(metric(np.array([1, 0]), np.ones(2), **costs), 10.0)
        np.testing.assert_allclose(metric(np.array([1, 0]), np.zeros(2), **costs), 50.0 / 2)

    def test_direct_marketing_cost_matrix(self):
        amounts = np.array([15.0, 0.0])
        cm, costs = direct_marketing_cost_matrix(amounts, contact_cost=0.68)
        np.testing.assert_array_equal(costs['amount'], amounts)

        metric = Metric(cm, Cost())
        np.testing.assert_allclose(metric(np.array([1, 0]), np.ones(2), **costs), 0.68)
        np.testing.assert_allclose(metric(np.array([1, 0]), np.zeros(2), **costs), 15.0 / 2)


class TestProcessors:
    @pytest.mark.parametrize('backend', _BACKENDS)
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

    @pytest.mark.parametrize('backend', _BACKENDS)
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

    @pytest.mark.parametrize('backend', _BACKENDS)
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

    @pytest.mark.parametrize('backend', _BACKENDS)
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

    @pytest.mark.parametrize('backend', _BACKENDS)
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

    @pytest.mark.parametrize('backend', _BACKENDS)
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

    @pytest.mark.parametrize('backend', _BACKENDS)
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

    @pytest.mark.parametrize('backend', _BACKENDS)
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

    @pytest.mark.parametrize('backend', _BACKENDS)
    def test_process_kddcup09_churn(self, backend):
        raw = {
            'Var1': ['10.5', '?'],
            'Var191': ['cat_a', '?'],
            'CHURN': ['-1', '1'],
        }
        feat, target = process_kddcup09_churn(raw, backend)
        assert feat.columns == ['var1', 'var191']
        np.testing.assert_array_equal(target.to_numpy(), [0, 1])


class TestLocalVubDataset:
    @pytest.mark.parametrize('backend', _BACKENDS)
    def test_load_vub_credit_scoring(self, backend):
        ds = load_vub_credit_scoring(backend=backend)
        assert isinstance(ds, Dataset)
        assert isinstance(ds.cost_matrix, CostMatrix)
        assert len(ds.data) == 18917
        assert len(ds.feature_names) == 16
        assert int(np.asarray(ds.target).sum()) == 3206
        assert np.all(ds.instance_costs['cl'] > 0)
        np.testing.assert_allclose(
            ds.instance_costs['fp_cost'],
            _bahnsen_fp_cost(ds.instance_costs['cl'], pi_1=3206 / 18917),
        )

        metric = Metric(ds.cost_matrix, Cost())
        score = metric(ds.target, np.full(len(ds.target), 0.5), **ds.instance_costs)
        assert np.isfinite(score)


class TestMissingDataHandling:
    @pytest.mark.parametrize(
        'fetcher',
        [
            fetch_telco_customer_churn,
            fetch_default_credit_card_clients,
            fetch_credit_card_fraud,
            fetch_home_equity,
            fetch_ieee_fraud_detection,
            fetch_kdd98,
            fetch_kddcup09_churn,
            fetch_south_german_credit,
            fetch_cell2cell,
        ],
    )
    def test_fetch_missing_raises_oserror(self, tmp_path, fetcher):
        with pytest.raises(OSError, match='download_if_missing'):
            fetcher(backend=pd, data_home=tmp_path, download_if_missing=False)


_SNAKE_CASE = re.compile(r'^[a-z0-9]+(_[a-z0-9]+)*$')


def _check_remote(ds, *, n_samples, n_positives, n_features, cost_keys):
    assert isinstance(ds, Dataset)
    assert isinstance(ds.cost_matrix, CostMatrix)
    assert len(ds.data) == n_samples
    assert len(ds.target) == n_samples
    assert len(ds.feature_names) == n_features
    assert [name for name in ds.feature_names if not _SNAKE_CASE.match(name)] == []
    assert int(np.asarray(ds.target).sum()) == n_positives
    assert set(ds.instance_costs) == cost_keys
    for arr in ds.instance_costs.values():
        assert arr.shape == (n_samples,)

    score = Metric(ds.cost_matrix, Cost())(ds.target, np.full(n_samples, 0.5), **ds.instance_costs)
    assert np.isfinite(score)


@pytest.mark.remote
class TestRemoteDatasets:
    @pytest.mark.parametrize('backend', _BACKENDS)
    def test_fetch_telco_customer_churn(self, tmp_path, backend):
        ds = fetch_telco_customer_churn(backend=backend, data_home=tmp_path)
        _check_remote(ds, n_samples=7032, n_positives=1869, n_features=19, cost_keys={'monthly_charges'})

    def test_fetch_default_credit_card_clients(self, tmp_path):
        ds = fetch_default_credit_card_clients(backend=pd, data_home=tmp_path)
        _check_remote(ds, n_samples=30000, n_positives=6636, n_features=23, cost_keys={'cl', 'fp_cost'})

    def test_fetch_credit_card_fraud(self, tmp_path):
        ds = fetch_credit_card_fraud(backend=pd, data_home=tmp_path)
        _check_remote(ds, n_samples=282982, n_positives=465, n_features=29, cost_keys={'amount'})
        assert np.all(ds.instance_costs['amount'] > 0)

    def test_fetch_ieee_fraud_detection(self, tmp_path):
        ds = fetch_ieee_fraud_detection(backend=pd, data_home=tmp_path)
        _check_remote(ds, n_samples=590540, n_positives=20663, n_features=431, cost_keys={'amount'})

    def test_fetch_kdd98(self, tmp_path):
        ds = fetch_kdd98(backend=pd, data_home=tmp_path)
        _check_remote(ds, n_samples=191779, n_positives=9716, n_features=22, cost_keys={'amount'})
        donated = np.asarray(ds.target) == 1
        assert np.all(ds.instance_costs['amount'][donated] > 0)
        assert np.all(ds.instance_costs['amount'][~donated] == 0)

    def test_fetch_home_equity(self, tmp_path):
        ds = fetch_home_equity(backend=pd, data_home=tmp_path)
        _check_remote(ds, n_samples=5960, n_positives=1189, n_features=12, cost_keys={'cl', 'fp_cost'})

    @pytest.mark.parametrize('backend', _BACKENDS)
    def test_fetch_south_german_credit(self, tmp_path, backend):
        ds = fetch_south_german_credit(backend=backend, data_home=tmp_path)
        _check_remote(ds, n_samples=1000, n_positives=300, n_features=20, cost_keys={'cl', 'fp_cost'})

    def test_fetch_kddcup09_churn(self, tmp_path):
        ds = fetch_kddcup09_churn(backend=pd, data_home=tmp_path)
        _check_remote(ds, n_samples=50000, n_positives=3672, n_features=230, cost_keys={'clv'})
        np.testing.assert_array_equal(ds.instance_costs['clv'], 200.0)

    @pytest.mark.parametrize('backend', _BACKENDS)
    def test_fetch_cell2cell(self, tmp_path, backend):
        ds = fetch_cell2cell(backend=backend, data_home=tmp_path)
        _check_remote(ds, n_samples=50891, n_positives=14641, n_features=56, cost_keys={'monthly_revenue'})
