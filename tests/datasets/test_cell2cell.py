from unittest.mock import patch

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from empulse.datasets import Dataset, fetch_cell2cell
from empulse.datasets._cost_matrices import churn_retention_monthly_cost_matrix
from empulse.datasets._process import process_cell2cell
from empulse.metrics import Cost, Metric

_BACKENDS = [
    pytest.param(pd, id='pandas'),
    pytest.param(pl, id='polars'),
]

_MOCK_RAW_CELL2CELL = {
    'CustomerID': ['3000001', '3000002', '3000003', '3000004'],
    'Churn': ['Yes', 'No', 'Yes', 'No'],
    'MonthlyRevenue': ['50.0', '100.0', '', '75.5'],
    'MonthlyMinutes': ['200', '500', '150', '300'],
    'TotalRecurringCharge': ['40.0', '80.0', '30.0', '60.0'],
    'DirectorAssistedCalls': ['0.5', '1.0', '0.0', '0.25'],
    'OverageMinutes': ['10', '0', '5', '20'],
    'RoamingCalls': ['0', '2', '0', '1'],
    'PercChangeMinutes': ['-10', '5', '0', '15'],
    'PercChangeRevenues': ['-5', '10', '0', '8'],
    'DroppedCalls': ['1.0', '0.5', '2.0', '0.0'],
    'BlockedCalls': ['0.0', '1.0', '0.0', '0.5'],
    'UnansweredCalls': ['5.0', '10.0', '2.0', '3.0'],
    'CustomerCareCalls': ['0', '1', '0', '2'],
    'ThreewayCalls': ['0', '0', '1', '0'],
    'ReceivedCalls': ['50', '100', '30', '80'],
    'OutboundCalls': ['20', '40', '10', '30'],
    'InboundCalls': ['5', '10', '2', '8'],
    'PeakCallsInOut': ['30', '60', '20', '50'],
    'OffPeakCallsInOut': ['20', '40', '10', '30'],
    'DroppedBlockedCalls': ['1.0', '1.5', '2.0', '0.5'],
    'CallForwardingCalls': ['0', '0', '0', '0'],
    'CallWaitingCalls': ['1', '2', '0', '1'],
    'MonthsInService': ['12', '24', '6', '36'],
    'UniqueSubs': ['1', '2', '1', '1'],
    'ActiveSubs': ['1', '1', '1', '1'],
    'ServiceArea': ['SEAPOR503', 'OKCTUL918', 'SEAPOR503', 'MILMIL414'],
    'Handsets': ['1', '2', '1', '3'],
    'HandsetModels': ['1', '1', '1', '2'],
    'CurrentEquipmentDays': ['300', '600', '150', '900'],
    'AgeHH1': ['45', '50', '30', '60'],
    'AgeHH2': ['0', '48', '0', '55'],
    'ChildrenInHH': ['No', 'Yes', 'No', 'No'],
    'HandsetRefurbished': ['No', 'No', 'Yes', 'No'],
    'HandsetWebCapable': ['Yes', 'Yes', 'Yes', 'Yes'],
    'TruckOwner': ['No', 'Yes', 'No', 'No'],
    'RVOwner': ['No', 'No', 'No', 'Yes'],
    'Homeownership': ['Known', 'Known', 'Unknown', 'Known'],
    'BuysViaMailOrder': ['Yes', 'Yes', 'No', 'Yes'],
    'RespondsToMailOffers': ['Yes', 'No', 'No', 'Yes'],
    'OptOutMailings': ['No', 'No', 'No', 'No'],
    'NonUSTravel': ['No', 'No', 'No', 'Yes'],
    'OwnsComputer': ['Yes', 'Yes', 'No', 'Yes'],
    'HasCreditCard': ['Yes', 'Yes', 'No', 'Yes'],
    'RetentionCalls': ['0', '1', '0', '0'],
    'RetentionOffersAccepted': ['0', '1', '0', '0'],
    'NewCellphoneUser': ['No', 'No', 'Yes', 'No'],
    'NotNewCellphoneUser': ['Yes', 'Yes', 'No', 'Yes'],
    'ReferralsMadeBySubscriber': ['0', '1', '0', '0'],
    'IncomeGroup': ['4', '6', '2', '8'],
    'OwnsMotorcycle': ['No', 'No', 'No', 'No'],
    'AdjustmentsToCreditRating': ['0', '0', '0', '1'],
    'HandsetPrice': ['30', '100', 'Unknown', '150'],
    'MadeCallToRetentionTeam': ['No', 'Yes', 'No', 'No'],
    'CreditRating': ['1-Highest', '2-High', '3-Good', '1-Highest'],
    'PrizmCode': ['Suburban', 'Town', 'Rural', 'Other'],
    'Occupation': ['Professional', 'Professional', 'Other', 'Crafts'],
    'MaritalStatus': ['No', 'Yes', 'Unknown', 'Yes'],
}


class TestProcessCell2Cell:
    @pytest.mark.parametrize('backend', _BACKENDS)
    def test_process_cell2cell_filtering_and_types(self, backend):
        feat, target, monthly_revenue = process_cell2cell(_MOCK_RAW_CELL2CELL, backend)

        # Row with missing MonthlyRevenue should be filtered out: 4 -> 3
        assert len(target) == 3
        assert len(monthly_revenue) == 3
        df = nw.to_native(feat)
        assert len(df) == 3

        # Target checks
        target_np = nw.to_native(target)
        target_np = target_np.to_numpy() if hasattr(target_np, 'to_numpy') else np.asarray(target_np)
        np.testing.assert_array_equal(target_np, [1, 0, 0])

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

    @pytest.mark.parametrize('backend', _BACKENDS)
    def test_process_cell2cell_cost_evaluation(self, backend):
        _feat, _target, monthly_revenue = process_cell2cell(_MOCK_RAW_CELL2CELL, backend)

        cm, costs = churn_retention_monthly_cost_matrix(
            monthly_revenue, clv_months=12, incentive_fraction=0.05, contact_cost=1, accept_rate=0.3
        )
        metric = Metric(cm, Cost())
        y_true = np.array([1, 0, 0])
        y_score = np.array([0.9, 0.1, 0.2])
        score = metric(y_true, y_score, **costs)
        assert np.isfinite(score)


class TestChurnRetentionMonthlyCostMatrix:
    def test_costs_follow_verbraken_with_clv_from_revenue(self):
        monthly_revenue = np.array([50.0, 80.0])
        cm, costs = churn_retention_monthly_cost_matrix(
            monthly_revenue, clv_months=12, incentive_fraction=0.05, contact_cost=1, accept_rate=0.3
        )
        assert set(costs) == {'monthly_revenue'}

        # Predict every customer as a churner: a churner yields the retention benefit, a
        # non-churner costs the incentive plus the contact.
        y_true = np.array([1, 0])
        clv = 12 * monthly_revenue
        tp_benefit = 0.3 * (clv[0] - 0.05 * clv[0] - 1) - 0.7 * 1
        fp_cost = 0.05 * clv[1] + 1
        expected = (-tp_benefit + fp_cost) / 2

        score = Metric(cm, Cost())(y_true, np.ones(2), **costs)
        np.testing.assert_allclose(score, expected)

    def test_clv_months_is_overridable(self):
        monthly_revenue = np.array([50.0])
        cm, costs = churn_retention_monthly_cost_matrix(
            monthly_revenue, clv_months=12, incentive_fraction=0.05, contact_cost=1, accept_rate=0.3
        )
        score = Metric(cm, Cost())(np.array([0]), np.ones(1), clv_months=24, **costs)
        np.testing.assert_allclose(score, 0.05 * 24 * 50.0 + 1)


class TestFetchCell2Cell:
    @patch('empulse.datasets._remote._fetch_cell2cell_raw', return_value=_MOCK_RAW_CELL2CELL)
    def test_fetch_cell2cell_mocked(self, mock_fetch, tmp_path):
        ds = fetch_cell2cell(backend=pd, data_home=tmp_path)
        assert isinstance(ds, Dataset)
        assert ds.name == 'Cell2Cell Customer Churn'
        assert len(ds.target) == 3
        assert 'monthly_revenue' in ds.instance_costs
        assert ds.target_names == ['no churn', 'churn']
        assert len(ds.feature_names) == 56
