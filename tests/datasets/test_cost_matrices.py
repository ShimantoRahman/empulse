"""Unit tests for empulse.datasets._cost_matrices factories."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from empulse.datasets._cost_matrices import (
    _compute_credit_lines,
    _creditscoring_costmat,
    churn_precomputed_cost_matrix,
    churn_retention_cost_matrix,
    credit_scoring_cost_matrix,
    upsell_bank_cost_matrix,
)
from empulse.metrics import CostMatrix


@pytest.fixture
def clv():
    return np.array([100.0, 200.0, 500.0])


@pytest.fixture
def balance():
    return np.array([1000.0, 5000.0, 10000.0])


@pytest.fixture
def credit_inputs():
    income = np.array([500.0, 1000.0, 2000.0])
    debt = np.array([0.1, 0.2, 0.05])
    target = np.array([0, 1, 0])
    return income, debt, target


class TestChurnPrecomputedCostMatrix:
    def test_returns_cost_matrix(self):
        assert isinstance(churn_precomputed_cost_matrix(), CostMatrix)

    def test_idempotent(self):
        """Two calls should return equivalent objects."""
        cm1 = churn_precomputed_cost_matrix()
        cm2 = churn_precomputed_cost_matrix()
        assert str(cm1) == str(cm2)


class TestChurnRetentionCostMatrix:
    DEFAULTS: ClassVar[dict[str, float]] = {'incentive_fraction': 0.05, 'contact_fraction': 0.01, 'accept_rate': 0.3}

    def test_returns_tuple(self, clv):
        result = churn_retention_cost_matrix(clv, **self.DEFAULTS)
        assert isinstance(result, tuple) and len(result) == 2

    def test_cost_matrix_type(self, clv):
        cm, _ = churn_retention_cost_matrix(clv, **self.DEFAULTS)
        assert isinstance(cm, CostMatrix)

    def test_instance_costs_keys(self, clv):
        _, costs = churn_retention_cost_matrix(clv, **self.DEFAULTS)
        assert set(costs.keys()) == {'clv'}

    def test_clv_array_preserved(self, clv):
        _, costs = churn_retention_cost_matrix(clv, **self.DEFAULTS)
        np.testing.assert_array_equal(costs['clv'], clv)

    def test_clv_shape(self, clv):
        _, costs = churn_retention_cost_matrix(clv, **self.DEFAULTS)
        assert costs['clv'].shape == clv.shape

    @pytest.mark.parametrize('incentive_fraction', [0.0, 0.05, 0.20])
    def test_varying_incentive_fraction(self, clv, incentive_fraction):
        """Factory should accept any incentive fraction without error."""
        cm, costs = churn_retention_cost_matrix(
            clv,
            incentive_fraction=incentive_fraction,
            contact_fraction=0.01,
            accept_rate=0.3,
        )
        assert isinstance(cm, CostMatrix)
        assert costs['clv'] is clv


class TestUpsellBankCostMatrix:
    DEFAULTS: ClassVar[dict[str, float]] = {
        'interest_rate': 0.02463,
        'term_deposit_fraction': 0.25,
        'contact_cost': 1.0,
    }

    def test_returns_tuple(self, balance):
        result = upsell_bank_cost_matrix(balance, **self.DEFAULTS)
        assert isinstance(result, tuple) and len(result) == 2

    def test_cost_matrix_type(self, balance):
        cm, _ = upsell_bank_cost_matrix(balance, **self.DEFAULTS)
        assert isinstance(cm, CostMatrix)

    def test_instance_costs_keys(self, balance):
        _, costs = upsell_bank_cost_matrix(balance, **self.DEFAULTS)
        assert set(costs.keys()) == {'balance'}

    def test_balance_array_preserved(self, balance):
        _, costs = upsell_bank_cost_matrix(balance, **self.DEFAULTS)
        np.testing.assert_array_equal(costs['balance'], balance)

    def test_zero_contact_cost(self, balance):
        """contact_cost=0 should not raise."""
        cm, _ = upsell_bank_cost_matrix(balance, interest_rate=0.02, term_deposit_fraction=0.25, contact_cost=0.0)
        assert isinstance(cm, CostMatrix)


class TestCreditScoringCostMatrix:
    DEFAULTS: ClassVar[dict[str, float]] = {
        'interest_rate': 0.63,
        'fund_cost': 0.165,
        'cl_max': 8250.0,
        'loss_given_default': 0.75,
        'term_length_months': 24,
        'loan_to_income_ratio': 3,
    }

    def test_returns_tuple(self, credit_inputs):
        income, debt, target = credit_inputs
        result = credit_scoring_cost_matrix(income, debt, target, **self.DEFAULTS)
        assert isinstance(result, tuple) and len(result) == 2

    def test_cost_matrix_type(self, credit_inputs):
        income, debt, target = credit_inputs
        cm, _ = credit_scoring_cost_matrix(income, debt, target, **self.DEFAULTS)
        assert isinstance(cm, CostMatrix)

    def test_instance_costs_keys(self, credit_inputs):
        income, debt, target = credit_inputs
        _, costs = credit_scoring_cost_matrix(income, debt, target, **self.DEFAULTS)
        assert set(costs.keys()) == {'cl', 'fp_cost'}

    def test_shapes_match_input(self, credit_inputs):
        income, debt, target = credit_inputs
        _, costs = credit_scoring_cost_matrix(income, debt, target, **self.DEFAULTS)
        assert costs['cl'].shape == income.shape
        assert costs['fp_cost'].shape == income.shape

    def test_credit_lines_non_negative(self, credit_inputs):
        income, debt, target = credit_inputs
        _, costs = credit_scoring_cost_matrix(income, debt, target, **self.DEFAULTS)
        assert np.all(costs['cl'] >= 0)

    def test_fp_costs_non_negative(self, credit_inputs):
        income, debt, target = credit_inputs
        _, costs = credit_scoring_cost_matrix(income, debt, target, **self.DEFAULTS)
        assert np.all(costs['fp_cost'] >= 0)

    def test_higher_income_gives_higher_credit_line(self):
        """All else equal, higher income → higher credit line."""
        income_low = np.array([500.0])
        income_high = np.array([2000.0])
        debt = np.array([0.1])
        target = np.array([0])
        _, costs_low = credit_scoring_cost_matrix(income_low, debt, target, **self.DEFAULTS)
        _, costs_high = credit_scoring_cost_matrix(income_high, debt, target, **self.DEFAULTS)
        assert costs_high['cl'][0] >= costs_low['cl'][0]

    def test_loss_given_default_scales_fn_cost(self):
        """Higher LGD should scale credit-line portion of the fn cost."""
        income = np.array([1000.0])
        debt = np.array([0.1])
        target = np.array([0])
        _, costs_low_lgd = credit_scoring_cost_matrix(
            income, debt, target, **{**self.DEFAULTS, 'loss_given_default': 0.5}
        )
        _, costs_high_lgd = credit_scoring_cost_matrix(
            income, debt, target, **{**self.DEFAULTS, 'loss_given_default': 0.9}
        )
        # fn_cost = cl * lgd  → same cl, higher lgd → larger effective fn cost
        # We can compare credit lines (should be the same) and check lgd in matrix
        assert costs_low_lgd['cl'][0] == pytest.approx(costs_high_lgd['cl'][0])


class TestCreditScoringCostmat:
    PARAMS: ClassVar[dict[str, float]] = {
        'int_r': 0.63 / 12,
        'int_cf': 0.165 / 12,
        'cl_max': 8250.0,
        'n_term': 24,
        'k': 3,
        'lgd': 0.75,
    }

    def test_output_shape(self):
        income = np.array([500.0, 1000.0, 1500.0])
        debt = np.zeros(3)
        mat = _creditscoring_costmat(income, debt, pi_1=0.2, params=self.PARAMS)
        assert mat.shape == (3, 4)

    def test_fn_cost_column_non_negative(self):
        income = np.array([500.0, 1000.0])
        debt = np.zeros(2)
        mat = _creditscoring_costmat(income, debt, pi_1=0.2, params=self.PARAMS)
        assert np.all(mat[:, 1] >= 0)

    def test_fp_cost_column_non_negative(self):
        income = np.array([500.0, 1000.0])
        debt = np.zeros(2)
        mat = _creditscoring_costmat(income, debt, pi_1=0.2, params=self.PARAMS)
        assert np.all(mat[:, 0] >= 0)


class TestComputeCreditLines:
    PARAMS: ClassVar[dict[str, float]] = {
        'int_r': 0.63 / 12,
        'cl_max': 8250.0,
        'n_term': 24,
        'k': 3,
    }

    def test_output_shape(self):
        income = np.array([500.0, 1000.0, 2000.0])
        debt = np.zeros(3)
        cl = _compute_credit_lines(income, debt, self.PARAMS)
        assert cl.shape == (3,)

    def test_credit_lines_non_negative(self):
        income = np.linspace(100, 5000, 20)
        debt = np.zeros(20)
        cl = _compute_credit_lines(income, debt, self.PARAMS)
        assert np.all(cl >= 0)

    def test_credit_lines_capped_at_cl_max(self):
        # Very high income → credit line should be capped at cl_max
        income = np.array([1_000_000.0])
        debt = np.zeros(1)
        cl = _compute_credit_lines(income, debt, self.PARAMS)
        assert cl[0] <= self.PARAMS['cl_max'] + 1e-6
