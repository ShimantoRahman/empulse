import random
from typing import Callable, Generator

from .test_metrics import BaseTestMetric, BaseTestRelationMetrics, Metric

from empulse.metrics import empa, mpa


class TestEMPA(BaseTestMetric.TestMetric):
    parameters = [
        {},  # default
        {"alpha": 10},
        {"beta": 0.02},
        {"contact_cost": 100},
        {"sales_cost": 1000},
        {"direct_selling": 0.5},
        {
            "direct_selling": 0.0,
            "commission": 0.5
        },
        {
            "alpha": 10,
            "beta": 0.0012,
            "contact_cost": 80,
            "sales_cost": 400,
            "direct_selling": 0.3,
            "commission": 0.2,
        },
    ]
    expected_values = {
        "perfect_prediction": (3725.0000000021705, 0.5),
        "incorrect_prediction": (3700.0, 1.0),
        "half_correct_prediction": (3700.0, 1.0),
        "different_parameters": [
            (2467.422967103394, 0.8774901985912619),
            (2020.9862113961788, 0.8751818393818439),
            (24.033377408837765, 0.25971110011640897),
            (2423.857763376393, 0.856980037413865),
            (2300.009595128851, 0.8766236068510703),
            (2417.1959571513835, 0.8775861680061278),
            (1295.5538077904482, 0.8659949979925511),
            (2289.417003293194, 0.8673144783905945),
        ],
    }

    @property
    def metric(self) -> Callable:
        return empa

    def test_bad_parameters(self):
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], alpha=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], beta=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], sales_cost=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], contact_cost=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], direct_selling=5)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], direct_selling=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], commission=5)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], commission=-1)


class TestMPA(BaseTestMetric.TestMetric):

    parameters = [
        {},  # default
        {"contribution": 2_000},
        {"contact_cost": 100},
        {"sales_cost": 1000},
        {"direct_selling": 0.5},
        {
            "direct_selling": 0.0,
            "commission": 0.5
        },
        {
            "contribution": 12_000,
            "contact_cost": 80,
            "sales_cost": 400,
            "direct_selling": 0.3,
            "commission": 0.2,
        },
    ]
    expected_values = {
        "perfect_prediction": (3725.0, 0.5),
        "incorrect_prediction": (3700.0, 1.0),
        "half_correct_prediction": (3700.0, 1.0),
        "different_parameters": [
            (2467.420814479638, 0.8778280542986425),
            (462.21719457013575, 0.665158371040724),
            (2423.529411764706, 0.8778280542986425),
            (2300.0, 0.8778280542986425),
            (2417.1945701357467, 0.8778280542986425),
            (1295.475113122172, 0.8778280542986425),
            (3345.158371040724, 0.8778280542986425),
        ],
    }

    @property
    def metric(self) -> Metric:
        return mpa

    def test_bad_parameters(self):
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], contribution=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], sales_cost=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], contact_cost=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], direct_selling=5)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], direct_selling=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], commission=5)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], commission=-1)


class TestRelationAcquisitionMetrics(BaseTestRelationMetrics.TestRelationshipMetrics):

    @property
    def stochastic_metric(self) -> Metric:
        return empa

    @property
    def deterministic_metric(self) -> Metric:
        return mpa

    def generate_parameters(self) -> Generator[dict, None, None]:
        """Generates parameters for the metric."""
        while True:
            yield {
                "alpha": random.uniform(0.00001, 100),
                "beta": random.uniform(0.00001, 100),
                "sales_cost": random.uniform(0, 500),
                "contact_cost": random.uniform(0, 100),
                "commission": random.uniform(0, 1),
                "direct_selling": random.uniform(0, 1),
            }

    def to_deterministic_params(self, params: dict) -> dict:
        """Converts the parameters to deterministic values."""
        deterministic_params = params.copy()
        deterministic_params["contribution"] = params['alpha'] / params['beta']
        del deterministic_params["alpha"]
        del deterministic_params["beta"]
        return deterministic_params
