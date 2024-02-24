import random
from typing import Generator

from empulse.metrics import empc, mpc
from .test_metrics import BaseTestMetric, Metric, BaseTestRelationMetrics


class TestEMPC(BaseTestMetric.TestMetric):

    parameters = [
        {},  # default
        {"alpha": 2},
        {"beta": 5},
        {"clv": 50},
        {"incentive_cost": 100},
        {"contact_cost": 15},
        {
            "alpha": 10,
            "beta": 7,
            "clv": 300,
            "incentive_cost": 50,
            "contact_cost": 25,
        },
    ]
    expected_values = {
        "perfect_prediction": (28.0000000000391, 0.499999999728095),
        "incorrect_prediction": (22.5007912244511, 0.9991601018889205),
        "half_correct_prediction": (22.5007912244511, 0.9991601018889205),
        "different_parameters": [
            (14.9946647690104, 0.5807038600143406),
            (5.04518077557418, 0.3984799748965802),
            (30.2276738879406, 0.6595939761685953),
            (1.62729137977619, 0.3397172176773871),
            (1.95497116583216, 0.11706222855204537),
            (8.47435415117762, 0.40263339050268493),
            (23.8505727106226, 0.4360480478313282),
        ],
    }

    @property
    def metric(self) -> Metric:
        return empc

    def test_bad_parameters(self):
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], alpha=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], beta=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], clv=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], clv=1)  # clv < incentive_cost
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], incentive_cost=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], contact_cost=-1)


class TestMPC(BaseTestMetric.TestMetric):

    parameters = [
        {},  # default
        {"accept_rate": 0.2},
        {"clv": 50},
        {"incentive_cost": 100},
        {"contact_cost": 15},
        {
            "accept_rate": 0.9,
            "clv": 300,
            "incentive_cost": 50,
            "contact_cost": 25,
        },
    ]
    expected_values = {
        "perfect_prediction": (28.0, 0.5),
        "incorrect_prediction": (22.5, 1.0),
        "half_correct_prediction": (22.5, 1.0),
        "different_parameters": [
            (14.814479638009047, 0.665158371040724),
            (9.02262443438914, 0.4434389140271493),
            (1.493212669683258, 0.4434389140271493),
            (1.9095022624434383, 0.12669683257918551),
            (8.316742081447961, 0.4434389140271493),
            (46.380090497737555, 0.4434389140271493),
            (23.8505727106226, 0.4360480478313282),
        ],
    }

    @property
    def metric(self) -> Metric:
        return mpc

    def test_bad_parameters(self):
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], accept_rate=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], accept_rate=2)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], clv=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], clv=1)  # clv < incentive_cost
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], incentive_cost=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], contact_cost=-1)


class TestRelationChurnMetrics(BaseTestRelationMetrics.TestRelationshipMetrics):

    @property
    def stochastic_metric(self) -> Metric:
        return empc

    @property
    def deterministic_metric(self) -> Metric:
        return mpc

    def generate_parameters(self) -> Generator[dict, None, None]:
        """Generates parameters for the metric."""
        while True:
            incentive_cost = random.uniform(0, 500)
            contact_cost = random.uniform(0, 100)
            clv = random.uniform(0, 1000)
            if clv < contact_cost + incentive_cost:
                clv = contact_cost + incentive_cost + 1
            yield {
                "alpha": random.uniform(0.00001, 100),
                "beta": random.uniform(0.00001, 100),
                "clv": clv,
                "incentive_cost": incentive_cost,
                "contact_cost": contact_cost,
            }

    def to_deterministic_params(self, params: dict) -> dict:
        """Converts the parameters to deterministic values."""
        deterministic_params = params.copy()
        deterministic_params["accept_rate"] = params['alpha'] / (params['alpha'] + params['beta'])
        del deterministic_params["alpha"]
        del deterministic_params["beta"]
        return deterministic_params
