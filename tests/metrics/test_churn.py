import random
from typing import Generator

import numpy as np
from empulse.metrics import empc, mpc, mpc_cost_score, empb
from .test_metrics import BaseTestMetric, BaseTestRelationMetrics


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
    def metric(self):
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
        ],
    }

    @property
    def metric(self):
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
    def stochastic_metric(self):
        return empc

    @property
    def deterministic_metric(self):
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


class TestMPCScore(BaseTestMetric.TestMetric):
    parameters = [
        {},  # default
        {"accept_rate": 0.2},
        {"clv": 50},
        {"incentive_fraction": 0.5},
        {"contact_cost": 15},
        {
            "accept_rate": 0.9,
            "clv": 300,
            "incentive_fraction": 0.1,
            "contact_cost": 25,
        },
        {"clv": np.arange(221)}  # array-like clv
    ]
    expected_values = {
        "perfect_prediction": -28.0,
        "incorrect_prediction": 5.5,
        "different_parameters": [
            -10.080448865102996,
            -6.089389456506361,
            -2.247842651901284,
            9.35838608395974,
            -4.998083663446319,
            -37.37878730798912,
            -6.02363958556662,
        ],
    }

    def assertAlmostEqualMetric(self, generated: float, expected: float):
        self.assertAlmostEqual(generated, expected)

    @property
    def metric(self):
        return mpc_cost_score

    def test_half_correct_prediction(self):
        self.assertAlmostEqualMetric(
            self.metric([0, 1] * 10, [1, 1] * 10),
            -22.5
        )
        self.assertAlmostEqualMetric(
            self.metric([1, 0] * 10, [1, 1] * 10),
            -22.5
        )
        self.assertAlmostEqualMetric(
            self.metric([0, 1] * 10, [0, 0] * 10),
            0.0
        )
        self.assertAlmostEqualMetric(
            self.metric([1, 0] * 10, [0, 0] * 10),
            0.0
        )
        self.assertAlmostEqualMetric(
            self.metric([0, 1] * 10, [0.5, 0.5] * 10),
            -11.25
        )
        self.assertAlmostEqualMetric(
            self.metric([1, 0] * 10, [0.5, 0.5] * 10),
            -11.25
        )

    def test_bad_parameters(self):
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], accept_rate=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], accept_rate=2)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], incentive_fraction=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], incentive_fraction=2)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], contact_cost=-1)


class TestEMPB(BaseTestMetric.TestMetric):
    parameters = [
        {},  # default
        {"alpha": 2},
        {"beta": 5},
        {"incentive_fraction": 0.9},
        {"contact_cost": 15},
        {
            "alpha": 10,
            "beta": 7,
            "incentive_fraction": 0.02,
            "contact_cost": 25,
        },
    ]
    expected_values = {
        "perfect_prediction": (525.0, 0.475),
        "incorrect_prediction": (325.0, 0.975),
        "different_parameters": [
            (1202.3649999999961, 0.445),
            (297.00623162923307, 0.15),
            (2748.0045454553197, 0.445),
            (0.0, 0.0),
            (1202.3649999999961, 0.445),
            (2541.7823529411744, 0.445),
        ],
    }

    @property
    def metric(self):
        return empb

    def test_perfect_prediction(self):
        self.assertAlmostEqualMetric(
            self.metric([0, 1] * 10, [0, 1] * 10, clv=[100, 200] * 10),
            self.expected_values["perfect_prediction"]
        )
        self.assertAlmostEqualMetric(
            self.metric([1, 0] * 10, [1, 0] * 10, clv=[200, 100] * 10),
            self.expected_values["perfect_prediction"]
        )

    def test_incorrect_prediction(self):
        self.assertAlmostEqualMetric(
            self.metric([0, 1] * 10, [1, 0] * 10, clv=[100, 200] * 10),
            self.expected_values["incorrect_prediction"]
        )

    def test_half_correct_prediction(self):
        self.assertAlmostEqualMetric(
            self.metric([0, 1] * 10, [1, 1] * 10, clv=[100, 200] * 10),
            (345.0, 0.925)
        )
        self.assertAlmostEqualMetric(
            self.metric([1, 0] * 10, [1, 1] * 10, clv=[200, 100] * 10),
            (325.0, 0.975)
        )
        self.assertAlmostEqualMetric(
            self.metric([0, 1] * 10, [0, 0] * 10, clv=[100, 200] * 10),
            (345.0, 0.925)
        )
        self.assertAlmostEqualMetric(
            self.metric([1, 0] * 10, [0, 0] * 10, clv=[200, 100] * 10),
            (325.0, 0.975)
        )
        self.assertAlmostEqualMetric(
            self.metric([0, 1] * 10, [0.5, 0.5] * 10, clv=[100, 200] * 10),
            (345.0, 0.925)
        )
        self.assertAlmostEqualMetric(
            self.metric([1, 0] * 10, [0.5, 0.5] * 10, clv=[200, 100] * 10),
            (325.0, 0.975)
        )

    def test_arraylikes(self):
        import pandas as pd
        self.assertAlmostEqualMetric(
            self.metric((0, 1) * 10, (0, 1) * 10, clv=[100, 200] * 10),
            self.expected_values["perfect_prediction"]
        )
        self.assertAlmostEqualMetric(
            self.metric(pd.Series([0, 1] * 10), pd.Series([0, 1] * 10), clv=[100, 200] * 10),
            self.expected_values["perfect_prediction"]
        )

    def test_all_y_true_same_value(self):
        with self.assertRaises(ValueError):
            self.metric([0, 0], [0.25, 0.75], clv=[100, 200] * 10)
        with self.assertRaises(ValueError):
            self.metric([1, 1], [0.25, 0.75], clv=[100, 200] * 10)

    def test_missing_y_true(self):
        with self.assertRaises(ValueError):
            self.metric([1, 0, np.nan], [0.25, 0.75, 0.5], clv=[100, 200] * 10)
        with self.assertRaises(TypeError):  # type error because array will be of type object
            self.metric([1, 0, None], [0.25, 0.75, 0.5], clv=[100, 200] * 10)

    def test_missing_y_score(self):
        with self.assertRaises(ValueError):
            self.metric([1, 0, 1], [0.25, 0.75, np.nan], clv=[100, 200] * 10)
        with self.assertRaises(TypeError):  # type error because array will be of type object
            self.metric([1, 0, 1], [0.25, 0.75, None], clv=[100, 200] * 10)

    def test_infinite_y_true(self):
        with self.assertRaises(ValueError):
            self.metric([1, 0, np.inf], [0.25, 0.75, 0.5], clv=[100, 200] * 10)

    def test_infinite_y_score(self):
        with self.assertRaises(ValueError):
            self.metric([1, 0, 1], [0.25, 0.75, np.inf], clv=[100, 200] * 10)

    def test_non_numeric_y_true(self):
        with self.assertRaises(TypeError):
            self.metric(["a", "b"], [0.25, 0.75], clv=[100, 200] * 10)

    def test_non_numeric_y_score(self):
        with self.assertRaises(TypeError):
            self.metric([0, 1], ["a", "b"], clv=[100, 200] * 10)

    def test_unequal_array_lengths(self):
        with self.assertRaises(ValueError):
            self.metric([0, 0, 1], [0.25, 0.75], clv=[100, 200] * 10)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.5, 0.75], clv=[100, 200] * 10)

    def test_different_parameters(self):
        data = np.loadtxt(self.path_predictions)
        y_true = data[:, 0]
        y_score = data[:, 1]

        for params, expected in zip(self.parameters, self.expected_values["different_parameters"]):
            with self.subTest(params):
                self.assertAlmostEqualMetric(
                    self.metric(y_true, y_score, clv=np.arange(y_true.shape[0]), **params),
                    expected
                )

    def test_bad_parameters(self):
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], clv=[10, 20], alpha=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], clv=[10, 20], beta=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], clv=[10, 20], incentive_fraction=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], clv=[10, 20], incentive_fraction=2)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], clv=[10, 20], contact_cost=-1)
