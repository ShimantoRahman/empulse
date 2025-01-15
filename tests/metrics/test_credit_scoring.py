from empulse.metrics import empcs, mpcs
from .test_metrics import BaseTestMetric


class TestEMPCS(BaseTestMetric.TestMetric):
    parameters = [
        {},  # default
        {"success_rate": 0.7},
        {"default_rate": 0.01},
        {"roi": 0.1},
        {
            "success_rate": 0.01,
            "default_rate": 0.7,
            "roi": 0.7,
        },
    ]
    expected_values = {
        "perfect_prediction": (0.1375, 0.225),
        "incorrect_prediction": (0.08412689400000001, 0.35746),
        "half_correct_prediction": (0.08412689400000001, 0.35746),
        "different_parameters": [
            (0.06275887634143887, 0.17452781458718136),
            (0.04653003664242532, 0.11873470465098149),
            (0.05009375482148049, 0.16809417828645784),
            (0.07738088359641128, 0.25536875341114096),
            (0.14636925360203037, 0.38412016747189753),
        ],
    }

    @property
    def metric(self):
        return empcs

    def test_bad_parameters(self):
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], success_rate=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], success_rate=2)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], default_rate=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], default_rate=2)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], roi=-1)


class TestMPCS(BaseTestMetric.TestMetric):
    parameters = [
        {},  # default
        {"loan_lost_rate": 0.7},
        {"roi": 0.1},
        {
            "loan_lost_rate": 0.01,
            "roi": 0.7,
        },
    ]
    expected_values = {
        "perfect_prediction": (0.1375, 0.5),
        "incorrect_prediction": (0.005299999999999999, 1.0),
        "half_correct_prediction": (0.005299999999999999, 1.0),
        "different_parameters": [
            (0.03896108597285068, 0.4434389140271493),
            (0.16203800904977372, 0.4434389140271493),
            (0.06425339366515838, 0.4434389140271493),
            (4.524886877828054e-05, 0.004524886877828055),
        ],
    }

    @property
    def metric(self):
        return mpcs

    def test_bad_parameters(self):
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], loan_lost_rate=-1)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], loan_lost_rate=2)
        with self.assertRaises(ValueError):
            self.metric([0, 1], [0.25, 0.75], roi=-1)
