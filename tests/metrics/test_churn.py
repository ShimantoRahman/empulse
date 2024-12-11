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
        y_true = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                  0, 1, 0, 0,
                  1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                  0, 0, 0, 1,
                  1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
                  1, 0, 0, 0,
                  0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                  0, 0, 1, 0,
                  0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                  0, 0, 0, 1,
                  0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1,
                  1, 0, 0]
        y_score = [0.021992219683176, 0.0261283707394984, 0.617550469398662, 0.507844207594384, 0.0689278415230867,
                   0.634666532184113, 0.0333194568213845, 0.172182005960011, 0.573240925914627, 0.171196797249527,
                   0.140117673851262, 0.524708416048729, 0.110873962537451, 0.97421711638746, 0.0363991529189672,
                   0.0369268806132579, 0.0547088125342946, 0.814621308201928, 0.119531604319641, 0.107642371144209,
                   0.0166290869179551, 0.430180757181547, 0.398788439352216, 0.24468921596579, 0.0399946322344566,
                   0.205358184240838, 0.737755170890566, 0.120406903890372, 0.252361035025407, 0.301483557295849,
                   0.130322152269363, 0.529561011317475, 0.228540311304375, 0.289372326036415, 0.741898786489765,
                   0.103663747412982, 0.159053009511051, 0.177013692327633, 0.596561315187138, 0.68791338415648,
                   0.0764654110832665, 0.0380429047162668, 0.0955794593610204, 0.308606923152478, 0.953632418463787,
                   0.758135038783222, 0.0394769117532799, 0.995240631824313, 0.273044259437478, 0.183078179518509,
                   0.0211745215685914, 0.953505732985242, 0.610883178976933, 0.503535526852141, 0.170239420661519,
                   0.0662416937367266, 0.952569127666721, 0.0235124930614956, 0.389843327997835, 0.402892556387082,
                   0.45409263606926, 0.0442499705696209, 0.759598734548082, 0.995607413363586, 0.666354296358261,
                   0.0373860753910581, 0.875427700148495, 0.0312333174412777, 0.498777766954593, 0.500497017571849,
                   0.849971946355577, 0.637359280744781, 0.0296903193364025, 0.181761985231305, 0.963888010432597,
                   0.196596621421334, 0.675297846906073, 0.904176051620105, 0.332259139268038, 0.180421711491755,
                   0.112724618252376, 0.483760804866299, 0.551229444922031, 0.0388069929536769, 0.1511996740988,
                   0.804054237811406, 0.194177661892815, 0.348575449499428, 0.725153691304662, 0.375548457677746,
                   0.338073172935937, 0.123283599859036, 0.223736202162603, 0.946683017067355, 0.216056810823931,
                   0.23859048208279, 0.152281849808273, 0.589678698555602, 0.0222761439945392, 0.130558648851825,
                   0.638897143483432, 0.208228392619354, 0.316843558106919, 0.915287559980507, 0.207860634990667,
                   0.546852875418941, 0.045944125119147, 0.990131955944472, 0.0294883427491257, 0.087090453978068,
                   0.0220009700400732, 0.083692989538496, 0.0348756949481749, 0.361769825276756, 0.205476220689286,
                   0.100269237406797, 0.087344945805354, 0.840814107315031, 0.133027003133436, 0.855985030148661,
                   0.167775025538271, 0.764539704395445, 0.656383505398512, 0.15561205601821, 0.0597424375476083,
                   0.175077551108632, 0.545956958818802, 0.0403563461533558, 0.0320401836850442, 0.0437654936091769,
                   0.219964964192491, 0.998060547171893, 0.0776128454283348, 0.0132921699364248, 0.291072027110432,
                   0.355301191253285, 0.0698230598140226, 0.0502067185947528, 0.628847897068818, 0.207090577985603,
                   0.546256296690515, 0.162633345257203, 0.385819544028046, 0.52366763615621, 0.0975739075113509,
                   0.0534048605432653, 0.352871497118956, 0.0432674910111086, 0.12640028022513, 0.0859671143275691,
                   0.11102000863473, 0.287507619711779, 0.489038860711864, 0.812509235682482, 0.0144399383814023,
                   0.0334806593267197, 0.104836371130298, 0.399021697336736, 0.46793625229057, 0.133789972891713,
                   0.956483540507921, 0.952616656918121, 0.082480835719772, 0.0940482501576516, 0.049892554853392,
                   0.675061034197803, 0.136399152900304, 0.154474153673202, 0.531920853672081, 0.147573487747709,
                   0.836471007924649, 0.922881501476747, 0.169819535736351, 0.62458206076582, 0.0819786680244113,
                   0.251914613500917, 0.0218373572001962, 0.514684405971582, 0.501085110678586, 0.82749345131409,
                   0.651988072986418, 0.104090597009933, 0.0525477505214838, 0.0845612209633105, 0.601721283224274,
                   0.015312908385536, 0.665154673507338, 0.741111483518058, 0.456023535483595, 0.0362621607677878,
                   0.1730805893012, 0.318586832670877, 0.741688675961793, 0.211482333518581, 0.742567383536038,
                   0.130621394573414, 0.0298506530347167, 0.487408551860671, 0.196911036208975, 0.142830858793263,
                   0.29857631503355, 0.819791019639498, 0.943983828869089, 0.884707910351011, 0.154967258276995,
                   0.339931767822726, 0.299198463354478, 0.918752242547175, 0.169886998769205, 0.102694015426334,
                   0.887166980555358, 0.0711477127378713, 0.991733542344771, 0.564759907903372, 0.781921391247473,
                   0.241198494244602, 0.811850768764747, 0.511660796813233, 0.947702671244453, 0.286781442903701,
                   0.0518662316348774]
        y_true = np.array(y_true)
        y_score = np.array(y_score)

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
