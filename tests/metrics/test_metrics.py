import unittest
import random
from itertools import islice
from typing import Generator

import numpy as np


# wrap with empty class to avoid unittest discovery
class BaseTestMetric:
    class TestMetric(unittest.TestCase):
        path_predictions = "tests/data/predictions.dat"
        parameters = {}
        expected_values = {}

        @property
        def metric(self):
            return ...

        def assertAlmostEqualMetric(self, generated: tuple[float, float], expected: tuple[float, float]):
            """Assert that the generated value and threshold is equal to the expected value and threshold."""
            self.assertAlmostEqual(generated[0], expected[0])
            self.assertAlmostEqual(generated[1], expected[1])

        def test_perfect_prediction(self):
            self.assertAlmostEqualMetric(
                self.metric([0, 1] * 10, [0, 1] * 10),
                self.expected_values["perfect_prediction"]
            )
            self.assertAlmostEqualMetric(
                self.metric([1, 0] * 10, [1, 0] * 10),
                self.expected_values["perfect_prediction"]
            )

        def test_incorrect_prediction(self):
            self.assertAlmostEqualMetric(
                self.metric([0, 1] * 10, [1, 0] * 10),
                self.expected_values["incorrect_prediction"]
            )

        def test_half_correct_prediction(self):
            self.assertAlmostEqualMetric(
                self.metric([0, 1] * 10, [1, 1] * 10),
                self.expected_values["half_correct_prediction"]
            )
            self.assertAlmostEqualMetric(
                self.metric([1, 0] * 10, [1, 1] * 10),
                self.expected_values["half_correct_prediction"]
            )
            self.assertAlmostEqualMetric(
                self.metric([0, 1] * 10, [0, 0] * 10),
                self.expected_values["half_correct_prediction"]
            )
            self.assertAlmostEqualMetric(
                self.metric([1, 0] * 10, [0, 0] * 10),
                self.expected_values["half_correct_prediction"]
            )
            self.assertAlmostEqualMetric(
                self.metric([0, 1] * 10, [0.5, 0.5] * 10),
                self.expected_values["half_correct_prediction"]
            )
            self.assertAlmostEqualMetric(
                self.metric([1, 0] * 10, [0.5, 0.5] * 10),
                self.expected_values["half_correct_prediction"]
            )

        def test_arraylikes(self):
            import pandas as pd
            self.assertAlmostEqualMetric(
                self.metric((0, 1) * 10, (0, 1) * 10),
                self.expected_values["perfect_prediction"]
            )
            self.assertAlmostEqualMetric(
                self.metric(pd.Series([0, 1] * 10), pd.Series([0, 1] * 10)),
                self.expected_values["perfect_prediction"]
            )

        def test_all_y_true_same_value(self):
            with self.assertRaises(ValueError):
                self.metric([0, 0], [0.25, 0.75])
            with self.assertRaises(ValueError):
                self.metric([1, 1], [0.25, 0.75])

        def test_missing_y_true(self):
            with self.assertRaises(ValueError):
                self.metric([1, 0, np.nan], [0.25, 0.75, 0.5])
            with self.assertRaises(TypeError):  # type error because array will be of type object
                self.metric([1, 0, None], [0.25, 0.75, 0.5])

        def test_missing_y_score(self):
            with self.assertRaises(ValueError):
                self.metric([1, 0, 1], [0.25, 0.75, np.nan])
            with self.assertRaises(TypeError):  # type error because array will be of type object
                self.metric([1, 0, 1], [0.25, 0.75, None])

        def test_infinite_y_true(self):
            with self.assertRaises(ValueError):
                self.metric([1, 0, np.inf], [0.25, 0.75, 0.5])

        def test_infinite_y_score(self):
            with self.assertRaises(ValueError):
                self.metric([1, 0, 1], [0.25, 0.75, np.inf])

        def test_non_numeric_y_true(self):
            with self.assertRaises(TypeError):
                self.metric(["a", "b"], [0.25, 0.75])

        def test_non_numeric_y_score(self):
            with self.assertRaises(TypeError):
                self.metric([0, 1], ["a", "b"])

        def test_unequal_array_lengths(self):
            with self.assertRaises(ValueError):
                self.metric([0, 0, 1], [0.25, 0.75])
            with self.assertRaises(ValueError):
                self.metric([0, 1], [0.25, 0.5, 0.75])

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
                    self.assertAlmostEqualMetric(self.metric(y_true, y_score, **params), expected)


# wrap with empty class to avoid unittest discovery
class BaseTestRelationMetrics:
    class TestRelationshipMetrics(unittest.TestCase):

        @property
        def stochastic_metric(self):
            return ...

        @property
        def deterministic_metric(self):
            return ...

        def generate_data(self) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
            """Generates data for the metric."""
            while True:
                N_SAMPLES = 1000
                positive_class_prior = random.random()
                n_positive_class = int(positive_class_prior * N_SAMPLES)
                if n_positive_class == 0:
                    n_positive_class = 1
                n_negative_class = N_SAMPLES - n_positive_class
                random_param = lambda x: random.random() * x
                positive_class_predictions = np.random.beta(random_param(20), random_param(20), n_positive_class)
                negative_class_predictions = np.random.beta(random_param(20), random_param(20), n_negative_class)
                predictions = np.concatenate([positive_class_predictions, negative_class_predictions])
                labels = np.concatenate([
                    np.ones(n_positive_class, dtype=np.int8),
                    np.zeros(n_negative_class, dtype=np.int8)
                ])
                yield labels, predictions

        def generate_parameters(self) -> Generator[dict, None, None]:
            """Generates parameters for the metric."""
            ...

        def to_deterministic_params(self, params: dict) -> dict:
            """Converts the parameters to deterministic values."""
            ...

        def test_deterministic_metric_lower_bound(self):
            """Test whether the deterministic metric is lower or equal than the stochastic metric."""
            for params, (y_true, y_pred) in zip(islice(self.generate_parameters(), 1000),
                                                islice(self.generate_data(), 1000)):
                stochastic_score, _ = self.stochastic_metric(y_true, y_pred, **params)
                params = self.to_deterministic_params(params)
                deterministic_score, _ = self.deterministic_metric(y_true, y_pred, **params)
                with self.subTest(params):
                    self.assertTrue(deterministic_score <= stochastic_score + 0.000001)


if __name__ == "__main__":
    unittest.main()
