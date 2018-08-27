""" Contains full system tests for GamaRegressor """
import unittest
import numpy as np

from gama.ea.metrics import Metric


def metrics_test_suite():
    test_cases = [MetricsTestCase]
    return unittest.TestSuite(map(unittest.TestLoader().loadTestsFromTestCase, test_cases))


class MetricsTestCase(unittest.TestCase):
    """ Unit tests for Metric. """

    def setUp(self):
        self.y_true = np.asarray([1, 0, 0, 0, 1])
        self.y_true_str = np.asarray([str(x) for x in self.y_true])
        self.y_true_ohe = np.asarray([[0, 1], [1, 0], [1, 0], [1, 0], [0, 1]])

        self.y_1_mistake = np.asarray([1, 1, 0, 0, 1])
        self.y_1_mistake_str = np.asarray([str(x) for x in self.y_1_mistake])
        self.y_1_mistake_ohe = np.asarray([[0, 1], [0, 1], [1, 0], [1, 0], [0, 1]])
        self.y_probabilities = np.asarray([[0.1, 0.9], [0.2, 0.8], [0.7, 0.3], [0.95, 0.05], [0.1, 0.9]])

    def tearDown(self):
        pass

    def test_accuracy_numeric(self):
        accuracy_metric = Metric('accuracy')
        self.assertEqual(accuracy_metric.score(self.y_true, self.y_true), 1.0)
        self.assertEqual(accuracy_metric.maximizable_score(self.y_true, self.y_true), 1.0)

        self.assertEqual(accuracy_metric.score(self.y_true_str, self.y_1_mistake_str), 0.8)
        self.assertEqual(accuracy_metric.maximizable_score(self.y_true_str, self.y_1_mistake_str), 0.8)

    def test_accuracy_string(self):
        accuracy_metric = Metric('accuracy')
        self.assertEqual(accuracy_metric.score(self.y_true_str, self.y_true_str), 1.0)
        self.assertEqual(accuracy_metric.maximizable_score(self.y_true_str, self.y_true_str), 1.0)

        self.assertEqual(accuracy_metric.score(self.y_true_str, self.y_1_mistake_str), 0.8)
        self.assertEqual(accuracy_metric.maximizable_score(self.y_true_str, self.y_1_mistake_str), 0.8)

    def test_logloss_numeric(self):
        accuracy_metric = Metric('log_loss')
        self.assertAlmostEquals(accuracy_metric.score(self.y_true_ohe, self.y_true_ohe), 0.0)
        self.assertAlmostEquals(accuracy_metric.maximizable_score(self.y_true_ohe, self.y_true_ohe), 0.0)

        one_mistake_logloss = 6.907755278982137
        self.assertAlmostEquals(accuracy_metric.score(self.y_true_ohe, self.y_1_mistake_ohe), one_mistake_logloss)
        self.assertAlmostEquals(accuracy_metric.maximizable_score(self.y_true_ohe, self.y_1_mistake_ohe), -one_mistake_logloss)

        probabilities_logloss = 0.44562543641520713
        self.assertAlmostEquals(accuracy_metric.score(self.y_true_ohe, self.y_probabilities), probabilities_logloss)
        self.assertAlmostEquals(accuracy_metric.maximizable_score(self.y_true_ohe, self.y_probabilities), -probabilities_logloss)


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(metrics_test_suite())
