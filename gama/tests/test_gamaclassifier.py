""" Contains full system tests for GamaClassifier """
import unittest

import numpy as np
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

from gama.utilities.generic.stopwatch import Stopwatch
from gama import GamaClassifier


def gamaclassifier_test_suite():
    test_cases = [GamaClassifierSystemTestCase]
    return unittest.TestSuite(map(unittest.TestLoader().loadTestsFromTestCase, test_cases))


# While we could derive statistics dynamically, we want to know if any changes ever happen, so we save them statically.
breast_cancer = dict(
    name='breast cancer',
    load=load_breast_cancer,
    test_size=143,
    n_classes=2,
    base_accuracy=0.62937,
    base_log_loss=12.80138
)

wine = dict(
    name='wine',
    load=load_wine,
    test_size=45,
    n_classes=3,
    base_accuracy=0.4,
    base_log_loss=20.72326,
)

class GamaClassifierSystemTestCase(unittest.TestCase):
    """ Contain complete system tests for Gama. """

    def setUp(self):
        self._fit_time_margin = 1.1

    def tearDown(self):
        pass

    def _test_dataset_problem(self, data, metric, labelled_y=False):
        X, y = data['load'](return_X_y=True)
        if labelled_y:
            databunch = data['load']()
            y = [databunch.target_names[c_i] for c_i in databunch.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

        gama = GamaClassifier(random_state=0, max_total_time=60, objectives=(metric, 'size'))
        with Stopwatch() as sw:
            gama.fit(X_train, y_train, auto_ensemble_n=5)

        self.assertLessEqual(sw.elapsed_time, 60 * self._fit_time_margin, 'fit must stay within 110% of allotted time.')

        class_predictions = gama.predict(X_test)
        self.assertTrue(isinstance(class_predictions, np.ndarray), 'predictions should be numpy arrays.')
        self.assertEqual(class_predictions.shape, (data['test_size'],), 'predict should return (N,) shaped array.')

        # Majority classifier on this split achieves 0.6293706293706294
        accuracy = accuracy_score(y_test, class_predictions)
        print(data['name'], metric, 'accuracy:', accuracy)
        self.assertGreaterEqual(accuracy, data['base_accuracy'],
                                'predictions should be at least as good as majority class.')

        class_probabilities = gama.predict_proba(X_test)
        self.assertTrue(isinstance(class_probabilities, np.ndarray), 'probability predictions should be numpy arrays.')
        self.assertEqual(class_probabilities.shape, (data['test_size'], data['n_classes']),
                         'predict_proba should return (N,K) shaped array.')

        # Majority classifier on this split achieves 12.80138131184662
        logloss = log_loss(y_test, class_probabilities)
        print(data['name'], metric, 'log-loss:', logloss)
        self.assertLessEqual(logloss, data['base_log_loss'],
                             'predictions should be at least as good as majority class.')

    def test_binary_classification_accuracy(self):
        self._test_dataset_problem(breast_cancer, 'accuracy')

    def test_binary_classification_logloss(self):
        self._test_dataset_problem(breast_cancer, 'log_loss')

    def test_multiclass_classification_accuracy(self):
        self._test_dataset_problem(wine, 'accuracy')

    def test_multiclass_classification_logloss(self):
        self._test_dataset_problem(wine, 'log_loss')
        
    def test_string_label_classification_accuracy(self):
        self._test_dataset_problem(breast_cancer, 'accuracy', labelled_y=True)

    def test_string_label_classification_log_loss(self):
        self._test_dataset_problem(breast_cancer, 'log_loss', labelled_y=True)


if __name__ == '__main__':
    import logging
    import sys
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)
    unittest.TextTestRunner(verbosity=2).run(gamaclassifier_test_suite)
