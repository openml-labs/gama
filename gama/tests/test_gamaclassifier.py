""" Contains full system tests for GamaClassifier """
import unittest

import numpy as np
from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

from gama import GamaClassifier


def gamaclassifier_test_suite():
    test_cases = [GamaClassifierSystemTestCase]
    return unittest.TestSuite(map(unittest.TestLoader().loadTestsFromTestCase, test_cases))


class GamaClassifierSystemTestCase(unittest.TestCase):
    """ Contain complete system tests for Gama. """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_binary_classification_accuracy(self):
        gama = GamaClassifier(random_state=0, max_total_time=60, objectives=('accuracy', 'size'))
        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

        gama.fit(X_train, y_train)
        class_predictions = gama.predict(X_test)
        self.assertTrue(isinstance(class_predictions, np.ndarray), 'predictions should be numpy arrays.')
        self.assertEqual(class_predictions.shape, (143,), 'predict should return (N,) shaped array.')

        # Majority classifier on this split achieves 0.6293706293706294
        accuracy = accuracy_score(y_test, class_predictions)
        print('breast cancer accuracy accuracy:', accuracy)
        self.assertGreaterEqual(accuracy, 0.6, 'predictions should be at least as good as majority class.')

        class_probabilities = gama.predict_proba(X_test)
        self.assertTrue(isinstance(class_probabilities, np.ndarray), 'probability predictions should be numpy arrays.')
        self.assertEqual(class_probabilities.shape, (143, 2), 'predict_proba should return (N,K) shaped array.')

        # Majority classifier on this split achieves 12.80138131184662
        logloss = log_loss(y_test, class_probabilities)
        print('breast cancer accuracy log-loss:', logloss)
        self.assertLessEqual(logloss, 13, 'predictions should be at least as good as majority class.')
