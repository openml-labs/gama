""" Contains full system tests for GamaClassifier """
import unittest

import arff
import numpy as np
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

from gama.utilities.generic.stopwatch import Stopwatch
from gama import GamaClassifier


def gamaclassifier_test_suite():
    test_cases = [GamaClassifierARFFSystemTestCase]#, GamaClassifierSystemTestCase]
    return unittest.TestSuite(map(unittest.TestLoader().loadTestsFromTestCase, test_cases))


# While we could derive statistics dynamically, we want to know if any changes ever happen, so we save them statically.
breast_cancer = dict(
    name='breast_cancer',
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

iris_arff = dict(
    name='iris',
    train='gama/tests/data/iris_train.arff',
    test='gama/tests/data/iris_test.arff',
    test_size=50,
    n_classes=3,
    base_accuracy=0.3333,
    base_log_loss=1.09861
)

diabetes_arff = dict(
    name='diabetes',
    train='gama/tests/data/diabetes_train.arff',
    test='gama/tests/data/diabetes_test.arff',
    test_size=150,
    n_classes=2,
    base_accuracy=0.65104,
    base_log_loss=0.63705
)

class GamaClassifierSystemTestCase(unittest.TestCase):
    """ Contain complete system tests for Gama from numpy data. """

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
        """ GamaClassifier can do binary classification with predict metric from numpy data. """
        self._test_dataset_problem(breast_cancer, 'accuracy')

    def test_binary_classification_logloss(self):
        """ GamaClassifier can do binary classification with predict-proba metric from numpy data. """
        self._test_dataset_problem(breast_cancer, 'log_loss')

    def test_multiclass_classification_accuracy(self):
        """ GamaClassifier can do multi-class with predict metric from numpy data. """
        self._test_dataset_problem(wine, 'accuracy')

    def test_multiclass_classification_logloss(self):
        """ GamaClassifier can do multi-class with predict-proba metric from numpy data. """
        self._test_dataset_problem(wine, 'log_loss')
        
    def test_string_label_classification_accuracy(self):
        """ GamaClassifier can work with string-like target labels when using predict-metric from numpy data. """
        self._test_dataset_problem(breast_cancer, 'accuracy', labelled_y=True)

    def test_string_label_classification_log_loss(self):
        """ GamaClassifier can work with string-type target labels when using predict-proba metric from numpy data. """
        self._test_dataset_problem(breast_cancer, 'log_loss', labelled_y=True)

    def test_missing_value_classification(self):
        """ GamaClassifier handles missing data from numpy data. """
        data = breast_cancer
        metric = 'log_loss'

        X, y = data['load'](return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
        X_train[1:300:2, 0] = X_train[2:300:5, 1] = float("NaN")
        X_test[1:100:2, 0] = X_test[2:100:5, 1] = float("NaN")

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


class GamaClassifierARFFSystemTestCase(unittest.TestCase):
    """ Contain complete system tests for Gama from ARFF data. """

    def setUp(self):
        self._fit_time_margin = 1.1

    def tearDown(self):
        pass

    def _test_dataset_problem(self, data, metric):
        train_path = 'data/{}_train.arff'.format(data['name'])
        test_path = 'data/{}_test.arff'.format(data['name'])
        gama = GamaClassifier(random_state=0, max_total_time=60, objectives=(metric, 'size'))

        X, y = data['load'](return_X_y=True)
        #if labelled_y:
        #    databunch = data['load']()
        #    y = [databunch.target_names[c_i] for c_i in databunch.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
        y_test = [str(val) for val in y_test]

        with Stopwatch() as sw:
            gama.fit(arff_file_path=train_path, auto_ensemble_n=5)

        self.assertLessEqual(sw.elapsed_time, 60 * self._fit_time_margin, 'fit must stay within 110% of allotted time.')

        class_predictions = gama.predict(arff_file_path=test_path)
        self.assertTrue(isinstance(class_predictions, np.ndarray), 'predictions should be numpy arrays.')
        self.assertEqual(class_predictions.shape, (data['test_size'],), 'predict should return (N,) shaped array.')

        accuracy = accuracy_score(y_test, class_predictions)
        print(data['name'], metric, 'accuracy:', accuracy)
        self.assertGreaterEqual(accuracy, data['base_accuracy'],
                                'predictions should be at least as good as majority class.')

        class_probabilities = gama.predict_proba(arff_file_path=test_path)

        print(list(zip(class_predictions, y_test, class_probabilities)))
        self.assertTrue(isinstance(class_probabilities, np.ndarray), 'probability predictions should be numpy arrays.')
        self.assertEqual(class_probabilities.shape, (data['test_size'], data['n_classes']),
                         'predict_proba should return (N,K) shaped array.')

        logloss = log_loss(y_test, class_probabilities)
        print(data['name'], metric, 'log-loss:', logloss)
        self.assertLessEqual(logloss, data['base_log_loss'],
                             'predictions should be at least as good as majority class.')

    def test_binary_classification_accuracy(self):
        """ GamaClassifier can do binary classification with predict metric. """
        self._test_dataset_problem(breast_cancer, 'accuracy')

    def test_binary_classification_logloss(self):
        """ GamaClassifier can do binary classification with predict-proba metric. """
        self._test_dataset_problem(breast_cancer, 'log_loss')

    def test_multiclass_classification_accuracy(self):
        """ GamaClassifier can do multi-class with predict metric. """
        self._test_dataset_problem(wine, 'accuracy')

    def test_multiclass_classification_logloss(self):
        """ GamaClassifier can do multi-class with predict-proba metric. """
        self._test_dataset_problem(wine, 'log_loss')

    #def test_string_label_classification_accuracy(self):
    #    """ GamaClassifier can work with string-like target labels when using predict-metric. """
    #    self._test_dataset_problem(breast_cancer, 'accuracy', labelled_y=True)

    #def test_string_label_classification_log_loss(self):
    #    """ GamaClassifier can work with string-type target labels when using predict-proba metric. """
    #    self._test_dataset_problem(breast_cancer, 'log_loss', labelled_y=True)

    # def test_missing_value_classification(self):
    #     """ GamaClassifier handles missing data. """
    #     data = breast_cancer
    #     metric = 'log_loss'
    #
    #     X, y = data['load'](return_X_y=True)
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
    #     X_train[1:300:2, 0] = X_train[2:300:5, 1] = float("NaN")
    #     X_test[1:100:2, 0] = X_test[2:100:5, 1] = float("NaN")
    #
    #     gama = GamaClassifier(random_state=0, max_total_time=60, objectives=(metric, 'size'))
    #     with Stopwatch() as sw:
    #         gama.fit(X_train, y_train, auto_ensemble_n=5)
    #
    #     self.assertLessEqual(sw.elapsed_time, 60 * self._fit_time_margin, 'fit must stay within 110% of allotted time.')
    #
    #     class_predictions = gama.predict(X_test)
    #     self.assertTrue(isinstance(class_predictions, np.ndarray), 'predictions should be numpy arrays.')
    #     self.assertEqual(class_predictions.shape, (data['test_size'],), 'predict should return (N,) shaped array.')
    #
    #     # Majority classifier on this split achieves 0.6293706293706294
    #     accuracy = accuracy_score(y_test, class_predictions)
    #     print(data['name'], metric, 'accuracy:', accuracy)
    #     self.assertGreaterEqual(accuracy, data['base_accuracy'],
    #                             'predictions should be at least as good as majority class.')
    #
    #     class_probabilities = gama.predict_proba(X_test)
    #     self.assertTrue(isinstance(class_probabilities, np.ndarray), 'probability predictions should be numpy arrays.')
    #     self.assertEqual(class_probabilities.shape, (data['test_size'], data['n_classes']),
    #                      'predict_proba should return (N,K) shaped array.')
    #
    #     # Majority classifier on this split achieves 12.80138131184662
    #     logloss = log_loss(y_test, class_probabilities)
    #     print(data['name'], metric, 'log-loss:', logloss)
    #     self.assertLessEqual(logloss, data['base_log_loss'],
    #                          'predictions should be at least as good as majority class.')


if __name__ == '__main__':
    unittest.main()