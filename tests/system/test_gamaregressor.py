""" Contains full system tests for GamaRegressor """
import unittest

import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from gama.utilities.generic.stopwatch import Stopwatch
from gama import GamaRegressor


def gamaregressor_test_suite():
    test_cases = [GamaRegressorSystemTestCase]
    return unittest.TestSuite(map(unittest.TestLoader().loadTestsFromTestCase, test_cases))


# While we could derive statistics dynamically, we want to know if any changes ever happen, so we save them statically.
boston = dict(
    name='boston',
    load=load_boston,
    test_size=127,
    base_mse=81.790
)


class GamaRegressorSystemTestCase(unittest.TestCase):
    """ Contain complete system tests for Gama. """

    def setUp(self):
        self._fit_time_margin = 1.1

    def tearDown(self):
        pass

    def _test_dataset_problem(self, data, metric):
        X, y = data['load'](return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        gama = GamaRegressor(random_state=0, max_total_time=60, scoring=metric)
        with Stopwatch() as sw:
            gama.fit(X_train, y_train, auto_ensemble_n=5)

        self.assertLessEqual(sw.elapsed_time, 60 * self._fit_time_margin, 'fit must stay within 110% of allotted time.')

        predictions = gama.predict(X_test)
        self.assertTrue(isinstance(predictions, np.ndarray), 'predictions should be numpy arrays.')
        self.assertEqual(predictions.shape, (data['test_size'],), 'predict should return (N,) shaped array.')

        # Majority classifier on this split achieves 0.6293706293706294
        mse = mean_squared_error(y_test, predictions)
        print(data['name'], metric, 'mse:', mse)
        self.assertLessEqual(mse, data['base_mse'],
                                'predictions should be at least as good as predicting mean.')

    def test_regression_mean_squared_error(self):
        """ GamaRegressor works on all-numeric data. """
        self._test_dataset_problem(boston, 'mean_squared_error')

    def test_missing_value_regression(self):
        """ GamaRegressor works when missing values are present. """
        data = boston
        metric = 'mean_squared_error'
        X, y = data['load'](return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        X_train[1:300:2, 0] = X_train[2:300:5, 1] = float("NaN")
        X_test[1:100:2, 0] = X_test[2:100:5, 1] = float("NaN")

        gama = GamaRegressor(random_state=0, max_total_time=60, scoring=metric)
        with Stopwatch() as sw:
            gama.fit(X_train, y_train, auto_ensemble_n=5)

        self.assertLessEqual(sw.elapsed_time, 60 * self._fit_time_margin, 'fit must stay within 110% of allotted time.')

        predictions = gama.predict(X_test)
        self.assertTrue(isinstance(predictions, np.ndarray), 'predictions should be numpy arrays.')
        self.assertEqual(predictions.shape, (data['test_size'],), 'predict should return (N,) shaped array.')

        # Majority classifier on this split achieves 0.6293706293706294
        mse = mean_squared_error(y_test, predictions)
        print(data['name'], metric, 'mse:', mse)
        self.assertLessEqual(mse, data['base_mse'],
                                'predictions should be at least as good as predicting mean.')


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
    unittest.TextTestRunner(verbosity=2).run(gamaregressor_test_suite)
