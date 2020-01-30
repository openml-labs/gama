import pandas as pd
import numpy as np
import time
from functools import partial
from .GamaRegressor import GamaRegressor
from .genetic_programming.compilers.scikitlearn import evaluate_individual, cross_val_predict_timeseries


def timeseries_cv(X, numfolds):
    n = X.shape[0]
    foldsize = n / (numfolds + 1)
    for i in range(numfolds):
        train = np.arange(0, int(foldsize * (i+1)), dtype=int)
        test = np.arange(int(foldsize * (i+1)), int(foldsize * (i+2)), dtype=int)
        yield train, test


class GamaTimeSeriesForecaster(GamaRegressor):
    """ Wrapper for the toolbox logic executing the AutoML pipeline for time
    series forecasting. """

    # We require a special cv, which ensures that training and test sets are contiguous in time
    def _set_evaluator(self, timeout: int = 1e6):
        deadline = time.time() + timeout
        evaluate_args = dict(evaluate_pipeline_length=self._regularize_length, X=self._X, y_train=self._y,
                             metrics=self._metrics, cache_dir=self._cache_dir, timeout=self._max_eval_time,
                             deadline=deadline, cv=[split for split in timeseries_cv(self._X, 5)],
                             cvpredict=cross_val_predict_timeseries)
        self._operator_set.evaluate = partial(evaluate_individual, **evaluate_args)

