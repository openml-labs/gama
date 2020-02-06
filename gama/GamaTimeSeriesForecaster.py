import pandas as pd
import numpy as np
import time
from functools import partial
from .GamaRegressor import GamaRegressor
from .genetic_programming.compilers.scikitlearn import evaluate_individual, cross_val_predict_timeseries

GKEY_TYPE = "https://metadata.datadrivendiscovery.org/types/GroupingKey"
SGKEY_TYPE = "https://metadata.datadrivendiscovery.org/types/SuggestedGroupingKey"

def timeseries_cv(indices, numfolds):
    n = len(indices)
    foldsize = n / (numfolds + 1)
    for i in range(numfolds):
        train = np.asarray(indices[0:int(foldsize * (i+1))], dtype=int)
        test = np.asarray(indices[int(foldsize * (i+1)):int(foldsize * (i+2))], dtype=int)
        # train = np.arange(0, int(foldsize * (i+1)), dtype=int)
        # test = np.arange(int(foldsize * (i+1)), int(foldsize * (i+2)), dtype=int)
        yield train, test


def time_series_groups(X):
    grouping_keys = X.metadata.get_columns_with_semantic_type(GKEY_TYPE)
    suggested_grouping_keys = X.metadata.get_columns_with_semantic_type(SGKEY_TYPE)
    if len(grouping_keys) == 0:
        grouping_keys = suggested_grouping_keys
    # No grouping keys.  All the data represents one time series.
    if len(grouping_keys) == 0:
        yield list(X.index)
        return
    assert len(grouping_keys) == 1
    colname = X.columns[grouping_keys[0]]
    for val, df in X.groupby(colname):
        yield list(df.index)


def timeseries_cv_grouped(X, numfolds):
    for group in time_series_groups(X):
        for train, test in timeseries_cv(group, numfolds):
            yield train, test
        break

class GamaTimeSeriesForecaster(GamaRegressor):
    """ Wrapper for the toolbox logic executing the AutoML pipeline for time
    series forecasting. """

    # We require a special cv, which ensures that training and test sets are contiguous in time
    def _set_evaluator(self, timeout: int = 1e6):
        deadline = time.time() + timeout
        evaluate_args = dict(evaluate_pipeline_length=self._regularize_length, X=self._X, y_train=self._y,
                             metrics=self._metrics, cache_dir=self._cache_dir, timeout=self._max_eval_time,
                             deadline=deadline, cv=[split for split in timeseries_cv_grouped(self._X, 5)],
                             cvpredict=cross_val_predict_timeseries)
        self._operator_set.evaluate = partial(evaluate_individual, **evaluate_args)

