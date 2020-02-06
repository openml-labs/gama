
import time
from functools import partial
from .GamaClassifier import GamaClassifier
from .genetic_programming.compilers.scikitlearn import evaluate_individual, cross_val_predict_timeseries
from .utilities.timeseries import timeseries_cv_grouped


class GamaTimeSeriesClassifier(GamaClassifier):
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

