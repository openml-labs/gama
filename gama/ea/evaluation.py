import os
import pickle
import time
import uuid
from collections import namedtuple
from functools import partial

import numpy as np
import stopit
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

from gama.ea.automl_gp import log
from gama.utilities import log_parseable_event, TOKENS


def evaluate(metric, y, p):
    """ Metrics are difficult. Some need a 1d-array. Some don't allow probabilities. This method
    formats y and p probably, and evaluates the metric.

    :param metric:
    :param y:
    :param p:
    :return:
    """
    formatted_predictions = p
    formatted_y = y

    if metric.is_classification:
        if metric.requires_1d:
            if p.ndim > 1:
                formatted_predictions = np.argmax(p, axis=1)
            if y.ndim > 1:
                formatted_y = np.argmax(y, axis=1)
        else:
            if y.ndim == 1:
                formatted_y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).todense()
            if p.ndim == 1:
                formatted_predictions = OneHotEncoder().fit_transform(p.reshape(-1, 1)).todense()

    score = metric.fn(formatted_y, formatted_predictions)
    return score


def neg(fn):
    def negative_result(*args, **kwargs):
        return -1 * fn(*args, **kwargs)
    return negative_result


# Scikit-learn does not have an option to return predictions and score at the same time. Furthermore, the only string
# interpretation of scoring functions automatically make 'scorers' which train the model internally, also throwing
# away any predictions. So we need to make our own conversion of scoring string to function, predict, score, and return
# both. Construction of metric_strings copied with minor modifications from SCORERS of scikit-learn. See also:
# https://github.com/scikit-learn/scikit-learn/blob/a24c8b464d094d2c468a16ea9f8bf8d42d949f84/sklearn/metrics/scorer.py#L530
# https://stackoverflow.com/questions/41003897/scikit-learn-cross-validates-score-and-predictions-at-one-go
Metric = namedtuple("Metric", ["name", "fn", "requires_1d", "is_classification", "is_regression"])
Metric.__new__.__defaults__ = (False, False, False,)
metric_strings = dict(
    accuracy=metrics.accuracy_score,
    roc_auc=metrics.roc_auc_score,
    explained_variance=metrics.explained_variance_score,
    r2=metrics.r2_score,
    neg_median_absolute_error=neg(metrics.median_absolute_error),
    neg_mean_absolute_error=neg(metrics.mean_absolute_error),
    neg_mean_squared_error=neg(metrics.mean_squared_error),
    neg_mean_squared_log_error=neg(metrics.mean_squared_log_error),
    median_absolute_error=metrics.median_absolute_error,
    mean_squared_error=metrics.mean_squared_error,
    average_precision=metrics.average_precision_score,
    log_loss=metrics.log_loss,
    neg_log_loss=neg(metrics.log_loss)
)

# Below is also based on scikit-learn code:
for name, metric in [('precision', metrics.precision_score),
                     ('recall', metrics.recall_score), ('f1', metrics.f1_score)]:
    metric_strings[name] = metric
    for average in ['macro', 'micro', 'samples', 'weighted']:
        qualified_name = '{0}_{1}'.format(name, average)
        metric_strings[qualified_name] = partial(metric, pos_label=None, average=average)

metrics = {}
for name, fn in metric_strings.items():
    needs_1d = any([s in name for s in ['accuracy', 'precision', 'recall', 'f1']])
    is_regression = ('error' in name) or (name == 'r2')
    is_classification = not is_regression
    metrics[name] = Metric(name, fn, needs_1d, is_classification, is_regression)


def string_to_metric(scoring):
    if isinstance(scoring, str) and scoring not in metric_strings:
        raise ValueError('scoring argument', scoring, 'is invalid. It can be one of', list(metric_strings))
    return metrics[scoring]


def cross_val_predict_score(estimator, X, y=None, groups=None, scoring=None, cv=None, n_jobs=1, verbose=0,
                            fit_params=None, pre_dispatch='2*n_jobs'): # , method=None
    metric = string_to_metric(scoring)
    method = 'predict_proba' if hasattr(estimator, 'predict_proba') else 'predict'
    predictions = cross_val_predict(estimator, X, y, groups, cv, n_jobs, verbose, fit_params, pre_dispatch, method)

    if metric.is_classification:
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = np.squeeze(predictions)
        if predictions.ndim == 1:
            predictions = OneHotEncoder(n_values=len(set(y))).fit_transform(predictions.reshape(-1, 1)).todense()

        if metric.requires_1d:
            formatted_predictions = np.argmax(predictions, axis=1)
        else:
            formatted_predictions = predictions
            y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).todense()
    elif metric.is_regression:
        # single-output regression is always one-dimensional.
        formatted_predictions = predictions

    score = metric.fn(y, formatted_predictions)
    return predictions, score


def object_is_valid_pipeline(o):
    return (o is not None and
            hasattr(o, 'fit') and
            hasattr(o, 'predict') and
            hasattr(o, 'steps'))


def evaluate_pipeline(pl, X, y, timeout, scoring='accuracy', cv=5, cache_dir=None, logger=None):
    """ Evaluates a pipeline used k-Fold CV. """
    if not logger:
        logger = log

    if not object_is_valid_pipeline(pl):
        return ValueError('Pipeline is not valid. Must not be None and have `fit`, `predict` and `steps`.')

    start = time.process_time()
    with stopit.ThreadingTimeout(timeout) as c_mgr:
        try:
            prediction, score = cross_val_predict_score(pl, X, y, cv=cv, scoring=scoring)
        except stopit.TimeoutException:
            raise
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.info('{} encountered while evaluating pipeline.'.format(type(e)))#, exc_info=True)
            single_line_pipeline = ''.join(str(pl).split('\n'))
            log_parseable_event(logger, TOKENS.EVALUATION_ERROR, single_line_pipeline, type(e), e)
            score = -float("inf")

    if cache_dir and score != -float("inf"):
        pl_filename = str(uuid.uuid4())

        try:
            with open(os.path.join(cache_dir, pl_filename + '.pkl'), 'wb') as fh:
                pickle.dump((pl, prediction, score), fh)
        except FileNotFoundError:
            log.warning("File not found while saving predictions. This can happen in the multi-process case if the "
                        "cache gets deleted within `max_eval_time` of the end of the search process.", exc_info=True)

    evaluation_time = time.process_time() - start
    pipeline_length = len(pl.steps)

    if c_mgr.state == c_mgr.INTERRUPTED:
        # A TimeoutException was raised, but not by the context manager.
        # This indicates that the outer context manager (the ea) timed out.
        logger.info("Outer-timeout during evaluation of {}".format(pl))
        raise stopit.utils.TimeoutException()

    if not c_mgr:
        # For now we treat a eval timeout the same way as e.g. NaN exceptions.
        fitness_values = (-float("inf"), timeout, pipeline_length)
        logger.info('Timeout encountered while evaluating pipeline.')#, exc_info=True)
        single_line_pipeline = ''.join(str(pl).split('\n'))
        log_parseable_event(logger, TOKENS.EVALUATION_TIMEOUT, single_line_pipeline)
        logger.debug("Timeout after {}s: {}".format(timeout, pl))
    else:
        fitness_values = (score, evaluation_time, pipeline_length)

    return fitness_values
