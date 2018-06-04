from collections import namedtuple
from functools import partial

import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder


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
Metric = namedtuple("Metric", ["name", "fn", "requires_1d"])
Metric.__new__.__defaults__ = (False,)
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
    metrics[name] = Metric(name, fn, needs_1d)


def string_to_metric(scoring):
    if isinstance(scoring, str) and scoring not in metric_strings:
        raise ValueError('scoring argument', scoring, 'is invalid. It can be one of', list(metric_strings))
    return metrics[scoring]


def cross_val_predict_score(estimator, X, y=None, groups=None, scoring=None, cv=None, n_jobs=1, verbose=0,
                            fit_params=None, pre_dispatch='2*n_jobs'): # , method=None
    metric = string_to_metric(scoring)
    method = 'predict_proba' if hasattr(estimator, 'predict_proba') else 'predict'
    predictions = cross_val_predict(estimator, X, y, groups, cv, n_jobs, verbose, fit_params, pre_dispatch, method)
    if predictions.ndim == 1:
        predictions = OneHotEncoder(n_values=len(set(y))).fit_transform(predictions.reshape(-1, 1)).todense()

    if metric.requires_1d:
        formatted_predictions = np.argmax(predictions, axis=1)
    else:
        formatted_predictions = predictions
        y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).todense()

    score = metric.fn(y, formatted_predictions)
    return predictions, score
