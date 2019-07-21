import itertools
import pytest
import numpy as np
import pandas as pd

from gama.genetic_programming.algorithms.metrics import Metric, all_metrics, scoring_to_metric


def _test_metric(metric, y_true, y_pred, optimal_score, prediction_score):
    def N_1_array(list_):
        return np.asarray(list_).reshape(-1, 1)

    accepted_formats = [np.asarray, N_1_array, pd.Series, pd.DataFrame]
    for truth_format, prediction_format in itertools.product(accepted_formats, accepted_formats):
        if ((y_true.ndim == 2 and y_true.shape[1] > 1 and truth_format in [pd.Series, N_1_array])
                or (y_pred.ndim == 2 and y_pred.shape[1] > 1 and prediction_format in [pd.Series, N_1_array])):
            # If a (N, K)-array is provided, we don't squash it to (N,1)
            continue

        truth = truth_format(y_true)
        prediction = prediction_format(y_pred)
        assert optimal_score == pytest.approx(metric.score(truth, truth))
        assert prediction_score == pytest.approx(metric.score(truth, prediction))


def test_accuracy_numeric():
    accuracy_metric = Metric.from_string('accuracy')
    y_true = np.asarray([1, 0, 0, 0, 1])
    y_1_mistake = np.asarray([1, 1, 0, 0, 1])
    _test_metric(accuracy_metric, y_true, y_1_mistake, optimal_score=1.0, prediction_score=0.8)


def test_accuracy_string():
    accuracy_metric = Metric.from_string('accuracy')
    y_true_str = np.asarray([str(x) for x in [1, 0, 0, 0, 1]])
    y_1_mistake_str = np.asarray([str(x) for x in [1, 1, 0, 0, 1]])
    _test_metric(accuracy_metric, y_true_str, y_1_mistake_str, optimal_score=1.0, prediction_score=0.8)


def test_logloss_numeric():
    log_loss_metric = Metric.from_string('log_loss')
    y_true = np.asarray([1, 0, 0, 0, 1])
    y_1_mistake_ohe = np.asarray([[0, 1], [0, 1], [1, 0], [1, 0], [0, 1]])
    one_mistake_logloss = 6.907755278982137
    _test_metric(log_loss_metric, y_true, y_1_mistake_ohe, optimal_score=0, prediction_score=one_mistake_logloss)

    y_true_ohe = np.asarray([[0, 1], [1, 0], [1, 0], [1, 0], [0, 1]])
    y_probabilities = np.asarray([[0.1, 0.9], [0.2, 0.8], [0.7, 0.3], [0.95, 0.05], [0.1, 0.9]])
    probabilities_logloss = 0.44562543641520713
    _test_metric(log_loss_metric, y_true_ohe, y_probabilities, optimal_score=0, prediction_score=probabilities_logloss)


def test_all_metrics_instantiate():
    for metric in all_metrics:
        Metric.from_string(metric)


def test_scoring_to_metric_mixed():
    metrics = list(all_metrics)
    mixed_metrics = [Metric.from_string(metric) for metric in metrics[:2]] + metrics[2:]
    scoring_to_metric(mixed_metrics)
