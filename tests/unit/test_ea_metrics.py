import itertools
import pytest
import numpy as np
import pandas as pd

from gama.utilities.metrics import Metric, all_metrics, scoring_to_metric


def _test_metric(metric, y_true, y_pred, max_score, prediction_score):
    """ Metric is calculated directly with different input formats. """

    def as_1d_array(list_):
        return np.asarray(list_).reshape(-1, 1)

    def is_nd(arr):
        return arr.ndim == 2 and arr.shape[1] > 1

    formats = [np.asarray, as_1d_array, pd.Series, pd.DataFrame]
    for y_format, pred_format in itertools.product(formats, formats):
        if (is_nd(y_true) and y_format in [pd.Series, as_1d_array]) or (
            is_nd(y_pred) and pred_format in [pd.Series, as_1d_array]
        ):
            continue  # If a (N, K)-array is provided, we don't squash it to (N,1)

        truth = y_format(y_true)
        prediction = pred_format(y_pred)
        assert max_score == pytest.approx(metric.maximizable_score(truth, truth))
        score = pytest.approx(metric.maximizable_score(truth, prediction))
        assert prediction_score == score


def test_accuracy_numeric():
    accuracy_metric = Metric("accuracy")
    y_true = np.asarray([1, 0, 0, 0, 1])
    y_1_mistake = np.asarray([1, 1, 0, 0, 1])
    _test_metric(
        accuracy_metric, y_true, y_1_mistake, max_score=1.0, prediction_score=0.8
    )


def test_accuracy_string():
    accuracy_metric = Metric("accuracy")
    y_true_str = np.asarray([str(x) for x in [1, 0, 0, 0, 1]])
    y_1_mistake_str = np.asarray([str(x) for x in [1, 1, 0, 0, 1]])
    _test_metric(
        accuracy_metric,
        y_true_str,
        y_1_mistake_str,
        max_score=1.0,
        prediction_score=0.8,
    )


def test_logloss_numeric():
    log_loss_metric = Metric("neg_log_loss")
    y_true = np.asarray([1, 0, 0, 0, 1])
    y_1_mistake_ohe = np.asarray([[0, 1], [0, 1], [1, 0], [1, 0], [0, 1]])
    one_mistake_logloss = -6.907755278982137
    _test_metric(
        log_loss_metric,
        y_true,
        y_1_mistake_ohe,
        max_score=0,
        prediction_score=one_mistake_logloss,
    )

    y_true_ohe = np.asarray([[0, 1], [1, 0], [1, 0], [1, 0], [0, 1]])
    y_probabilities = np.asarray(
        [[0.1, 0.9], [0.2, 0.8], [0.7, 0.3], [0.95, 0.05], [0.1, 0.9]]
    )
    probabilities_logloss = -0.44562543641520713
    _test_metric(
        log_loss_metric,
        y_true_ohe,
        y_probabilities,
        max_score=0,
        prediction_score=probabilities_logloss,
    )


def test_all_metrics_instantiate():
    for metric in all_metrics:
        Metric(metric)


def test_scoring_to_metric_mixed():
    metrics = list(all_metrics)
    mixed_metrics = [Metric(metric) for metric in metrics[:2]] + metrics[2:]
    scoring_to_metric(mixed_metrics)
