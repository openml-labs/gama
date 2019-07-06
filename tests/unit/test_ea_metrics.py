import pytest
import numpy as np

from gama.genetic_programming.algorithms.metrics import Metric, all_metrics


def test_accuracy_numeric():
    accuracy_metric = Metric.from_string('accuracy')
    y_true = np.asarray([1, 0, 0, 0, 1])
    y_1_mistake = np.asarray([1, 1, 0, 0, 1])

    assert 1.0 == accuracy_metric.score(y_true, y_true)
    assert 1.0 == accuracy_metric.maximizable_score(y_true, y_true)

    assert 0.8 == accuracy_metric.score(y_true, y_1_mistake)
    assert 0.8 == accuracy_metric.maximizable_score(y_true, y_1_mistake)


def test_accuracy_string():
    accuracy_metric = Metric.from_string('accuracy')
    y_true_str = np.asarray([str(x) for x in [1, 0, 0, 0, 1]])
    y_1_mistake_str = np.asarray([str(x) for x in [1, 1, 0, 0, 1]])

    assert 1.0 == accuracy_metric.score(y_true_str, y_true_str)
    assert 1.0 == accuracy_metric.maximizable_score(y_true_str, y_true_str)

    assert 0.8 == accuracy_metric.score(y_true_str, y_1_mistake_str)
    assert 0.8 == accuracy_metric.maximizable_score(y_true_str, y_1_mistake_str)


def test_logloss_numeric():
    accuracy_metric = Metric.from_string('log_loss')
    y_true_ohe = np.asarray([[0, 1], [1, 0], [1, 0], [1, 0], [0, 1]])
    y_1_mistake_ohe = np.asarray([[0, 1], [0, 1], [1, 0], [1, 0], [0, 1]])
    y_probabilities = np.asarray([[0.1, 0.9], [0.2, 0.8], [0.7, 0.3], [0.95, 0.05], [0.1, 0.9]])

    assert 0 == pytest.approx(accuracy_metric.score(y_true_ohe, y_true_ohe))
    assert 0 == pytest.approx(accuracy_metric.maximizable_score(y_true_ohe, y_true_ohe))

    one_mistake_logloss = 6.907755278982137
    assert  one_mistake_logloss == pytest.approx(accuracy_metric.score(y_true_ohe, y_1_mistake_ohe))
    assert -one_mistake_logloss == pytest.approx(accuracy_metric.maximizable_score(y_true_ohe, y_1_mistake_ohe))

    probabilities_logloss = 0.44562543641520713
    assert probabilities_logloss == accuracy_metric.score(y_true_ohe, y_probabilities)
    assert -probabilities_logloss == accuracy_metric.maximizable_score(y_true_ohe, y_probabilities)


def test_all_metrics_instantiate():
    for metric in all_metrics:
        Metric.from_string(metric)
