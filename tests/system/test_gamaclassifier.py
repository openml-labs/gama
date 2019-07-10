""" Contains full system tests for GamaClassifier """
import numpy as np
import pandas as pd
import pytest
from typing import Type

from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

from gama.utilities.generic.stopwatch import Stopwatch
from gama import GamaClassifier

import warnings
warnings.filterwarnings("error")

FIT_TIME_MARGIN = 1.1

# While we could derive statistics dynamically, we want to know if any changes ever happen, so we save them statically.
breast_cancer = dict(
    name='breast_cancer',
    load=load_breast_cancer,
    test_size=143,
    n_classes=2,
    base_accuracy=0.62937,
    base_log_loss=12.80138
)

breast_cancer_missing = dict(
    name='breast_cancer_missing',
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
    train='tests/data/iris_train.arff',
    test='tests/data/iris_test.arff',
    test_size=50,
    n_classes=3,
    base_accuracy=0.3333,
    base_log_loss=1.09861
)

diabetes_arff = dict(
    name='diabetes',
    train='tests/data/diabetes_train.arff',
    test='tests/data/diabetes_test.arff',
    test_size=150,
    n_classes=2,
    base_accuracy=0.65104,
    base_log_loss=0.63705
)


def _test_dataset_problem(data, metric: str, arff: bool=False, y_type: Type=pd.DataFrame):
    """

    :param data:
    :param metric:
    :param arff:
    :param y_type: pd.DataFrame, pd.Series, np.ndarray or str
    :return:
    """
    gama = GamaClassifier(random_state=0, max_total_time=60, scoring=metric)
    if arff:
        train_path = 'tests/data/{}_train.arff'.format(data['name'])
        test_path = 'tests/data/{}_test.arff'.format(data['name'])

        X, y = data['load'](return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
        y_test = [str(val) for val in y_test]

        with Stopwatch() as sw:
            gama.fit_arff(train_path, auto_ensemble_n=5)
        class_predictions = gama.predict_arff(test_path)
        class_probabilities = gama.predict_proba_arff(test_path)
    else:
        X, y = data['load'](return_X_y=True)
        if y_type == str:
            databunch = data['load']()
            y = np.asarray([databunch.target_names[c_i] for c_i in databunch.target])
        if y_type in [pd.Series, pd.DataFrame]:
            y = y_type(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

        with Stopwatch() as sw:
            gama.fit(X_train, y_train, auto_ensemble_n=5)
        class_predictions = gama.predict(X_test)
        class_probabilities = gama.predict_proba(X_test)

    assert 60 * FIT_TIME_MARGIN > sw.elapsed_time, 'fit must stay within 110% of allotted time.'

    assert isinstance(class_predictions, np.ndarray), 'predictions should be numpy arrays.'
    assert (data['test_size'],) == class_predictions.shape, 'predict should return (N,) shaped array.'

    accuracy = accuracy_score(y_test, class_predictions)
    # Majority classifier on this split achieves 0.6293706293706294
    print(data['name'], metric, 'accuracy:', accuracy)
    assert data['base_accuracy'] <= accuracy, 'predictions should be at least as good as majority class.'

    assert isinstance(class_probabilities, np.ndarray), 'probability predictions should be numpy arrays.'
    assert (data['test_size'], data['n_classes']) == class_probabilities.shape, ('predict_proba should return'
                                                                                 ' (N,K) shaped array.')

    # Majority classifier on this split achieves 12.80138131184662
    logloss = log_loss(y_test, class_probabilities)
    print(data['name'], metric, 'log-loss:', logloss)
    assert data['base_log_loss'] >= logloss, 'predictions should be at least as good as majority class.'

    score_to_match = logloss if metric == 'log_loss' else accuracy
    assert score_to_match == pytest.approx(gama.score(X_test, y_test))


def test_binary_classification_accuracy():
    """ GamaClassifier can do binary classification with predict metric from numpy data. """
    _test_dataset_problem(breast_cancer, 'accuracy')


def test_binary_classification_logloss():
    """ GamaClassifier can do binary classification with predict-proba metric from numpy data. """
    _test_dataset_problem(breast_cancer, 'log_loss')


def test_multiclass_classification_accuracy():
    """ GamaClassifier can do multi-class with predict metric from numpy data. """
    _test_dataset_problem(wine, 'accuracy')


def test_multiclass_classification_logloss():
    """ GamaClassifier can do multi-class with predict-proba metric from numpy data. """
    _test_dataset_problem(wine, 'log_loss')


def test_string_label_classification_accuracy():
    """ GamaClassifier can work with string-like target labels when using predict-metric from numpy data. """
    _test_dataset_problem(breast_cancer, 'accuracy', y_type=str)


def test_string_label_classification_log_loss():
    """ GamaClassifier can work with string-type target labels when using predict-proba metric from numpy data. """
    _test_dataset_problem(breast_cancer, 'log_loss', y_type=str)


def test_binary_classification_accuracy_arff():
    """ GamaClassifier can do binary classification with predict metric. """
    _test_dataset_problem(breast_cancer, 'accuracy', arff=True)


def test_binary_classification_logloss_arff():
    """ GamaClassifier can do binary classification with predict-proba metric. """
    _test_dataset_problem(breast_cancer, 'log_loss', arff=True)


def test_multiclass_classification_accuracy_arff():
    """ GamaClassifier can do multi-class with predict metric. """
    _test_dataset_problem(wine, 'accuracy', arff=True)


def test_multiclass_classification_logloss_arff():
    """ GamaClassifier can do multi-class with predict-proba metric. """
    _test_dataset_problem(wine, 'log_loss', arff=True)


def test_missing_value_classification_arff():
    """ GamaClassifier handles missing data. """
    _test_dataset_problem(breast_cancer_missing, 'log_loss', arff=True)


def test_missing_value_classification():
    """ GamaClassifier handles missing data from numpy data. """
    data = breast_cancer
    metric = 'log_loss'

    X, y = data['load'](return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
    X_train[1:300:2, 0] = X_train[2:300:5, 1] = float("NaN")
    X_test[1:100:2, 0] = X_test[2:100:5, 1] = float("NaN")

    gama = GamaClassifier(random_state=0, max_total_time=60, scoring=metric)
    with Stopwatch() as sw:
        gama.fit(X_train, y_train, auto_ensemble_n=5)

    assert 60 * FIT_TIME_MARGIN >= sw.elapsed_time, 'fit must stay within 110% of allotted time.'

    class_predictions = gama.predict(X_test)
    assert isinstance(class_predictions, np.ndarray), 'predictions should be numpy arrays.'
    assert (data['test_size'],) == class_predictions.shape, 'predict should return (N,) shaped array.'

    # Majority classifier on this split achieves 0.6293706293706294
    accuracy = accuracy_score(y_test, class_predictions)
    print(data['name'], metric, 'accuracy:', accuracy)
    assert data['base_accuracy'] <= accuracy, 'predictions should be at least as good as majority class.'

    class_probabilities = gama.predict_proba(X_test)
    assert isinstance(class_probabilities, np.ndarray), 'probability predictions should be numpy arrays.'
    assert (data['test_size'], data['n_classes']) == class_probabilities.shape, ('predict_proba should return'
                                                                                 ' (N,K) shaped array.')

    # Majority classifier on this split achieves 12.80138131184662
    logloss = log_loss(y_test, class_probabilities)
    print(data['name'], metric, 'log-loss:', logloss)
    assert data['base_log_loss'] >= logloss, 'predictions should be at least as good as majority class.'
