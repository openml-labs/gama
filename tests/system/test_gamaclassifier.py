""" Contains full system tests for GamaClassifier """
import numpy as np
import pandas as pd
import pytest
from typing import Type

from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline

from gama.postprocessing import EnsemblePostProcessing
from gama.search_methods import AsynchronousSuccessiveHalving, AsyncEA, RandomSearch
from gama.search_methods.base_search import BaseSearch
from gama.utilities.generic.stopwatch import Stopwatch
from gama import GamaClassifier

import warnings

warnings.filterwarnings("error")

FIT_TIME_MARGIN = 1.1

# While we could derive statistics dynamically,
# we want to know if any changes ever happen, so we save them statically.
breast_cancer = dict(
    name="breast_cancer",
    load=load_breast_cancer,
    test_size=143,
    n_classes=2,
    base_accuracy=0.62937,
    base_log_loss=12.80138,
)

breast_cancer_missing = dict(
    name="breast_cancer_missing",
    load=load_breast_cancer,
    target="status",
    test_size=143,
    n_classes=2,
    base_accuracy=0.62937,
    base_log_loss=12.80138,
)

wine = dict(
    name="wine",
    load=load_wine,
    test_size=45,
    n_classes=3,
    base_accuracy=0.4,
    base_log_loss=20.72326,
)

iris_arff = dict(
    name="iris",
    train="tests/data/iris_train.arff",
    test="tests/data/iris_test.arff",
    test_size=50,
    n_classes=3,
    base_accuracy=0.3333,
    base_log_loss=1.09861,
)

diabetes_arff = dict(
    name="diabetes",
    train="tests/data/diabetes_train.arff",
    test="tests/data/diabetes_test.arff",
    test_size=150,
    n_classes=2,
    base_accuracy=0.65104,
    base_log_loss=0.63705,
)


def _test_dataset_problem(
    data,
    metric: str,
    arff: bool = False,
    y_type: Type = pd.DataFrame,
    search: BaseSearch = AsyncEA(),
    missing_values: bool = False,
    max_time: int = 60,
):
    """

    :param data:
    :param metric:
    :param arff:
    :param y_type: pd.DataFrame, pd.Series, np.ndarray or str
    :return:
    """
    gama = GamaClassifier(
        random_state=0,
        max_total_time=max_time,
        scoring=metric,
        search=search,
        n_jobs=1,
        post_processing=EnsemblePostProcessing(ensemble_size=5),
        store="nothing",
    )
    if arff:
        train_path = f"tests/data/{data['name']}_train.arff"
        test_path = f"tests/data/{data['name']}_test.arff"

        X, y = data["load"](return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=0
        )
        y_test = [str(val) for val in y_test]

        with Stopwatch() as sw:
            gama.fit_from_file(train_path, target_column=data["target"])
        class_predictions = gama.predict_from_file(
            test_path, target_column=data["target"]
        )
        class_probabilities = gama.predict_proba_from_file(
            test_path, target_column=data["target"]
        )
        gama_score = gama.score_from_file(test_path)
    else:
        X, y = data["load"](return_X_y=True)
        if y_type == str:
            databunch = data["load"]()
            y = np.asarray([databunch.target_names[c_i] for c_i in databunch.target])
        if y_type in [pd.Series, pd.DataFrame]:
            y = y_type(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=0
        )
        if missing_values:
            X_train[1:300:2, 0] = X_train[2:300:5, 1] = float("NaN")
            X_test[1:100:2, 0] = X_test[2:100:5, 1] = float("NaN")

        with Stopwatch() as sw:
            gama.fit(X_train, y_train)
        class_predictions = gama.predict(X_test)
        class_probabilities = gama.predict_proba(X_test)
        gama_score = gama.score(X_test, y_test)

    assert (
        60 * FIT_TIME_MARGIN > sw.elapsed_time
    ), "fit must stay within 110% of allotted time."

    assert isinstance(
        class_predictions, np.ndarray
    ), "predictions should be numpy arrays."
    assert (
        data["test_size"],
    ) == class_predictions.shape, "predict should return (N,) shaped array."

    accuracy = accuracy_score(y_test, class_predictions)
    # Majority classifier on this split achieves 0.6293706293706294
    print(data["name"], metric, "accuracy:", accuracy)
    assert (
        data["base_accuracy"] <= accuracy
    ), "predictions should be at least as good as majority class."

    assert isinstance(
        class_probabilities, np.ndarray
    ), "probability predictions should be numpy arrays."
    assert (data["test_size"], data["n_classes"]) == class_probabilities.shape, (
        "predict_proba should return" " (N,K) shaped array."
    )

    # Majority classifier on this split achieves 12.80138131184662
    logloss = log_loss(y_test, class_probabilities)
    print(data["name"], metric, "log-loss:", logloss)
    assert (
        data["base_log_loss"] >= logloss
    ), "predictions should be at least as good as majority class."

    score_to_match = logloss if metric == "neg_log_loss" else accuracy
    assert score_to_match == pytest.approx(gama_score)
    gama.cleanup("all")
    return gama


def test_binary_classification_accuracy():
    """ Binary classification, accuracy, numpy data and ensemble code export """
    gama = _test_dataset_problem(breast_cancer, "accuracy")

    x, y = breast_cancer["load"](return_X_y=True)
    code = gama.export_script(file=None)
    local = {}
    exec(code, {}, local)
    pipeline = local["pipeline"]  # should be defined in exported code
    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.steps[-1][-1], VotingClassifier)
    pipeline.fit(x, y)
    assert 0.9 < pipeline.score(x, y)


def test_binary_classification_accuracy_asha():
    """ Binary classification, accuracy, numpy data, ASHA search. """
    _test_dataset_problem(
        breast_cancer, "accuracy", search=AsynchronousSuccessiveHalving(), max_time=60
    )


def test_binary_classification_accuracy_random_search():
    """ Binary classification, accuracy, numpy data, random search. """
    _test_dataset_problem(breast_cancer, "accuracy", search=RandomSearch())


def test_binary_classification_logloss():
    """ Binary classification, log loss (probabilities), numpy data, ASHA search. """
    _test_dataset_problem(breast_cancer, "neg_log_loss")


def test_multiclass_classification_accuracy():
    """ Multiclass classification, accuracy, numpy data. """
    _test_dataset_problem(wine, "accuracy")


def test_multiclass_classification_logloss():
    """ Multiclass classification, log loss (probabilities), numpy data. """
    _test_dataset_problem(wine, "neg_log_loss")


def test_string_label_classification_accuracy():
    """ Binary classification, accuracy, target is str. """
    _test_dataset_problem(breast_cancer, "accuracy", y_type=str)


def test_string_label_classification_log_loss():
    """ Binary classification, log loss (probabilities), target is str. """
    _test_dataset_problem(breast_cancer, "neg_log_loss", y_type=str)


def test_missing_value_classification_arff():
    """ Binary classification, log loss (probabilities), arff data. """
    _test_dataset_problem(breast_cancer_missing, "neg_log_loss", arff=True)


def test_missing_value_classification():
    """ Binary classification, log loss (probabilities), missing values. """
    _test_dataset_problem(breast_cancer_missing, "neg_log_loss", missing_values=True)
