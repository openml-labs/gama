import pytest
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import OneHotEncoder

from gama.genetic_programming.algorithms.metrics import scoring_to_metric
from gama.genetic_programming.compilers.scikitlearn import cross_val_predict_score


def test_cross_val_predict_score():
    estimator = DecisionTreeClassifier()
    x, y = load_iris(return_X_y=True)
    y_ohe = OneHotEncoder().fit_transform(y.reshape(-1, 1))
    x, y = pd.DataFrame(x), pd.Series(y)

    metrics = scoring_to_metric(['accuracy', 'log_loss'])
    predictions, scores = cross_val_predict_score(estimator, x, y, metrics=metrics)
    accuracy, logloss = scores

    assert accuracy_score(y_ohe, predictions) == pytest.approx(accuracy)
    assert -1 * log_loss(y_ohe, predictions) == pytest.approx(logloss)



