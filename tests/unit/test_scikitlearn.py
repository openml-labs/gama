import time

import pytest
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import OneHotEncoder

from gama.utilities.metrics import scoring_to_metric
from gama.genetic_programming.compilers.scikitlearn import \
    cross_val_predict_score, evaluate_individual, compile_individual, evaluate_pipeline
from tests.unit.unit_fixtures import BernoulliNBStandardScaler, pset


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


def test_evaluate_individual(BernoulliNBStandardScaler, mocker):
    import datetime
    reported_start_time = datetime.datetime.now()
    def fake_evaluate(*args, **kwargs):
        # (scores), start, walltime, processtime
        return (1.0, -0.5), reported_start_time, 0.5, 0.7

    mocker.patch('gama.genetic_programming.compilers.scikitlearn.evaluate_pipeline', new=fake_evaluate)
    individual = evaluate_individual(BernoulliNBStandardScaler, evaluate_pipeline_length=True)
    assert individual == BernoulliNBStandardScaler
    assert hasattr(individual, 'fitness')
    assert individual.fitness.values == (1.0, -0.5, -2)
    assert individual.fitness.start_time == reported_start_time
    assert individual.fitness.wallclock_time == 0.5
    assert individual.fitness.process_time == 0.7


def test_compile_individual(BernoulliNBStandardScaler):
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    pipeline = compile_individual(BernoulliNBStandardScaler)
    assert 2 == len(pipeline.steps)
    assert isinstance(pipeline.steps[0][1], StandardScaler)
    assert isinstance(pipeline.steps[1][1], BernoulliNB)

    extended_pipeline = compile_individual(BernoulliNBStandardScaler, preprocessing_steps=[MinMaxScaler()])
    assert 3 == len(extended_pipeline.steps)
    assert isinstance(extended_pipeline.steps[0][1], MinMaxScaler)
    assert isinstance(extended_pipeline.steps[1][1], StandardScaler)
    assert isinstance(extended_pipeline.steps[2][1], BernoulliNB)


def test_evaluate_pipeline(BernoulliNBStandardScaler):
    x, y = load_iris(return_X_y=True)
    x, y = pd.DataFrame(x), pd.Series(y)

    scores, start, wallclock, process = evaluate_pipeline(
        BernoulliNBStandardScaler, x, y, timeout=60, deadline=time.time()+60,
        metrics=scoring_to_metric('accuracy'))
    assert 1 == len(scores)
