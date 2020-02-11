import pytest
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import OneHotEncoder

from gama.utilities.evaluation_library import Evaluation
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
    predictions, scores, estimators = cross_val_predict_score(estimator, x, y, metrics=metrics)
    accuracy, logloss = scores

    assert accuracy_score(y_ohe, predictions) == pytest.approx(accuracy)
    assert -1 * log_loss(y_ohe, predictions) == pytest.approx(logloss)
    assert len(set(estimators)) == len(estimators)


def test_evaluate_individual(BernoulliNBStandardScaler):
    import datetime
    reported_start_time = datetime.datetime.now()

    def fake_evaluate_pipeline(pipeline, *args, **kwargs):
        # predictions, scores, estimators, errors
        return None, (1., ), [], None

    evaluation = evaluate_individual(
        BernoulliNBStandardScaler, evaluate_pipeline=fake_evaluate_pipeline, add_length_to_score=True
    )
    individual = evaluation.individual
    assert individual == BernoulliNBStandardScaler
    assert hasattr(individual, 'fitness')
    assert individual.fitness.values == (1., -2)
    assert (individual.fitness.start_time - reported_start_time).total_seconds() < 1.0


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

    prediction, scores, estimators, errors = evaluate_pipeline(
        BernoulliNBStandardScaler.pipeline, x, y, timeout=60, metrics=scoring_to_metric('accuracy')
    )
    assert 1 == len(scores)
    assert errors is None
    assert 5 == len(estimators)
    assert prediction.shape == (150,)
