import pandas as pd
from sklearn.datasets import load_iris
from gama.genetic_programming.compilers.scikitlearn import (
    evaluate_individual,
    compile_individual,
    evaluate_pipeline,
)
from gama.utilities.metrics import Metric, scoring_to_metric


def test_evaluate_individual(SS_BNB):
    import datetime

    reported_start_time = datetime.datetime.now()

    def fake_evaluate_pipeline(pipeline, *args, **kwargs):
        # predictions, scores, estimators, errors
        return None, (1.0,), [], None

    evaluation = evaluate_individual(
        SS_BNB, evaluate_pipeline=fake_evaluate_pipeline, add_length_to_score=True,
    )
    individual = evaluation.individual
    assert individual == SS_BNB
    assert hasattr(individual, "fitness")
    assert individual.fitness.values == (1.0, -2)
    assert (individual.fitness.start_time - reported_start_time).total_seconds() < 1.0


def test_compile_individual(SS_BNB):
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    pipeline = compile_individual(SS_BNB)
    assert 2 == len(pipeline.steps)
    assert isinstance(pipeline.steps[0][1], StandardScaler)
    assert isinstance(pipeline.steps[1][1], BernoulliNB)

    mm_scale = [("scaler", MinMaxScaler())]
    extended_pipeline = compile_individual(SS_BNB, preprocessing_steps=mm_scale)
    assert 3 == len(extended_pipeline.steps)
    assert isinstance(extended_pipeline.steps[0][1], MinMaxScaler)
    assert isinstance(extended_pipeline.steps[1][1], StandardScaler)
    assert isinstance(extended_pipeline.steps[2][1], BernoulliNB)


def test_evaluate_pipeline(SS_BNB):
    x, y = load_iris(return_X_y=True)
    x, y = pd.DataFrame(x), pd.Series(y)

    prediction, scores, estimators, errors = evaluate_pipeline(
        SS_BNB.pipeline, x, y, timeout=60, metrics=scoring_to_metric("accuracy"),
    )
    assert 1 == len(scores)
    assert errors is None
    assert 5 == len(estimators)
    assert prediction.shape == (150,)


def test_evaluate_invalid_pipeline(InvalidLinearSVC):
    x, y = load_iris(return_X_y=True)
    x, y = pd.DataFrame(x), pd.Series(y)

    prediction, scores, estimators, error = evaluate_pipeline(
        InvalidLinearSVC.pipeline,
        x,
        y,
        timeout=60,
        metrics=scoring_to_metric("accuracy"),
    )
    assert (float("-inf"),) == scores
    assert str(error).startswith("Unsupported set of arguments:")
    assert str(error).endswith("penalty='l1', loss='squared_hinge', dual=True")
    assert estimators is None
    assert prediction is None
