from typing import Optional, Tuple, List, Union
import numpy as np
import pandas as pd

from gama.genetic_programming.components import Individual
from gama.utilities.evaluation_library import Evaluation, EvaluationLibrary
from .unit_fixtures import pset, GaussianNB, RandomForestPipeline, LinearSVC


def _mock_evaluation(
        individual: Individual,
        predictions: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = np.zeros(30,),
        score: Optional[Tuple[float, ...]] = None,
        estimators: List[object] = None,
        start_time: int = 0,
        duration: int = 0,
        error: str = None
) -> Evaluation:
    return Evaluation(
        individual,
        predictions,
        score if score is not None else tuple(np.random.random(size=(3,))),
        estimators,
        start_time,
        duration,
        error
    )


def test_evaluation_convert_predictions_from_1darray_to_nparray(GaussianNB):
    array = np.random.random(size=(30,))
    assert isinstance(_mock_evaluation(GaussianNB, array).predictions, np.ndarray), "nparray converted"
    assert _mock_evaluation(GaussianNB, array).predictions.shape == (30,), "Shape should not change."


def test_evaluation_convert_predictions_from_2darray_to_nparray(GaussianNB):
    array = np.random.random(size=(30, 5))
    assert isinstance(_mock_evaluation(GaussianNB, array).predictions, np.ndarray), "nparray converted"
    assert _mock_evaluation(GaussianNB, array).predictions.shape == (30, 5), "Shape should not change."


def test_evaluation_convert_predictions_from_series_to_nparray(GaussianNB):
    series = pd.Series(np.random.random(size=(30,)))
    assert isinstance(_mock_evaluation(GaussianNB, series).predictions, np.ndarray), "Series not converted."
    assert _mock_evaluation(GaussianNB, series).predictions.shape == (30,), "Shape should not change."


def test_evaluation_convert_predictions_from_dataframe_to_nparray(GaussianNB):
    dataframe = pd.DataFrame(np.random.random(size=(30, 5)))
    assert isinstance(_mock_evaluation(GaussianNB, dataframe).predictions, np.ndarray), "Dataframe not converted."
    assert _mock_evaluation(GaussianNB, dataframe).predictions.shape == (30, 5), "Shape should not change."


def test_evaluation_library_max_number_evaluations(GaussianNB):
    """ Test `max_number_of_evaluations` correctly restricts the number of evaluations in `top_evaluations`. """
    lib200 = EvaluationLibrary(m=200, sample=None)
    lib_unlimited = EvaluationLibrary(m=None, sample=None)

    worst_evaluation = _mock_evaluation(GaussianNB, score=(0., 0., 0.))
    lib200.save_evaluation(worst_evaluation)
    lib_unlimited.save_evaluation(worst_evaluation)

    for _ in range(200):
        lib200.save_evaluation(_mock_evaluation(GaussianNB))
        lib_unlimited.save_evaluation(_mock_evaluation(GaussianNB))

    assert len(lib200.top_evaluations) == 200, "After 201 evaluations, lib200 should be at its cap of 200."
    assert worst_evaluation not in lib200.top_evaluations, "The worst of 201 evaluations should not be present."
    assert len(lib_unlimited.top_evaluations) == 201, "All evaluations should be present."
    assert worst_evaluation in lib_unlimited.top_evaluations, "All evaluations should be present."


def test_evaluation_library_n_best(GaussianNB):
    """ Test `max_number_of_evaluations` correctly restricts the number of evaluations in `top_evaluations`. """
    lib = EvaluationLibrary(m=None, sample=None)

    best_evaluation = _mock_evaluation(GaussianNB, score=(1., 1., 1.))
    worst_evaluation = _mock_evaluation(GaussianNB, score=(0., 0., 0.))
    lib.save_evaluation(best_evaluation)
    lib.save_evaluation(worst_evaluation)

    for _ in range(10):
        lib.save_evaluation(_mock_evaluation(GaussianNB))

    assert len(lib.n_best(10)) == 10, "n_best(10) should return 10 results as more than 10 evaluations are saved."
    assert best_evaluation is lib.n_best(10)[0], "`best_evaluation` should be number one in `n_best`"
    assert worst_evaluation not in lib.n_best(10), "`worst_evaluation` should not be in the top 10 of 12 evaluations."
    assert len(lib.n_best(100)) == 12, "`n > len(lib.top_evaluations)` should return all evaluations."


def _test_subsample(sample, predictions, subsample):
    """ Test the `predictions` correctly get sampled to `subsample`. """
    lib = EvaluationLibrary(sample=sample)
    best_evaluation = _mock_evaluation(GaussianNB, predictions=predictions)
    lib.save_evaluation(best_evaluation)
    assert best_evaluation.predictions.shape == subsample.shape, "Subsample does not have expected shape."
    assert np.array_equal(best_evaluation.predictions, subsample), "Content of subsample differs from expected."


def test_evaluation_library_sample_np2d_prediction(GaussianNB):
    """ Test `prediction_sample` set with np.ndarray correctly samples np.ndarray predictions with ndim=2. """
    probabilities = np.random.random(size=(30, 5))
    _test_subsample(sample=np.asarray([0, 1, 3]), predictions=probabilities, subsample=probabilities[[0, 1, 3], :])


def test_evaluation_library_sample_pd2d_prediction(GaussianNB):
    """ Test `prediction_sample` set with np.ndarray correctly samples pd.DataFrame predictions. """
    probabilities = pd.DataFrame(np.random.random(size=(30, 5)))
    _test_subsample(sample=np.asarray([0, 1, 3]), predictions=probabilities, subsample=probabilities.iloc[[0, 1, 3], :])


def test_evaluation_library_sample_np1d_prediction(GaussianNB):
    """ Test `prediction_sample` set with np.ndarray correctly samples np.ndarray predictions with ndim=1. """
    probabilities = np.random.random(size=(30,))
    _test_subsample(sample=np.asarray([0, 1, 3]), predictions=probabilities, subsample=probabilities[[0, 1, 3]])


def test_evaluation_library_sample_pd1d_prediction(GaussianNB):
    """ Test `prediction_sample` set with np.ndarray correctly samples pd.Series predictions. """
    probabilities = pd.Series(np.random.random(size=(30,)))
    _test_subsample(sample=np.asarray([0, 1, 3]), predictions=probabilities, subsample=probabilities.iloc[[0, 1, 3]])
