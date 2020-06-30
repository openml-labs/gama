from typing import Optional, Tuple, List, Union
import uuid
import numpy as np
import pandas as pd

from gama.genetic_programming.components import Individual
from gama.utilities.evaluation_library import Evaluation, EvaluationLibrary


def _short_name():
    return str(uuid.uuid4())[:4]


def _mock_evaluation(
    individual: Individual,
    predictions: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = np.zeros(30,),
    score: Optional[Tuple[float, ...]] = None,
    estimators: List[object] = None,
    start_time: int = 0,
    duration: int = 0,
    error: str = None,
) -> Evaluation:
    return Evaluation(
        individual,
        predictions,
        score if score is not None else tuple(np.random.random(size=(3,))),
        estimators,
        start_time,
        duration,
        error,
    )


def test_evaluation_convert_predictions_from_1darray_to_nparray(GNB):
    array = np.random.random(size=(30,))
    pred_is_np_array = isinstance(_mock_evaluation(GNB, array).predictions, np.ndarray)
    assert pred_is_np_array, "nparray converted"
    assert (30,) == _mock_evaluation(GNB, array).predictions.shape, "Shape changed."


def test_evaluation_convert_predictions_from_2darray_to_nparray(GNB):
    array = np.random.random(size=(30, 5))
    pred_is_np_array = isinstance(_mock_evaluation(GNB, array).predictions, np.ndarray)
    assert pred_is_np_array, "nparray converted"
    assert (30, 5) == _mock_evaluation(GNB, array).predictions.shape, "Shape changed."


def test_evaluation_convert_predictions_from_series_to_nparray(GNB):
    series = pd.Series(np.random.random(size=(30,)))
    pred_is_np_array = isinstance(_mock_evaluation(GNB, series).predictions, np.ndarray)
    assert pred_is_np_array, "Series not converted to np array."
    assert (30,) == _mock_evaluation(GNB, series).predictions.shape, "Shape changed."


def test_evaluation_convert_predictions_from_dataframe_to_nparray(GNB):
    df = pd.DataFrame(np.random.random(size=(30, 5)))
    pred_is_np_array = isinstance(_mock_evaluation(GNB, df).predictions, np.ndarray)
    assert pred_is_np_array, "Dataframe not converted to np array."
    assert (30, 5) == _mock_evaluation(GNB, df).predictions.shape, "Shape changed."


def test_evaluation_library_max_number_evaluations(GNB):
    """ `max_number_of_evaluations` restricts the size of `top_evaluations`. """
    lib200 = EvaluationLibrary(m=200, sample=None, cache=_short_name())
    lib_unlimited = EvaluationLibrary(m=None, sample=None, cache=_short_name())

    try:
        worst_evaluation = _mock_evaluation(GNB, score=(0.0, 0.0, 0.0))
        lib200.save_evaluation(worst_evaluation)
        lib_unlimited.save_evaluation(worst_evaluation)

        for _ in range(200):
            lib200.save_evaluation(_mock_evaluation(GNB))
            lib_unlimited.save_evaluation(_mock_evaluation(GNB))

        assert 200 == len(
            lib200.top_evaluations
        ), "After 201 evaluations, lib200 should be at its cap of 200."
        assert (
            worst_evaluation not in lib200.top_evaluations
        ), "The worst of 201 evaluations should not be present."
        assert 201 == len(
            lib_unlimited.top_evaluations
        ), "All evaluations should be present."
        assert (
            worst_evaluation in lib_unlimited.top_evaluations
        ), "All evaluations should be present."
    finally:
        lib200.clear_cache()
        lib_unlimited.clear_cache()


def test_evaluation_library_n_best(GNB):
    """ Test `n_best` normal usage.  """
    lib = EvaluationLibrary(m=None, sample=None, cache=_short_name())

    try:
        best_evaluation = _mock_evaluation(GNB, score=(1.0, 1.0, 1.0))
        worst_evaluation = _mock_evaluation(GNB, score=(0.0, 0.0, 0.0))
        lib.save_evaluation(best_evaluation)
        lib.save_evaluation(worst_evaluation)

        for _ in range(10):
            lib.save_evaluation(_mock_evaluation(GNB))

        assert (
            len(lib.n_best(10)) == 10
        ), "n_best(10) should return 10 results as more than 10 evaluations are saved."
        assert (
            best_evaluation is lib.n_best(10)[0]
        ), "`best_evaluation` should be number one in `n_best`"
        assert worst_evaluation not in lib.n_best(
            10
        ), "`worst_evaluation` should not be in the top 10 of 12 evaluations."
        assert (
            len(lib.n_best(100)) == 12
        ), "`n > len(lib.top_evaluations)` should return all evaluations."
    finally:
        lib.clear_cache()


def _test_subsample(sample, predictions, subsample, individual):
    """ Test the `predictions` correctly get sampled to `subsample`. """
    lib = EvaluationLibrary(sample=sample, cache=_short_name())

    try:
        best_evaluation = _mock_evaluation(individual, predictions=predictions)
        lib.save_evaluation(best_evaluation)
        assert (
            subsample.shape == best_evaluation.predictions.shape
        ), "Subsample does not have expected shape."
        assert np.array_equal(
            subsample, best_evaluation.predictions
        ), "Content of subsample differs from expected."
    finally:
        lib.clear_cache()


def test_evaluation_library_sample_np2d_prediction(GNB):
    """ `prediction_sample` set with np.ndarray samples predictions with ndim=2. """
    probabilities = np.random.random(size=(30, 5))
    _test_subsample(
        sample=np.asarray([0, 1, 3]),
        predictions=probabilities,
        subsample=probabilities[[0, 1, 3], :],
        individual=GNB,
    )


def test_evaluation_library_sample_pd2d_prediction(GNB):
    """ `prediction_sample` set with np.ndarray samples pd.DataFrame predictions. """
    probabilities = pd.DataFrame(np.random.random(size=(30, 5)))
    _test_subsample(
        sample=np.asarray([0, 1, 3]),
        predictions=probabilities,
        subsample=probabilities.iloc[[0, 1, 3], :],
        individual=GNB,
    )


def test_evaluation_library_sample_np1d_prediction(GNB):
    """ `prediction_sample` set with np.ndarray samples predictions with ndim=1. """
    probabilities = np.random.random(size=(30,))
    _test_subsample(
        sample=np.asarray([0, 1, 3]),
        predictions=probabilities,
        subsample=probabilities[[0, 1, 3]],
        individual=GNB,
    )


def test_evaluation_library_sample_pd1d_prediction(GNB):
    """ `prediction_sample` set with np.ndarray samples pd.Series predictions. """
    probabilities = pd.Series(np.random.random(size=(30,)))
    _test_subsample(
        sample=np.asarray([0, 1, 3]),
        predictions=probabilities,
        subsample=probabilities.iloc[[0, 1, 3]],
        individual=GNB,
    )
