from typing import Optional, Tuple, List
import numpy as np

from gama.genetic_programming.components import Individual
from gama.utilities.evaluation_library import Evaluation, EvaluationLibrary
from .unit_fixtures import pset, GaussianNB, RandomForestPipeline, LinearSVC


def _mock_evaluation(
        individual: Individual,
        score: Optional[Tuple[float, ...]] = None,
        predictions: Optional[np.ndarray] = object(),
        estimators: List[object] = None,
        start_time: int = 0,
        duration: int = 0,
        error: str = None
) -> Evaluation:
    evaluation = Evaluation(individual)
    if score is not None:
        evaluation.score = score
    else:
        evaluation.score = tuple(np.random.random(size=(3,)))
    evaluation.predictions = predictions
    evaluation.estimators = estimators
    evaluation.start_time = start_time
    evaluation.duration = duration
    evaluation.error = evaluation.error
    return evaluation


def test_evaluation_library_max_number_evaluations(GaussianNB):
    """ Test `max_number_of_evaluations` correctly restricts the number of evaluations in `top_evaluations`. """
    lib200 = EvaluationLibrary(max_number_of_evaluations=200, prediction_sample=None)
    lib_unlimited = EvaluationLibrary(max_number_of_evaluations=None, prediction_sample=None)

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
    lib = EvaluationLibrary(max_number_of_evaluations=None, prediction_sample=None)

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
