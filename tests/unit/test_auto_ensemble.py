from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier

from gama.genetic_programming.compilers.scikitlearn import compile_individual
from gama.genetic_programming.components.individual import Individual
from gama.postprocessing.ensemble import (
    EnsemblePostProcessing,
    fit_and_weight,
    EnsembleClassifier,
)
from gama.utilities.evaluation_library import Evaluation, EvaluationLibrary
from gama.utilities.metrics import Metric


def test_fit_and_weight():
    x, y = load_iris(return_X_y=True)
    good_estimator = LinearSVC()
    bad_estimator = LinearSVC(
        penalty="l1"
    )  # Not supported with default squared hinge loss solving the dual problem

    _, w = fit_and_weight((good_estimator, x, y, 1))
    assert 1 == w
    _, w = fit_and_weight((bad_estimator, x, y, 1))
    assert 0 == w


def test_code_export_produces_working_code(GNB, ForestPipeline):
    x, y = load_iris(return_X_y=True, as_frame=True)

    ensemble = EnsemblePostProcessing()

    ensemble._ensemble = EnsembleClassifier(
        Metric("neg_log_loss"),
        y,
        evaluation_library=EvaluationLibrary(n=None),
    )
    gnb = GNB
    gnb._to_pipeline = compile_individual
    fp = ForestPipeline
    fp._to_pipeline = compile_individual
    ensemble._ensemble._models = {
        "a": (Evaluation(gnb), 1),
        "b": (Evaluation(fp), 2),
    }
    ensemble._ensemble._metric = Metric("neg_log_loss")

    code = ensemble.to_code()
    local = {}
    exec(code, {}, local)
    exported_ensemble = local["ensemble"]  # should be defined in exported code
    assert isinstance(exported_ensemble, VotingClassifier)
    exported_ensemble.fit(x, y)
    assert 0.9 < exported_ensemble.score(x, y)
