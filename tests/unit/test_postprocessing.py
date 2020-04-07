import pytest

from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline

from gama.postprocessing import NoPostProcessing, BestFitPostProcessing


def test_no_post_processing():
    """ Test that NoPostProcessing does nothing and no model is returned. """
    postprocessing = NoPostProcessing()
    model = postprocessing.post_process()
    assert pytest.approx(0.0) == postprocessing.time_fraction
    assert model is None


def test_best_fit_processing(GNB):
    x, y = load_iris(return_X_y=True)
    bestfit = BestFitPostProcessing()

    model = bestfit.post_process(x, y, timeout=0, selection=[GNB])
    assert isinstance(model, Pipeline)
    assert 0.9 < model.score(x, y)  # accuracy

    code = bestfit.to_code(None)  # with preprocessing is tested in system test
    local = {}
    exec(code, {}, local)
    pipeline = local["pipeline"]  # should be defined in exported code
    assert isinstance(pipeline, Pipeline)
    pipeline.fit(x, y)
    assert 0.9 < pipeline.score(x, y)


# Ensemble processing and code export is for now only tested in system test,
# because the set up is quite complicated.
