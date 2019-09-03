import pytest

from gama.postprocessing import NoPostProcessing


def test_no_post_processing():
    """ Test that NoPostProcessing does nothing and no model is returned. """
    postprocessing = NoPostProcessing()
    postprocessing.dynamic_defaults(None)
    model = postprocessing.post_process()
    assert pytest.approx(0.0) == postprocessing.time_fraction
    assert model is None
