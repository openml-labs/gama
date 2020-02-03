import time
import pytest
from gama.utilities.generic.stopwatch import Stopwatch


ROUND_ERROR = 0.02


def test_stopwatch_initialization_zero():
    """ Test that elapsed time is 0 if stopwatch is not started yet. """
    sw = Stopwatch()
    assert pytest.approx(0, abs=ROUND_ERROR) == sw.elapsed_time


def test_stopwatch_elapsed_time_while_running():
    """ Tests that elapsed_time increments as expected while running. """
    with Stopwatch() as sw:
        assert pytest.approx(0, abs=ROUND_ERROR) == sw.elapsed_time
        time.sleep(1)
        assert pytest.approx(1, abs=ROUND_ERROR) == sw.elapsed_time


def test_stopwatch_elapsed_time_after_running():
    """ Tests that time elapsed is stored after exiting the context. """
    with Stopwatch() as sw:
        time.sleep(1)
    time.sleep(1)
    assert pytest.approx(1, abs=ROUND_ERROR) == sw.elapsed_time
