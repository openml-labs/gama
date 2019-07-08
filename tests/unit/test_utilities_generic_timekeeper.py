import time
import pytest
from gama.utilities.generic.timekeeper import TimeKeeper


ROUND_ERROR = 0.01


def test_timekeeper_total_time_remaning_error_if_total_time_zero():
    """ Ensure `total_time_remaining` is unavailable if `total_time` is not set. """
    timekeeper = TimeKeeper(total_time=0)
    with pytest.raises(RuntimeError):
        _ = timekeeper.total_time_remaining


def test_timekeeper_stopwatch_normal_behavior():
    """ Ensure normal stopwatch functionality for stopwatch returned by context manager. """
    timekeeper = TimeKeeper()
    with timekeeper.start_activity('test activity') as sw:
        assert pytest.approx(0, abs=ROUND_ERROR) == sw.elapsed_time
        assert sw._is_running
        time.sleep(1)
        assert pytest.approx(1, abs=ROUND_ERROR) == sw.elapsed_time
        assert sw._is_running

    time.sleep(1)
    assert not sw._is_running
    assert pytest.approx(1, abs=ROUND_ERROR) == sw.elapsed_time


def test_timekeeper_total_remaining_time():
    """ Ensure total remaining time is correct across activities. """
    total_time = 10
    timekeeper = TimeKeeper(total_time=total_time)
    assert timekeeper.total_time_remaining == total_time

    activity_length = 1
    with timekeeper.start_activity('part one'):
        time.sleep(activity_length)

    time.sleep(1)
    assert pytest.approx(activity_length, abs=ROUND_ERROR) == timekeeper.activities[-1].stopwatch.elapsed_time
    assert pytest.approx(total_time - activity_length, abs=ROUND_ERROR) == timekeeper.total_time_remaining
