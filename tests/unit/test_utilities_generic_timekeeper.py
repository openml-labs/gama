import time
import pytest
from gama.utilities.generic.timekeeper import TimeKeeper


def _time_approx(seconds: int):
    return pytest.approx(seconds, abs=0.03)


def test_timekeeper_total_time_remaning_error_if_total_time_zero():
    """ Ensure `total_time_remaining` is unavailable if `total_time` is not set. """
    timekeeper = TimeKeeper()
    with pytest.raises(RuntimeError):
        _ = timekeeper.total_time_remaining


def test_timekeeper_stopwatch_normal_behavior():
    """ Normal stopwatch functionality for stopwatch returned by context manager. """
    timekeeper = TimeKeeper()
    with timekeeper.start_activity("test activity", time_limit=3) as sw:
        assert _time_approx(0) == sw.elapsed_time
        assert _time_approx(0) == timekeeper.current_activity_time_elapsed
        assert _time_approx(3) == timekeeper.current_activity_time_left
        assert sw._is_running

        time.sleep(1)

        assert _time_approx(1) == sw.elapsed_time
        assert _time_approx(1) == timekeeper.current_activity_time_elapsed
        assert _time_approx(2) == timekeeper.current_activity_time_left
        assert sw._is_running

    time.sleep(1)
    assert not sw._is_running
    assert _time_approx(1) == sw.elapsed_time

    with pytest.raises(RuntimeError) as error:
        _ = timekeeper.current_activity_time_elapsed
    assert "No activity in progress." in str(error.value)

    with pytest.raises(RuntimeError) as error:
        _ = timekeeper.current_activity_time_left
    assert "No activity in progress." in str(error.value)


def test_timekeeper_total_remaining_time():
    """ Ensure total remaining time is correct across activities. """
    total_time = 10
    timekeeper = TimeKeeper(total_time=total_time)
    assert timekeeper.total_time_remaining == total_time

    with timekeeper.start_activity("part one"):
        time.sleep(1)

    time.sleep(1)
    assert _time_approx(1) == timekeeper.activities[-1].stopwatch.elapsed_time
    assert _time_approx(total_time - 1) == timekeeper.total_time_remaining
