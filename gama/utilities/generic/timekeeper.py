from contextlib import contextmanager
from typing import Iterator, Optional, NamedTuple
import logging

from .stopwatch import Stopwatch

log = logging.getLogger(__name__)


class Activity(NamedTuple):
    name: str
    time_limit: Optional[int]
    stopwatch: Stopwatch


class TimeKeeper:
    """ Simple object that helps keep track of time over multiple activities. """

    def __init__(self, total_time: int=0):
        self.total_time = total_time
        self.current_activity = None
        self.activities = []

    @property
    def total_time_remaining(self) -> float:
        """ Return time remaining in seconds. """
        if self.total_time > 0:
            return self.total_time - sum(map(lambda a: a.stopwatch.elapsed_time, self.activities))
        raise RuntimeError("Time Remaining only available if `total_time` was set on init.")

    @property
    def current_activity_time_elapsed(self) -> float:
        """ Return elapsed time in seconds of current activity. Raise RuntimeError if no current activity. """
        if self.current_activity is not None:
            return self.current_activity.stopwatch.elapsed_time
        else:
            raise RuntimeError("No activity in progress.")

    @property
    def current_activity_time_left(self) -> float:
        """ Return time left in seconds for current activity if a time limit was indicated. """
        if self.current_activity is not None and self.current_activity.time_limit is not None:
            return self.current_activity.time_limit - self.current_activity.stopwatch.elapsed_time
        elif self.current_activity is None:
            raise RuntimeError("No activity in progress.")
        else:
            raise RuntimeError("No time limit specified for activity {}.")

    @contextmanager
    def start_activity(self, activity: str, time_limit: Optional[int] = None) -> Iterator[Stopwatch]:
        """ Mark the start of a new activity and automatically time its duration.
            TimeManager does not currently support nested activities. """
        with Stopwatch() as sw:
            self.current_activity = Activity(activity, time_limit, sw)
            self.activities.append(self.current_activity)
            yield sw
        self.current_activity = None
        log.info("{} took {:.4f}s.".format(activity, sw.elapsed_time))
