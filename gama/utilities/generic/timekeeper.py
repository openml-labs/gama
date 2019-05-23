from contextlib import contextmanager
from .stopwatch import Stopwatch
import logging
from typing import Iterator


log = logging.getLogger(__name__)


class TimeKeeper:
    """ Simple object that helps keep track of time over multiple activities. """

    def __init__(self, total_time: int=0):
        self.activity_durations = {}
        self.total_time = total_time

    @property
    def total_time_remaining(self) -> float:
        """ Return time remaining in seconds. """
        if self.total_time > 0:
            return self.total_time - sum(self.activity_durations.values())
        raise ValueError("Time Remaining only available if `total_time` was set on init.")

    @contextmanager
    def start_activity(self, activity: str) -> Iterator[Stopwatch]:
        """ Mark the start of a new activity and automatically time its duration.
            TimeManager does not currently support nested activities. """
        with Stopwatch() as sw:
            yield sw
        self.activity_durations[activity] = sw.elapsed_time
        log.info("{} took {:.4f}s.".format(activity, sw.elapsed_time))
