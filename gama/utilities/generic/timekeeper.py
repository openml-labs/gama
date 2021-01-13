from contextlib import contextmanager
from typing import Iterator, Optional, NamedTuple, List, Any
import logging

from .stopwatch import Stopwatch

log = logging.getLogger(__name__)


class Activity(NamedTuple):
    name: str
    stopwatch: Stopwatch
    time_limit: Optional[int] = None

    @property
    def time_left(self) -> float:
        """ Time left in seconds.

        Raises a TypeError if `time_limit` was not specified.
        """
        return self.time_limit - self.stopwatch.elapsed_time

    def exceeded_limit(self, margin: float = 0.0) -> float:
        """ True iff a limit was specified and it is exceeded by `margin` seconds. """
        if self.time_limit is not None:
            return self.time_limit - self.stopwatch.elapsed_time < -margin
        return False


class TimeKeeper:
    """ Simple object that helps keep track of time over multiple activities. """

    def __init__(self, total_time: Optional[int] = None):
        """
        Parameters
        ----------
        total_time: int, optional (default=None)
            The total time available across activities.
            If set to None, the `total_time_remaining` property will be unavailable.

        """
        self.total_time = total_time
        self.current_activity: Optional[Activity] = None
        self.activities: List[Activity] = []

    @property
    def total_time_remaining(self) -> float:
        """ Return time remaining in seconds. """
        if self.total_time is not None:
            return self.total_time - sum(
                map(lambda a: a.stopwatch.elapsed_time, self.activities)
            )
        raise RuntimeError(
            "Time Remaining only available if `total_time` was set on init."
        )

    @property
    def current_activity_time_elapsed(self) -> float:
        """ Return elapsed time in seconds of current activity.

        Raise RuntimeError if no current activity.
        """
        if self.current_activity is not None:
            return self.current_activity.stopwatch.elapsed_time
        else:
            raise RuntimeError("No activity in progress.")

    @property
    def current_activity_time_left(self) -> float:
        """ Return time left in seconds of current activity.

        Raise RuntimeError if no current activity.
        """
        if (
            self.current_activity is not None
            and self.current_activity.time_limit is not None
        ):
            return (
                self.current_activity.time_limit
                - self.current_activity.stopwatch.elapsed_time
            )
        elif self.current_activity is None:
            raise RuntimeError("No activity in progress.")
        else:
            raise RuntimeError("No time limit set for current activity.")

    @contextmanager
    def start_activity(
        self,
        activity: str,
        time_limit: Optional[int] = None,
        activity_meta: Optional[List[Any]] = None,
    ) -> Iterator[Stopwatch]:
        """ Mark the start of a new activity and automatically time its duration.
            TimeManager does not currently support nested activities.

        Parameters
        ----------
        activity: str
            Name of the activity for reference in current activity or later look-ups.
        time_limit: int, optional (default=None)
            Intended time limit of the activity in seconds.
            Used to calculate time remaining.
        activity_meta: List[Any], optional (default=None)
            Any additional information about the activity to be logged.

        Returns
        -------
        ContextManager
            A context manager which when exited notes the end of the started activity.
        """
        if activity_meta is None:
            activity_meta = []
        act = f"{activity} {','.join(map(str, activity_meta))}"
        log.info(f"START: {act}")

        with Stopwatch() as sw:
            self.current_activity = Activity(activity, sw, time_limit)
            self.activities.append(self.current_activity)
            yield sw
        self.current_activity = None
        log.info(f"STOP: {act} after {sw.elapsed_time:.4f}s.")
