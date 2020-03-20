import time


class Stopwatch:
    """ A context manager that keeps track of wall clock time spent. """

    def __init__(self, timing_function=time.time):
        """

        Parameters
        ----------
        timing_function: Callable (default=time.time)
            The function used to measure time, e.g. time.time or time.process_time
        """
        self._is_running = False
        self._get_time = timing_function
        self._start = 0
        self._end = 0

    def __enter__(self):
        self._is_running = True
        self._start = self._get_time()
        return self

    def __exit__(self, *args):
        self._end = self._get_time()
        self._is_running = False
        return False  # do not suppress any exceptions.

    @property
    def elapsed_time(self):
        """ Time spent in seconds during with-statement (so far, if not yet exited). """
        if self._is_running:
            return self._get_time() - self._start
        else:
            return self._end - self._start
