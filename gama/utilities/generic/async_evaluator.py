"""
I don't want to be reinventing the wheel but I can't find a satisfying implementation.
I want to be able to execute arbitrary functions asynchronously on a different process.
Any ongoing subprocesses must immediately be able to be terminated without errors.
Results of cancelled subprocesses may be ignored.

`concurrent.futures.ProcessPoolExecutor` gets very close to the desired implementation,
but it has issues:
    - by default it waits for subprocesses to close on __exit__.
      Unfortunately it is possible the subprocesses can be running non-Python code,
      e.g. a C implementation of SVC whose subprocess won't end until fit is complete.
    - even if that is overwritten and no wait is performed,
      the subprocess will raise an error when it is done.
      Though that does not hinder the execution of the program,
      I don't want errors for expected behavior.
"""
import datetime
import logging
import multiprocessing
import os
import queue
import time
import traceback
import uuid
from typing import Optional, Callable, Dict, List

try:
    import resource
except ModuleNotFoundError:
    resource = None  # type: ignore

import psutil

from gama.logging import MACHINE_LOG_LEVEL

log = logging.getLogger(__name__)


class AsyncFuture:
    """ Reference to a function call executed on a different process. """

    def __init__(self, fn, *args, **kwargs):
        self.id = uuid.uuid4()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.exception = None
        self.traceback = None

    def execute(self, extra_kwargs):
        """ Execute the function call `fn(*args, **kwargs)` and record results. """
        try:
            # Don't update self.kwargs, as it will be pickled back to the main process
            kwargs = {**self.kwargs, **extra_kwargs}
            self.result = self.fn(*self.args, **kwargs)
        except Exception as e:
            self.exception = e
            self.traceback = traceback.format_exc()


class AsyncEvaluator:
    """ Manages subprocesses on which arbitrary functions can be evaluated.

    The function and all its arguments must be picklable.
    Using the same AsyncEvaluator in two different contexts raises a `RuntimeError`.

    class variables:
    n_jobs: int (default=multiprocessing.cpu_count())
        The default number of subprocesses to spawn.
        Ignored if the `n_workers` parameter is specified on init.
    defaults: Dict, optional (default=None)
        Default parameter values shared between all submit calls.
        This allows these defaults to be transferred only once per process,
        instead of twice per call (to and from the subprocess).
        Only supports keyword arguments.
    """

    n_jobs: int = multiprocessing.cpu_count()
    memory_limit_mb: int = 6000
    defaults: Dict = {}

    def __init__(
        self, n_workers: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        n_workers : int, optional (default=None)
            Maximum number of subprocesses to run for parallel evaluations.
            Defaults to `AsyncEvaluator.n_jobs`, using all cores unless overwritten.
        """
        self._has_entered = False
        self.futures: Dict[uuid.UUID, AsyncFuture] = {}
        self._processes: List[psutil.Process] = []
        self._n_jobs = n_workers if n_workers is not None else AsyncEvaluator.n_jobs

        self._queue_manager = multiprocessing.Manager()
        self._input = self._queue_manager.Queue()
        self._output = self._queue_manager.Queue()
        pid = os.getpid()
        self._main_process = psutil.Process(pid)

    def __enter__(self):
        if self._has_entered:
            raise RuntimeError(
                "You can not use the same AsyncEvaluator in two different contexts."
            )
        self._has_entered = True

        self._input = self._queue_manager.Queue()
        self._output = self._queue_manager.Queue()

        log.debug(
            f"Process {self._main_process.pid} starting {self._n_jobs} subprocesses."
        )
        for _ in range(self._n_jobs):
            mp_process = multiprocessing.Process(
                target=evaluator_daemon,
                args=(self._input, self._output, AsyncEvaluator.defaults),
                daemon=True,
            )
            mp_process.start()

            subprocess = psutil.Process(mp_process.pid)
            self._processes.append(subprocess)

            if resource and AsyncEvaluator.memory_limit_mb:
                limit = AsyncEvaluator.memory_limit_mb * (2 ** 20)
                resource.prlimit(subprocess.pid, resource.RLIMIT_AS, (limit, limit))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        log.debug(f"Terminating {len(self._processes)} subprocesses.")
        # This is ugly as the subprocesses use shared queues.
        # It is in direct conflict with guidelines:
        # https://docs.python.org/3/library/multiprocessing.html#all-start-methods
        for subprocess in self._processes:
            try:
                subprocess.terminate()
            except psutil.NoSuchProcess:
                log.debug(f"Daemon {subprocess.pid} stopped prematurely.")
        return False

    def submit(self, fn: Callable, *args, **kwargs) -> AsyncFuture:
        """ Submit fn(*args, **kwargs) to be evaluated on a subprocess.

        Parameters
        ----------
        fn: Callable
            Function to call on a subprocess.
        args
            Positional arguments to call `fn` with.
        kwargs
            Keyword arguments to call `fn` with.

        Returns
        -------
        AsyncFuture
            A Future of which the `result` or `exception` field will be populated
            once evaluation is finished.
        """
        future = AsyncFuture(fn, *args, **kwargs)
        self.futures[future.id] = future
        self._input.put(future)
        return future

    def wait_next(self, poll_time: float = 0.05) -> AsyncFuture:
        """ Wait until an AsyncFuture has been completed and return it.


        Parameters
        ----------
        poll_time: float (default=0.05)
            Time to sleep between checking if a future has been completed.

        Returns
        -------
        AsyncFuture
            The completed future that completed first.

        Raises
        ------
        RuntimeError
            If all futures have already been completed and returned.
        """
        if len(self.futures) == 0:
            raise RuntimeError("No Futures queued, must call `submit` first.")

        while True:
            try:
                completed_future = self._output.get(block=False)
                match = self.futures.pop(completed_future.id)
                match.result, match.exception, match.traceback = (
                    completed_future.result,
                    completed_future.exception,
                    completed_future.traceback,
                )
                self._log_memory_usage()
                return match
            except queue.Empty:
                time.sleep(poll_time)
                continue

    def _log_memory_usage(self):
        processes = [self._main_process] + self._processes
        mem_by_pid = [(p.pid, p.memory_info()[0] / (2 ** 20)) for p in processes]
        mem_str = ",".join([f"{pid},{mem_mb}" for (pid, mem_mb) in mem_by_pid])
        timestamp = datetime.datetime.now().isoformat()
        log.log(MACHINE_LOG_LEVEL, f"M,{timestamp},{mem_str}")


def evaluator_daemon(
    input_queue: queue.Queue,
    output_queue: queue.Queue,
    default_parameters: Optional[Dict] = None,
):
    """ Function for daemon subprocess that evaluates functions from AsyncFutures.

    Parameters
    ----------
    input_queue: queue.Queue[AsyncFuture]
        Queue to get AsyncFuture from.
        Queue should be managed by multiprocessing.manager.
    output_queue: queue.Queue[AsyncFuture]
        Queue to put AsyncFuture to.
        Queue should be managed by multiprocessing.manager.
    default_parameters: Dict, optional (default=None)
        Additional parameters to pass to AsyncFuture.Execute.
        This is useful to avoid passing lots of repetitive data through AsyncFuture.
    """
    try:
        while True:
            try:
                future = input_queue.get()
                future.execute(default_parameters)
                if future.result and isinstance(future.result.error, MemoryError):
                    # Can't pickle MemoryErrors. Should work around this later.
                    future.result.error = "MemoryError"
                output_queue.put(future)
            except MemoryError:
                future.result = None
                future.exception = "ProcessMemoryError"
                output_queue.put(future)
    except Exception as e:
        # There are no plans currently for recovering from any exception:
        print(f"Stopping daemon:{type(e)}:{str(e)}")
    # Not sure if required: give main process time to process log message.
    time.sleep(5)
