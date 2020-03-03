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
import logging
import multiprocessing
import queue
import time
import uuid
from typing import Optional, Callable, Dict, List

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

    def execute(self):
        """ Execute the function call `fn(*args, **kwargs)` and record results. """
        try:
            self.result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            self.exception = e


class AsyncEvaluator:
    """ Manages subprocesses on which arbitrary functions can be evaluated.

    The function and all its arguments must be picklable.
    Using the same AsyncEvaluator in two different contexts raises a `RuntimeError`.
    """

    n_jobs: int = multiprocessing.cpu_count()

    def __init__(self, n_workers: Optional[int] = None):
        """
        Parameters
        ----------
        n_workers : int, optional (default=None)
            Maximum number of subprocesses to run for parallel evaluations.
            Defaults to `AsyncEvaluator.n_jobs`, using all cores unless overwritten.
        """
        self._has_entered = False
        self.futures: Dict[uuid.UUID, AsyncFuture] = {}
        self._processes: List[multiprocessing.Process] = []
        self._n_jobs = n_workers if n_workers is not None else AsyncEvaluator.n_jobs

        self._queue_manager = multiprocessing.Manager()
        self._input_queue = self._queue_manager.Queue()
        self._output_queue = self._queue_manager.Queue()

    def __enter__(self):
        if self._has_entered:
            raise RuntimeError(
                "You can not use the same AsyncEvaluator in two different contexts."
            )
        self._has_entered = True

        self._input_queue = self._queue_manager.Queue()
        self._output_queue = self._queue_manager.Queue()

        log.debug(f"Starting {self._n_jobs} subprocesses.")
        for _ in range(self._n_jobs):
            subprocess = multiprocessing.Process(
                target=evaluator_daemon,
                args=(self._input_queue, self._output_queue),
                daemon=True,
            )
            self._processes.append(subprocess)
            subprocess.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        log.debug(f"Terminating {len(self._processes)} subprocesses.")
        # This is ugly as the subprocesses use shared queues.
        # It is in direct conflict with guidelines:
        # https://docs.python.org/3/library/multiprocessing.html#all-start-methods
        for subprocess in self._processes:
            subprocess.terminate()
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
        self._input_queue.put(future)
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
                completed_future = self._output_queue.get(block=False)
                matching_future = self.futures.pop(completed_future.id)
                matching_future.result, matching_future.exception = (
                    completed_future.result,
                    completed_future.exception,
                )
                return matching_future
            except queue.Empty:
                time.sleep(poll_time)
                continue


def evaluator_daemon(
    input_queue: queue.Queue, output_queue: queue.Queue, print_exit_message: bool = True
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
    print_exit_message: bool (default=True)
        If True, print to console the reason for shutting down.
        If False, shut down silently.
    """
    try:
        while True:
            future = input_queue.get()
            future.execute()
            output_queue.put(future)
    except KeyboardInterrupt:
        shutdown_message = "Helper process stopping due to keyboard interrupt."
    except (BrokenPipeError, EOFError):
        shutdown_message = "Helper process stopping due to a broken pipe or EOF."
    if print_exit_message:
        print(shutdown_message)
