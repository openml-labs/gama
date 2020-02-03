"""
I don't want to be reinventing the wheel but I can't find a satisfying implementation.
I want to be able to execute arbitrary functions asynchronously on a different process.
Any ongoing subprocesses must immediately be able to be terminated without errors.
Results of cancelled subprocesses may be ignored.

`concurrent.futures.ProcessPoolExecutor` gets very close to the desired implementation,
but it has issues:
    - by default it waits for subprocesses to close on __exit__.
      Unfortunately it is possible the subprocesses can be running non-Python code,
      e.g. a C implementation of SVC, meaning the subprocess won't end until the SVC fit is complete.
    - even if that is overwritten and no wait is performed, the subprocess will throw an error when it is done.
      Though that does not hinder the execution of the program, I don't want errors for expected behavior.
"""

import logging
import multiprocessing
import queue
import time
import uuid
from typing import Optional

log = logging.getLogger(__name__)


class AsyncFuture:

    def __init__(self, fn, *args, **kwargs):
        self._id = uuid.uuid4()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.exception = None

    def execute(self):
        try:
            self.result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            self.exception = e


def evaluator_daemon(
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        print_exit_message: bool = True):
    """ Function for daemon subprocess that evaluates functions from AsyncFutures.

    Parameters
    ----------
    input_queue: queue.Queue[AsyncFuture]
        Queue to get AsyncFuture from. Queue should be managed by multiprocessing.manager.
    output_queue: queue.Queue[AsyncFuture]
        Queue to put AsyncFuture to. Queue should be managed by multiprocessing.manager.
    print_exit_message: bool (default=True)
        If True, print to console the reason for shutting down.
        If False, shut down silently.
    Returns
    -------

    """
    try:
        while True:
            future = input_queue.get()
            future.execute()
            output_queue.put(future)
    except KeyboardInterrupt:
        shutdown_message = 'Helper process stopping due to keyboard interrupt.'
    except (BrokenPipeError, EOFError):
        shutdown_message = 'Helper process stopping due to a broken pipe or EOF.'
    if print_exit_message:
        print(shutdown_message)


def clear_queue(queue_: queue.Queue) -> int:
    """ Dequeue items until the queue is empty. Returns number of items removed from the queue. """
    items_cleared = 0
    while True:
        try:
            queue_.get(block=False)
            items_cleared += 1
        except queue.Empty:
            break
        except EOFError:
            log.warning('EOFError occurred while clearing queue.', exc_info=True)
    return items_cleared


class AsyncEvaluator:
    n_jobs: int = multiprocessing.cpu_count()

    def __init__(self, n_workers: Optional[int] = None):
        """

        Parameters
        ----------
        n_workers : int, optional (default=None)
            Maximum number of subprocesses to run for parallel evaluations.
            If None, use `AsyncEvaluator.n_jobs` which defaults to multiprocessing.cpu_count().
        """
        self.futures = []
        self._processes = []
        self._n_jobs = n_workers if n_workers is not None else AsyncEvaluator.n_jobs

        self._queue_manager = multiprocessing.Manager()
        self._input_queue = self._queue_manager.Queue()
        self._output_queue = self._queue_manager.Queue()

    def __enter__(self):
        log.debug(f"Starting {self._n_jobs} subprocesses.")
        for _ in range(self._n_jobs):
            subprocess = multiprocessing.Process(
                target=evaluator_daemon,
                args=(self._input_queue, self._output_queue),
                daemon=True
            )
            self._processes.append(subprocess)
            subprocess.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        log.debug(f"Terminating {len(self._processes)} subprocesses.")
        # This is ugly as the subprocesses use shared queues and in direct conflict with guidelines:
        # https://docs.python.org/3/library/multiprocessing.html#all-start-methods
        for subprocess in self._processes:
            subprocess.terminate()

        self._processes = []
        self.futures = []

        # No direct `clear` method? AutoProxy[Queue] has `queue` attribute.
        clear_queue(self._input_queue)
        clear_queue(self._output_queue)
        return False

    def submit(self, fn, *args, **kwargs) -> AsyncFuture:
        future = AsyncFuture(fn, *args, **kwargs)
        self.futures.append(future)
        self._input_queue.put(future)
        return future

    def wait_next(self, poll_time: float = 0.05) -> AsyncFuture:
        while True:
            try:
                return self._output_queue.get(block=False)
            except queue.Empty:
                time.sleep(poll_time)
                continue
