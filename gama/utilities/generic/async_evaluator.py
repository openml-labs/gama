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
import gc
import logging
import multiprocessing
import os
import psutil
import queue
import struct
import time
import traceback
from typing import Optional, Callable, Dict, List
import uuid

from psutil import NoSuchProcess

try:
    import resource
except ModuleNotFoundError:
    resource = None  # type: ignore


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

    defaults: Dict, optional (default=None)
        Default parameter values shared between all submit calls.
        This allows these defaults to be transferred only once per process,
        instead of twice per call (to and from the subprocess).
        Only supports keyword arguments.
    """

    defaults: Dict = {}

    def __init__(
        self,
        n_workers: Optional[int] = None,
        memory_limit_mb: Optional[int] = None,
        logfile: Optional[str] = None,
        wait_time_before_forced_shutdown: int = 10,
    ):
        """
        Parameters
        ----------
        n_workers : int, optional (default=None)
            Maximum number of subprocesses to run for parallel evaluations.
            Defaults to `AsyncEvaluator.n_jobs`, using all cores unless overwritten.
        memory_limit_mb : int, optional (default=None)
            The maximum number of megabytes that this process and its subprocesses
            may use in total. If None, no limit is enforced.
            There is no guarantee the limit is not violated.
        logfile : str, optional (default=None)
            If set, recorded resource usage will be written to this file.
        wait_time_before_forced_shutdown : int (default=10)
            Number of seconds to wait between asking the worker processes to shut down
            and terminating them forcefully if they failed to do so.
        """
        self._has_entered = False
        self.futures: Dict[uuid.UUID, AsyncFuture] = {}
        self._processes: List[psutil.Process] = []
        self._n_jobs = n_workers
        self._memory_limit_mb = memory_limit_mb
        self._mem_violations = 0
        self._mem_behaved = 0
        self._logfile = logfile
        self._wait_time_before_forced_shutdown = wait_time_before_forced_shutdown

        self._input: multiprocessing.Queue = multiprocessing.Queue()
        self._output: multiprocessing.Queue = multiprocessing.Queue()
        self._command: multiprocessing.Queue = multiprocessing.Queue()
        pid = os.getpid()
        self._main_process = psutil.Process(pid)

    def __enter__(self):
        if self._has_entered:
            raise RuntimeError(
                "You can not use the same AsyncEvaluator in two different contexts."
            )
        self._has_entered = True

        self._input = multiprocessing.Queue()
        self._output = multiprocessing.Queue()

        log.debug(
            f"Process {self._main_process.pid} starting {self._n_jobs} subprocesses."
        )
        for _ in range(self._n_jobs):
            self._start_worker_process()
        self._log_memory_usage()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        log.debug(f"Signaling {len(self._processes)} subprocesses to stop.")

        for _ in self._processes:
            self._command.put("stop")

        for i in range(self._wait_time_before_forced_shutdown + 1):
            if self._command.empty():
                break
            time.sleep(1)

        self.clear_queue(self._input)
        self.clear_queue(self._output)
        self.clear_queue(self._command)

        # Even processes which 'stop' need to be 'waited',
        # otherwise they become zombie processes.
        while len(self._processes) > 0:
            try:
                self._stop_worker_process(self._processes[0])
            except psutil.NoSuchProcess:
                pass
        return False

    def clear_queue(self, queue: multiprocessing.Queue):
        while not queue.empty():
            try:
                queue.get(timeout=0.001)
            except:
                pass
        queue.close()

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
            self._control_memory_usage()
            self._log_memory_usage()

            try:
                completed_future = self._output.get(block=False)
            except queue.Empty:
                time.sleep(poll_time)
                continue

            match = self.futures.pop(completed_future.id)
            match.result, match.exception, match.traceback = (
                completed_future.result,
                completed_future.exception,
                completed_future.traceback,
            )
            self._mem_behaved += 1
            return match

    def _start_worker_process(self) -> psutil.Process:
        """ Start a new worker node and add it to the process pool. """
        mp_process = multiprocessing.Process(
            target=evaluator_daemon,
            args=(self._input, self._output, self._command, AsyncEvaluator.defaults),
            daemon=True,
        )
        mp_process.start()
        subprocess = psutil.Process(mp_process.pid)
        self._processes.append(subprocess)
        return subprocess

    def _stop_worker_process(self, process: psutil.Process):
        """ Terminate a new worker node and remove it from the process pool. """
        process.terminate()
        process.wait(timeout=60)
        self._processes.remove(process)

    def _control_memory_usage(self, threshold=0.05):
        """ Dynamically restarts or kills processes to adhere to memory constraints. """
        if self._memory_limit_mb is None:
            return
        # If the memory usage of all processes (the main process, and the evaluation
        # subprocesses) exceeds the maximum allowed memory usage, we have to terminate
        # one of them.
        # If we were never to start new processes, eventually all subprocesses would
        # likely be killed due to 'silly' pipelines (e.g. multiple polynomial feature
        # steps).
        # On the other hand if there is e.g. a big dataset, by always restarting we
        # will set up the same scenario for failure over and over again.
        # So we want to dynamically find the right amount of evaluation processes, such
        # that the total memory usage is not exceeded "too often".
        # Here `threshold` defines the ratio of processes that should be allowed to
        # fail due to memory constraints. Setting it too high might lead to aggressive
        # subprocess killing and underutilizing compute resources. If it is too low,
        # the number of concurrent jobs might shrink too slowly inducing a lot of
        # loss in compute time due to interrupted evaluations.
        # ! Like the rest of this module, I hate to use custom code with this,
        # in particular there is a risk that terminating the process might leave
        # the multiprocess queue broken.
        mem_proc = list(self._get_memory_usage())
        if sum(map(lambda x: x[1], mem_proc)) > self._memory_limit_mb:
            log.info(
                f"GAMA exceeded memory usage "
                f"({self._mem_violations}, {self._mem_behaved})."
            )
            self._log_memory_usage()
            self._mem_violations += 1
            # Find the process with the most memory usage, that is not the main process
            proc, _ = max(mem_proc[1:], key=lambda t: t[1])
            n_evaluations = self._mem_violations + self._mem_behaved
            fail_ratio = self._mem_violations / n_evaluations
            if fail_ratio < threshold or len(self._processes) == 1:
                # restart `pid`
                log.info(f"Terminating {proc.pid} due to memory usage.")
                self._stop_worker_process(proc)
                log.info("Starting new evaluations process.")
                self._start_worker_process()
            else:
                # More than one process left alive and a violation of the threshold,
                # requires killing a subprocess.
                self._mem_behaved = 0
                self._mem_violations = 0
                log.info(f"Terminating {proc.pid} due to memory usage.")
                self._stop_worker_process(proc)
            # todo: update the Future of the evaluation that was terminated.

    def _log_memory_usage(self):
        if not self._logfile:
            return
        mem_by_pid = self._get_memory_usage()
        mem_str = ",".join([f"{proc.pid},{mem_mb}" for (proc, mem_mb) in mem_by_pid])
        timestamp = datetime.datetime.now().isoformat()

        with open(self._logfile, "a") as memory_log:
            memory_log.write(f"{timestamp},{mem_str}\n")

    def _get_memory_usage(self):
        processes = [self._main_process] + self._processes
        for process in processes:
            try:
                yield process, process.memory_info()[0] / (2 ** 20)
            except NoSuchProcess:
                # can never be the main process anyway
                self._processes = [p for p in self._processes if p.pid != process.pid]
                self._start_worker_process()


def evaluator_daemon(
    input_queue: queue.Queue,
    output_queue: queue.Queue,
    command_queue: queue.Queue,
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
    command_queue: queue.Queue[Str]
        Queue to put commands for the subprocess.
        Queue should be managed by multiprocessing.manager.
    default_parameters: Dict, optional (default=None)
        Additional parameters to pass to AsyncFuture.Execute.
        This is useful to avoid passing lots of repetitive data through AsyncFuture.
    """
    try:
        while True:
            try:
                command_queue.get(block=False)
                break
            except queue.Empty:
                pass

            try:
                future = input_queue.get(block=False)
                future.execute(default_parameters)
                if future.result:
                    if isinstance(future.result, tuple):
                        result = future.result[0]
                    else:
                        result = future.result
                    if isinstance(result.error, MemoryError):
                        # Can't pickle MemoryErrors. Should work around this later.
                        result.error = "MemoryError"
                        gc.collect()
                output_queue.put(future)
            except (MemoryError, struct.error) as e:
                future.result = None
                future.exception = str(type(e))
                gc.collect()
                output_queue.put(future)
            except queue.Empty:
                pass
    except Exception as e:
        # There are no plans currently for recovering from any exception:
        print(f"Stopping daemon:{type(e)}:{str(e)}")
        traceback.print_exc()
