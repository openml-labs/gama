import logging
from typing import Optional, Callable, Dict

from dask.distributed import Client, LocalCluster, as_completed, Future

log = logging.getLogger(__name__)


class AsyncEvaluator:
    """Manages subprocesses on which arbitrary functions can be evaluated.

    The function and all its arguments must be picklable.
    Using the same AsyncEvaluator in two different contexts raises a `RuntimeError`.

    defaults: Dict, optional (default=None)
        Default parameter values shared between all submit calls.
        This allows these defaults to be transferred only once per process,
        instead of twice per call (to and from the subprocess).
        Only supports keyword arguments.
    """

    defaults: Dict = {}
    provided_cluster = None

    def __init__(
        self,
        n_workers: int = 1,
        memory_limit_mb: Optional[int] = None,
        logfile: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        n_workers : int (default=1)
            Maximum number of subprocesses to run for parallel evaluations.
        memory_limit_mb : int, optional (default=None)
            The maximum number of megabytes that this process and its subprocesses
            may use in total. If None, no limit is enforced.
            There is no guarantee the limit is not violated.
        logfile : str, optional (default=None)
            If set, recorded resource usage will be written to this file.
        """
        self.defaults = {}
        self._n_jobs = n_workers
        self._memory_limit_mb = memory_limit_mb

        self.cluster = None
        self.client = None
        self.futures = None

    @property
    def job_queue_size(self) -> int:
        """The number of jobs that are waiting for an available worker."""
        # Dask does not distinguish between a future which is currently being evaluated
        # or one which is waiting for a worker, both are "pending".
        if self.futures:
            pending_futures = [f for f in self.futures.futures if f.status == "pending"]
            return len(pending_futures) - self._n_jobs
        return 0

    def __enter__(self):
        if self.cluster or self.client:
            raise RuntimeError(
                "You can not use the same `Workers` object in two different contexts."
            )

        mem_limit = (
            f"{self._memory_limit_mb/self._n_jobs}MB"
            if self._memory_limit_mb
            else "auto"
        )
        if not AsyncEvaluator.provided_cluster:
            log.debug(f"Starting local cluster: {mem_limit=}")
            self.cluster = LocalCluster(
                n_workers=self._n_jobs,
                processes=False,
                memory_limit=mem_limit,
                silence_logs=logging.ERROR,
            )
        else:
            log.debug(f"Using provided cluster: {mem_limit=}")
            print("using provided cluster")
            self.cluster = AsyncEvaluator.provided_cluster
        self.client = Client(self.cluster)

        for key, value in AsyncEvaluator.defaults.items():
            self.defaults[key] = self.client.scatter(value, broadcast=True)
        self.futures = as_completed([])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        log.debug("Clearing futures and closing client.")
        self.futures.clear()
        self.client.close()
        if not AsyncEvaluator.provided_cluster:
            log.debug("Stopping local cluster")
            self.client.shutdown()

        self.cluster = None
        self.client = None
        self.futures = None
        log.debug("Stopped local cluster")

    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        """Submit fn(*args, **kwargs) to be evaluated on a subprocess.

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
        if not (self.client and self.futures):
            raise RuntimeError("Submit called before starting cluster.")
        future = self.client.submit(fn, *args, **kwargs, **self.defaults)
        self.futures.add(future)
        return future

    def wait_next(self, poll_time: float = 0.05) -> Future:
        """Wait until an AsyncFuture has been completed and return it.


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
        if not (self.futures):
            raise RuntimeError("Submit called before starting cluster.")
        return next(self.futures)
