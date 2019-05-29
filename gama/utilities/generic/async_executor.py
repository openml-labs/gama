from concurrent.futures import ProcessPoolExecutor


class AsyncExecutor(ProcessPoolExecutor):
    """ ContextManager for ProcessPoolExecutor which on exit terminates subprocesses and does not wait on shutdown.

    By default, when concurrent.futures.ProcessPoolExecutor is used as a context manager, on exiting the context
    the executor cancels planned jobs, but waits for current running jobs to finish. This behavior is undesirable
    in cases where the jobs may run for a long time but results are not necessarily needed. For instance in
    pipeline search, where one can easily abort training a pipeline and only use results obtained so far.

    Additionally, even if not `wait=False` on the shutdown, it does still leave its subprocesses running.
    This is problematic if those subprocesses may interfer with later logic. In particular, jobs which handle
    pipeline evaluations may still write to the cache directory while it is being removed from disk.

    As far as I know, there is no proper way to shutdown these child processes without accessing internal variables.
    """
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('terminating child processes')
        for pid, process in self._processes.items():
            process.terminate()
        self.shutdown(wait=False)