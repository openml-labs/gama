import concurrent.futures


class AsyncExecutor(concurrent.futures.ProcessPoolExecutor):
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
        # Looping over self._processes.items() will result in `RuntimeError: dictionary changed size during iteration`
        processes = list(self._processes.values())
        self.shutdown(wait=False)
        for process in processes:
            process.terminate()
        return False

    def wait_first(self, futures, poll_time=.05):
        """ Wait for the first future in `futures` to complete through blocking calls every `poll_time` seconds.

        When waiting for futures, one should use ``concurrent.futures.wait``.
        Unfortunately, this is a blocking call, even if `timeout` is specified.
        This results in stopit's TimeoutException (and probably KeyboardInterrupt) being ignored.
        This `wait` patches it breaking a single blocking call up into multiple small ones,
        as any exceptions raised will be able to be handled in between the blocking calls.
        """
        done, not_done = set(), futures
        while len(done) == 0:
            done, not_done = concurrent.futures.wait(not_done, return_when='FIRST_COMPLETED', timeout=poll_time)
        return done, not_done
