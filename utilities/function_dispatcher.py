"""
I tried to use multiprocessing.pool.Pool instead, but it failed silently.
This can happen when objects passed are not pickleable, but I used the same objects.
So far now, I will have to work with this.
"""

import logging
import multiprocessing as mp
import queue
import random
import time
import uuid

import numpy as np

log = logging.getLogger(__name__)


def evaluator_daemon(input_queue, output_queue, fn, seed=0, print_exit_message=False):
    random.seed(seed)
    np.random.seed(seed)

    shutdown_message = 'Helper process stopping normally.'
    try:
        while True:
            identifier, input_ = input_queue.get()
            output = fn(input_)
            output_queue.put((identifier, output))
    except KeyboardInterrupt:
        shutdown_message = 'Helper process stopping due to keyboard interrupt.'
    except (BrokenPipeError, EOFError):
        shutdown_message = 'Helper process stopping due to a broken pipe or EOF.'

    if print_exit_message:
        print(shutdown_message)


def clear_queue(queue_):
    """ Dequeue items until the queue is empty. Returns number of items removed from the queue. """
    items_cleared = 0
    while True:
        try:
            queue_.get(block=False)
            items_cleared += 1
        except queue.Empty:
            break
    return items_cleared


class FunctionDispatcher(object):
    """ A manager for evaluating functions async in the background using multi-processing.

    This object will spawn `n_jobs` child processes which will run `func` on any input given through `queue_evaluation`.
    Return values of evaluations can be obtained by calling `get_next_result`.
    To keep track of which input leads to which output, `queue_evaluation` returns a unique identifier for each call.
    Finally, `get_next_result` will return the identifier and `item` alongside the output of `func(item)`.
    """

    def __init__(self, n_jobs, func, toolbox):
        if n_jobs <= 0:
            raise ValueError("n_jobs must be at least 1.")

        mp_manager = mp.Manager()
        self._input_queue = mp_manager.Queue()
        self._output_queue = mp_manager.Queue()
        self._n_jobs = n_jobs
        self._func = func
        self._toolbox = toolbox

        self._job_map = {}
        self._child_processes = []

    def start(self):
        """ Start child processes. """
        log.info('Starting {} child processes.'.format(self._n_jobs))
        self._job_map = {}
        for _ in range(self._n_jobs):
            p = mp.Process(target=evaluator_daemon,
                           args=(self._input_queue, self._output_queue, self._func))
            p.daemon = True
            self._child_processes.append(p)
            p.start()

    def stop(self):
        """ Dequeue all outstanding jobs, discard saved results and terminate child processes. """
        log.info('Terminating {} child processes.'.format(len(self._child_processes)))
        for process in self._child_processes:
            process.terminate()
        self._child_processes = []

        nr_cancelled = clear_queue(self._input_queue)
        nr_discarded = clear_queue(self._output_queue)
        log.debug("Cancelled {} outstanding jobs. Discarded {} results. Terminated {} currently executing jobs."
                  .format(nr_cancelled, nr_discarded, len(self._job_map) - nr_cancelled - nr_discarded))

    def restart(self):
        """ This is equivalent to calling `stop` then `start`."""
        self.stop()
        self.start()

    def queue_evaluation(self, item):
        """ Queue an item to be processed by a child process according to `func` passed to __init__.

        Returns the identifier of the job.
        """
        identifier = uuid.uuid4()
        self._job_map[identifier] = item
        self._input_queue.put((identifier, item))
        return identifier

    def _get_next_from_daemons(self):
        # If we just used the blocking queue.get, then KeyboardInterrupts/Timeout would not work.
        # Previously, specifying a timeout worked, but for some reason that seems no longer the case.
        # Using timeout prevents the stopit.Timeout exception from being received.
        # When waiting with sleep, we don't want to wait too long, but we never know when a pipeline
        # would finish evaluating.
        last_get_successful = True
        while True:
            try:
                if not last_get_successful:
                    time.sleep(0.1)  # seconds

                identifier, fitness = self._output_queue.get(block=False)
                return identifier, fitness

            except queue.Empty:
                last_get_successful = False
                continue

    def get_next_result(self):
        """ Get the result of an evaluation that was queued by calling `queue_evaluation`. This function is blocking.

        This function raises a ValueError if it is called when there is no job queued with `queue_evaluation`.
        """
        if len(self._job_map) <= 0:
            raise ValueError("You have to queue an evaluation for each time you call this function since last cancel.")

        if self._n_jobs > 1:
            identifier, output = self._get_next_from_daemons()
        else:
            # For n_jobs = 1, we do not want to spawn a separate process. Mimic behaviour.
            identifier, input_ = self._input_queue.get()
            output = self._func(input_)

        input_ = self._job_map.pop(identifier)
        return identifier, output, input_
