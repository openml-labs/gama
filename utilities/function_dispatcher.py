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
    """ """

    def __init__(self, n_jobs, evaluate_fn, toolbox):
        if n_jobs <= 0:
            raise ValueError("n_jobs must be at least 1.")

        mp_manager = mp.Manager()
        self._input_queue = mp_manager.Queue()
        self._output_queue = mp_manager.Queue()
        self._n_jobs = n_jobs
        self._evaluate_fn = evaluate_fn
        self._toolbox = toolbox

        self._job_map = {}
        self._child_processes = []

    def start(self):
        """ Start child processes. """
        log.info('Starting {} child processes.'.format(self._n_jobs))
        self._job_map = {}
        for _ in range(self._n_jobs):
            p = mp.Process(target=evaluator_daemon,
                           args=(self._input_queue, self._output_queue, self._evaluate_fn))
            p.daemon = True
            self._child_processes.append(p)
            p.start()

    def stop(self):
        """ Dequeue all outstandig jobs, discard saved results and terminate child processes. """
        log.info('Terminating {} child processes.'.format(len(self._child_processes)))
        for process in self._child_processes:
            process.terminate()

        nr_cancelled = clear_queue(self._input_queue)
        nr_discarded = clear_queue(self._output_queue)
        log.debug("Cancelled {} outstanding jobs. Discarded {} results. Terminated {} currently executing jobs."
                  .format(nr_cancelled, nr_discarded, len(self._job_map) - nr_cancelled - nr_discarded))

    def restart(self):
        """ This is equivalent to calling `stop` then `start`."""
        self.stop()
        self.start()

    def queue_evaluation(self, individual):
        """ Queue an individual to be evaluated by a child process according to `evaluate_fn` passed to __init__. """
        comp_ind = self._toolbox.compile(individual)
        identifier = uuid.uuid4()
        self._job_map[identifier] = individual
        self._input_queue.put((identifier, comp_ind))

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
            output = self._evaluate_fn(input_)

        individual = self._job_map.pop(identifier)
        return individual, output

