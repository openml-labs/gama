import logging
import multiprocessing as mp
import queue
import random
import time

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


class FunctionDispatcher(object):

    def __init__(self, n_jobs, evaluate_fn, toolbox):
        if n_jobs <= 0:
            raise ValueError("n_jobs must be at least 1.")

        mp_manager = mp.Manager()
        self._input_queue = mp_manager.Queue()
        self._output_queue = mp_manager.Queue()
        self._n_jobs = n_jobs
        self._evaluate_fn = evaluate_fn
        self._toolbox = toolbox

        self._outstanding_job_counter = 0
        self._subscribers = []
        self._job_map = {}
        self._child_processes = []

    def start(self):
        log.info('Setting up additional processes for parallel asynchronous evaluations.')
        for _ in range(self._n_jobs):
            p = mp.Process(target=evaluator_daemon,
                           args=(self._input_queue, self._output_queue, self._evaluate_fn))
            p.daemon = True
            self._child_processes.append(p)
            p.start()

    def queue_evaluation(self, individual):
        comp_ind = self._toolbox.compile(individual)
        self._job_map[str(comp_ind)] = individual
        self._input_queue.put((str(comp_ind), comp_ind))
        self._outstanding_job_counter += 1

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

                comp_ind_str, fitness = self._output_queue.get(block=False)
                return self._job_map[comp_ind_str], fitness

            except queue.Empty:
                last_get_successful = False
                continue

    def get_next_result(self):
        if self._outstanding_job_counter <= 0:
            raise ValueError("You have to queue an evaluation for each time you call this function.")
        else:
            self._outstanding_job_counter -= 1

        if self._n_jobs > 1:
            return self._get_next_from_daemons()
        else:
            # For n_jobs = 1, we do not want to spawn a separate process. Mimic behaviour.
            identifier, input_ = self._input_queue.get()
            output = self._evaluate_fn(input_)
            return self._job_map[identifier], output

    def cancel_all_evaluations(self):
        jobs_cancelled = 0
        while True:
            try:
                self._input_queue.get(block=False)
                jobs_cancelled += 1
            except queue.Empty:
                self._outstanding_job_counter -= jobs_cancelled
                if self._outstanding_job_counter > 0:
                    log.warning("Cancelled {} queued jobs, but {} are already being processed."
                                .format(jobs_cancelled, self._outstanding_job_counter))
                else:
                    log.info("Cancelled {} jobs, no more jobs being processed.".format(jobs_cancelled))
                break

    def shut_down(self):
        log.info('Shutting down additional processes used for parallel asynchronous evaluations.')
        for process in self._child_processes:
            process.terminate()
