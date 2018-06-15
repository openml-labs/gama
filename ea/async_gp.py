import multiprocessing as mp
import queue
import random
import logging
import time
from functools import partial

import numpy as np
from deap import tools

from . import automl_gp
from ..utilities.mp_logger import MultiprocessingLogger

log = logging.getLogger(__name__)


def evaluator_daemon(input_queue, output_queue, fn, shutdown, seed=0, print_exit_message=False):
    random.seed(seed)
    np.random.seed(seed)

    shutdown_message = 'Helper process stopping normally.'
    try:
        while not shutdown.value:
            identifier, input_ = input_queue.get()
            output = fn(input_)
            if not shutdown.value:
                output_queue.put((identifier, output))
    except KeyboardInterrupt:
        shutdown_message = 'Helper process stopping due to keyboard interrupt.'
    except (BrokenPipeError, EOFError):
        shutdown_message = 'Helper process stopping due to a broken pipe or EOF.'

    if print_exit_message:
        print(shutdown_message)


class EvaluationDispatcher(object):

    def __init__(self, n_jobs, evaluate_fn, toolbox):
        if n_jobs <= 0:
            raise ValueError("n_jobs must be at least 1.")

        mp_manager = mp.Manager()
        self._input_queue = mp_manager.Queue()
        self._output_queue = mp_manager.Queue()
        self._shutdown = mp_manager.Value('shutdown', False)
        self._n_jobs = n_jobs
        self._evaluate_fn = evaluate_fn
        self._toolbox = toolbox

        self._outstanding_job_counter = 0
        self._subscribers = []
        self._job_map = {}

    def start(self):
        log.info('Setting up additional processes for parallel asynchronous evaluations.')
        for _ in range(self._n_jobs):
            p = mp.Process(target=evaluator_daemon,
                           args=(self._input_queue, self._output_queue, self._evaluate_fn, self._shutdown))
            p.daemon = True
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
        while True:
            try:
                self._input_queue.get(block=False)
            except queue.Empty:
                break

    def shut_down(self):
        self._shutdown = True


def async_ea(objectives, population, toolbox, evaluation_callback=None, n_evaluations=10000, n_jobs=1):
    logger = MultiprocessingLogger()
    evaluation_dispatcher = EvaluationDispatcher(n_jobs, partial(toolbox.evaluate, logger=logger), toolbox)
    try:
        evaluation_dispatcher.start()
        # while improvements
        log.info('Starting ')
        for ind in population:
            evaluation_dispatcher.queue_evaluation(ind)
        # run ea
        max_population_size = len(population)
        current_population = []
        for _ in range(n_evaluations):
            individual, output = evaluation_dispatcher.get_next_result()
            score, evaluation_time, length = output
            if len(objectives) == 1:
                individual.fitness.values = (score,)
            elif objectives[1] == 'time':
                individual.fitness.values = (score, evaluation_time)
            elif objectives[1] == 'size':
                individual.fitness.values = (score, length)
            individual.fitness.time = evaluation_time

            logger.flush_to_log(log)
            if evaluation_callback:
                evaluation_callback(individual)

            # TODO: Measure improvements, possibly restart.

            current_population.append(individual)
            if len(current_population) > max_population_size:
                current_population = toolbox.eliminate(current_population, 1)

            if len(current_population) > 1:
                new_individual = toolbox.create(current_population, 1)[0]
                evaluation_dispatcher.queue_evaluation(new_individual)

    except KeyboardInterrupt:
        log.info('Shutting down EA due to KeyboardInterrupt.')
        # No need to communicate to processes since they also handle the KeyboardInterrupt directly.
    except Exception:
        log.error('Unexpected exception in asynchronous parallel algorithm.', exc_info=True)
        # Even in the event of an error we want the helper processes to shut down.
        evaluation_dispatcher.shut_down()
        raise

    evaluation_dispatcher.shut_down()
    return current_population, evaluation_dispatcher