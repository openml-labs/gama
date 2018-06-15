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


def evaluator_daemon(input_queue, output_queue, fn, shutdown, seed=0):
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
        
    print(shutdown_message)


#def async_ea(self, n_threads=1, *args, **kwargs):
#    if n_threads == 1:
#        return async_ea_sequential(self, *args, **kwargs)
#    else:
#        return async_ea_parallel(self, n_threads, *args, **kwargs)


def async_ea_sequential(objectives, pop, toolbox, X, y, cxpb=0.2, mutpb=0.8, n_evals=300, verbose=True, evaluation_callback=None):
    log.info('Starting sequential asynchronous algorithm.')
    max_pop_size = len(pop)
    running_pop = []

    for i in range(n_evals):
        log.debug(i)
        if i < len(pop):
            ind = pop[i]
        else:
            for _ in range(50):
                ind, = offspring_mate_and_mutate(running_pop, toolbox, cxpb, mutpb, n=1)[0]
                if str(ind) not in self._evaluated_individuals:
                    break

        comp_ind = toolbox.compile(ind)
        if comp_ind is None:
            log.debug('Invalid individual generated, assigning worst fitness.')
            fitness = (-float('inf'),)
        else:
            score, eval_time = toolbox.evaluate(comp_ind)

        if objectives[1] == 'size':
            fitness = (score, automl_gp.pipeline_length(ind))
        ind.fitness.values = fitness
        ind.fitness.time = eval_time
        if evaluation_callback:
            evaluation_callback(ind)

        # Add to population
        running_pop.append(ind)

        # Shrink population if needed
        if len(running_pop) > max_pop_size:
                ind_to_replace = select_to_replace(running_pop, len(self._objectives))
                running_pop.remove(ind_to_replace)
    return running_pop, None

class EvaluationDispatcher(object):

    def __init__(self, n_jobs, evaluate_fn, toolbox):
        mp_manager = mp.Manager()
        self._input_queue = mp_manager.Queue()
        self._output_queue = mp_manager.Queue()
        self._shutdown = mp_manager.Value('shutdown', False)
        self._n_jobs = n_jobs
        self._evaluate_fn = evaluate_fn
        self._toolbox = toolbox

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

    def get_next_result(self):
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

    def cancel_all_evaluations(self):
        while True:
            try:
                self._input_queue.get(block=False)
            except queue.Empty:
                break

    def shut_down(self):
        self._shutdown = True


def async_ea(objectives, population, toolbox, evaluation_callback=None, n_evaluations=10000, n_threads=1):
    logger = MultiprocessingLogger()
    evaluation_dispatcher = EvaluationDispatcher(n_threads, partial(toolbox.evaluate, logger=logger), toolbox)
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


def async_ea_parallel(objectives, n_threads, pop, toolbox, cxpb=0.2, mutpb=0.8, n_evals=300, evaluation_callback=None, verbose=True):
    log.info('Setting up additional processes for parallel asynchronous algorithm.')
    mp_manager = mp.Manager()
    input_queue = mp_manager.Queue()
    output_queue = mp_manager.Queue()
    mp_logger = MultiprocessingLogger()
    shutdown = mp_manager.Value('shutdown', False)

    n_processes = n_threads
    max_pop_size = len(pop)
    running_pop = []
    
    comp_ind_map = {}
    evaluate_fn = partial(toolbox.evaluate, logger=mp_logger)
    for _ in range(n_processes):
        p = mp.Process(target=evaluator_daemon, args=(input_queue, output_queue, evaluate_fn, shutdown))
        p.daemon = True
        p.start()

    log.info('Processes set up. Commencing asynchronous algorithm.')
    try:
        for ind in pop:
            comp_ind = toolbox.compile(ind)
            comp_ind_map[str(comp_ind)] = ind
            input_queue.put((str(comp_ind), comp_ind))

        for i in range(n_evals):
            received_evaluation = False
            last_get_successful = True
            while not received_evaluation:
                try:
                    # If we just used the blocking queue.get, then KeyboardInterrupts/Timeout would not work.
                    # Previously, specifying a timeout worked, but for some reason that seems no longer the case.
                    # Using timeout prevents the stopit.Timeout exception from being received.
                    # When waiting with sleep, we don't want to wait too long, but we never know when a pipeline
                    # would finish evaluating.
                    if not last_get_successful:
                        time.sleep(0.1)  # seconds
                    comp_ind_str, fitness = output_queue.get(block=False)
                    received_evaluation = True
                except queue.Empty:
                    last_get_successful = False
                    continue

            mp_logger.flush_to_log(log)
            #log.debug('Evaluated {} individuals.'.format(i))

            individual = comp_ind_map[comp_ind_str]
            score, evaluation_time, length = fitness
            if len(objectives) == 1:
                individual.fitness.values = (score,)
            elif objectives[1] == 'time':
                individual.fitness.values = (score, evaluation_time)
            elif objectives[1] == 'size':
                individual.fitness.values = (score, length)

            individual.fitness.time = evaluation_time

            if evaluation_callback:
                evaluation_callback(individual)

            # Add to population
            running_pop.append(individual)

            # Shrink population if needed
            if len(running_pop) > max_pop_size:
                ind_to_replace = select_to_replace(running_pop, len(self._objectives))
                running_pop.remove(ind_to_replace)

            # Create new individual if needed - or do we just always queue?
            if len(running_pop) < 2:
                ind = toolbox.individual()
            else:
                ind, = offspring_mate_and_mutate(running_pop, toolbox, cxpb, mutpb, n=1)[0]
            comp_ind = toolbox.compile(ind)
            comp_ind_map[str(comp_ind)] = ind
            input_queue.put((str(comp_ind), comp_ind))

        shutdown.value = True
        
    except KeyboardInterrupt:
        log.info('Shutting down EA due to KeyboardInterrupt.')
        # No need to communicate to processes since they also handle the KeyboardInterrupt directly.
    except Exception:
        log.error('Unexpected exception in asynchronous parallel algorithm.', exc_info=True)
        # Even in the event of an error we want the helper processes to shut down.
        shutdown.value = True
        raise

    return running_pop, shutdown

