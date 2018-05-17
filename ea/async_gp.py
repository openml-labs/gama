import multiprocessing as mp
import queue
import random
import logging
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


def async_ea(self, n_threads=1, *args, **kwargs):
    if n_threads == 1:
        return async_ea_sequential(self, *args, **kwargs)
    else:
        return async_ea_parallel(self, n_threads, *args, **kwargs)


def async_ea_sequential(self, pop, toolbox, X, y, cxpb=0.2, mutpb=0.8, n_evals=300, verbose=True, evaluation_callback=None):
    log.info('Starting sequential asynchronous algorithm.')
    max_pop_size = len(pop)
    running_pop = []

    for i in range(n_evals):
        if i < len(pop):
            ind = pop[i]
        else:
            for _ in range(50):
                ind, = offspring_mate_and_mutate(running_pop, toolbox, cxpb, mutpb, n=1)[0]
                if str(ind) not in self._evaluated_individuals:
                    break

        comp_ind = toolbox.compile(ind)
        fitness = toolbox.evaluate(comp_ind)
        if self._objectives[1] == 'size':
            fitness = (fitness[0], automl_gp.pipeline_length(ind))
        ind.fitness.values = fitness
        self._evaluated_individuals[str(ind)] = fitness
        if evaluation_callback:
            evaluation_callback(ind)

        # Add to population
        running_pop.append(ind)

        # Shrink population if needed
        if len(running_pop) > max_pop_size:
                ind_to_replace = select_to_replace(running_pop, len(self._objectives))
                running_pop.remove(ind_to_replace)
    return running_pop, None


def async_ea_parallel(self, n_threads, pop, toolbox, X, y, cxpb=0.2, mutpb=0.8, n_evals=300, evaluation_callback=None, verbose=True):
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
            while not received_evaluation:
                try:
                    # If we just used the blocking queue.get, then KeyboardInterrupts/Timeout would not work.
                    comp_ind_str, fitness = output_queue.get(timeout=100)
                    received_evaluation = True
                except queue.Empty:
                    continue
            
            mp_logger.flush_to_log(log)
            log.debug('Evaluated {} individuals.'.format(i))
                
            individual = comp_ind_map[comp_ind_str]
            if self._objectives[1] == 'size':
                fitness = (fitness[0], automl_gp.pipeline_length(ind))
            individual.fitness.values = fitness
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


def offspring_mate_and_mutate(pop, toolbox, cxpb, mutpb, n, always_return_list=False):
    """ Creates n new individuals based on the population. Can apply both crossover and mutation. """
    offspring = []
    for _ in range(n):
        ind1, ind2 = np.random.choice(range(len(pop)), size=2, replace=False)
        ind1, ind2 = toolbox.clone(pop[ind1]), toolbox.clone(pop[ind2])
        if np.random.random() < cxpb:
            ind1, ind2 = toolbox.mate(ind1, ind2)
        if np.random.random() < mutpb:
            ind1, = toolbox.mutate(ind1)
        offspring.append((ind1,))
    return offspring


def select_to_replace(pop, nr_objectives):
    """ Selects individual in population to replace. """
    if nr_objectives == 1:
        return min(pop, key=lambda x: x.fitness.values[0])
    elif nr_objectives == 2:
        return tools.selNSGA2(pop, k=len(pop))[-1]
