import multiprocessing as mp
import queue
import random

import numpy as np


def evaluator_daemon(input_queue, output_queue, fn, shutdown):
    random.seed(0)
    np.random.seed(0)

    shutdown_message = 'Helper process stopping normally.'
    try:
        while not shutdown.value:
            identifier, input_ = input_queue.get()
            output = fn(input_)
            if not shutdown.value:
                output_queue.put((identifier, output))
    except KeyboardInterrupt:
        shutdown_message = 'Helper process stopping due to keyboard interrupt.'
    except BrokenPipeError:
        shutdown_message = 'Helper process stopping due to a broken pipe.'
        
    print(shutdown_message)


def async_ea(n_threads=1, *args, **kwargs):
    if n_threads == 1:
        return async_ea_sequential(*args, **kwargs)
    else:
        return async_ea_parallel(n_threads, *args, **kwargs)


def async_ea_sequential(pop, toolbox, X, y, cxpb=0.2, mutpb=0.8, n_evals=300, verbose=True, halloffame=None):
    print('starting sequential')
    max_pop_size = len(pop)
    running_pop = []

    for i in range(n_evals):
        if i < len(pop):
            ind = pop[i]
        else:
            ind = toolbox.individual()

        comp_ind = toolbox.compile(ind)
        fitness = toolbox.evaluate(comp_ind)
        ind.fitness.values = fitness

        # Add to population
        running_pop.append(ind)
        halloffame.update([ind])

        # Shrink population if needed
        if len(running_pop) > max_pop_size:
            running_pop.remove(min(running_pop, key=lambda x: x.fitness.values[0]))
    return running_pop, None, None


def async_ea_parallel(n_threads, pop, toolbox, X, y, cxpb=0.2, mutpb=0.8, n_evals=300, verbose=True, halloffame=None):
    mp_manager = mp.Manager()
    input_queue = mp_manager.Queue()
    output_queue = mp_manager.Queue()
    shutdown = mp_manager.Value('shutdown', False)

    n_processes = n_threads
    max_pop_size = len(pop)
    running_pop = []
    
    comp_ind_map = {}
    
    for _ in range(n_processes):
        p = mp.Process(target=evaluator_daemon, args=(input_queue, output_queue, toolbox.evaluate, shutdown,))
        p.daemon = True
        p.start()
        
    try:
        print('Starting EA')
        for ind in pop:
            comp_ind = toolbox.compile(ind)
            comp_ind_map[str(comp_ind)] = ind
            input_queue.put((str(comp_ind), comp_ind))
        
        for i in range(n_evals):        
            received_evaluation = False
            while not received_evaluation:
                try:
                    comp_ind_str, fitness = output_queue.get(timeout=100)
                    received_evaluation = True
                except queue.Empty:
                    continue
                
            individual = comp_ind_map[comp_ind_str]
            individual.fitness.values = fitness
            
            # Add to population
            running_pop.append(individual)
            halloffame.update([individual])
            
            # Shrink population if needed        
            if len(running_pop) > max_pop_size:
                running_pop.remove(min(running_pop, key=lambda x: x.fitness.values[0]))
            
            # Create new individual if needed - or do we just always queue?
            ind = toolbox.individual()
            comp_ind = toolbox.compile(ind)
            comp_ind_map[str(comp_ind)] = ind
            input_queue.put((str(comp_ind), comp_ind))
        
        shutdown.value = True
        
    except KeyboardInterrupt:
        print("Shutting down EA due to KeyboardInterrupt.")
    except:
        shutdown.value = True
        
    return running_pop, None, shutdown
