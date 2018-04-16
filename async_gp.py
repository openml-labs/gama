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
    except (BrokenPipeError, EOFError):
        shutdown_message = 'Helper process stopping due to a broken pipe or EOF.'
        
    print(shutdown_message)


def async_ea(self, n_threads=1, *args, **kwargs):
    if n_threads == 1:
        return async_ea_sequential(self, *args, **kwargs)
    else:
        return async_ea_parallel(n_threads, *args, **kwargs)


def offspring_mate_and_mutate(pop, toolbox, cxpb, mutpb, n, always_return_list=False):
    """ Creates n new individuals based on the population. Can apply both crossover and mutation. """
    offspring = []
    for _ in range(n):
        try:
            ind1, ind2 = np.random.choice(range(len(pop)), size=2, replace=False)
            ind1, ind2 = pop[ind1], pop[ind2]
        except Exception as e:
            z =1
        ind1, ind2 = toolbox.clone(ind1), toolbox.clone(ind2)
        if np.random.random() < cxpb:
            ind1, ind2 = toolbox.mate(ind1, ind2)
        if np.random.random() < mutpb:
            ind1, = toolbox.mutate(ind1)
        offspring.append((ind1,))
    return offspring

def async_ea_sequential(self, pop, toolbox, X, y, cxpb=0.2, mutpb=0.8, n_evals=300, verbose=True, halloffame=None):
    print('starting sequential')
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
        ind.fitness.values = fitness
        self._evaluated_individuals[str(ind)] = fitness

        # Add to population
        running_pop.append(ind)
        halloffame.update([ind])

        # Shrink population if needed
        if len(running_pop) > max_pop_size:
            running_pop.remove(min(running_pop, key=lambda x: x.fitness.values[0]))
    return running_pop, None


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
                    # If we just used the blocking queue.get, then KeyboardInterrupts/Timeout would not work.
                    comp_ind_str, fitness = output_queue.get(timeout=100)
                    received_evaluation = True
                except queue.Empty:
                    continue
            print(i)
                
            individual = comp_ind_map[comp_ind_str]
            individual.fitness.values = fitness
            
            # Add to population
            running_pop.append(individual)
            halloffame.update([individual])
            
            # Shrink population if needed        
            if len(running_pop) > max_pop_size:
                running_pop.remove(min(running_pop, key=lambda x: x.fitness.values[0]))
            
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
        print("Shutting down EA due to KeyboardInterrupt.")
        # No need to communicate to processes since they also handle the KeyboardInterrupt directly.
    except:
        # Even in the event of an error we want the helper processes to shut down.
        shutdown.value = True
        raise
        
    return running_pop, shutdown
