from functools import partial
import multiprocessing as mp
#from pathos.helpers.mp.process import Process
#import pathos.multiprocessing as pathosmp

import numpy as np
from sklearn.model_selection import cross_val_score

def return_self(x):
    return x

def evaluator_daemon(input_queue, output_queue, fn):
    while True:
        identifier, input_ = input_queue.get()
        output = fn(input_)
        output_queue.put((identifier, output))

def evaluate_individual2(individual, X, y, cv = 5):
    """ Evaluates a pipeline used k-Fold CV. """
    pl = individual
    
    try:
        fitness_values = (np.mean(cross_val_score(pl, X, y, cv = cv)),)
    except:
        fitness_values = (-float("inf"),)
    
    return fitness_values

def evaluate_individual(individual, compile_fn, X, y, cv = 5):
    """ Evaluates a pipeline used k-Fold CV. """
    pl = compile_fn(individual)
    
    try:
        fitness_values = (np.mean(cross_val_score(pl, X, y, cv = cv)),)
    except:
        fitness_values = (-float("inf"),)
    
    return individual, fitness_values

# n_processes
# eaSimple(pop, self._toolbox, cxpb=0.2, mutpb=0.8, ngen=self._n_generations, verbose=True, halloffame=HallOfFame('log.txt'))
def async_ea(pop, toolbox, X, y, cxpb=0.2, mutpb=0.8, n_evals=300, verbose=True, halloffame=None):
    mp_manager = mp.Manager()
    input_queue = mp_manager.Queue()
    output_queue = mp_manager.Queue()
    
    n_processes = 7
    P = len(pop)
    running_pop = []
    
    comp_ind_map = {}
    
    for _ in range(n_processes):
        #p = mp.Process(target = evaluator_daemon, args = (input_queue, output_queue, toolbox.evaluate,))
        p = mp.Process(target = evaluator_daemon, args = (input_queue, output_queue, partial(evaluate_individual2, X=X, y=y),))
        p.daemon = True
        p.start()
        
    for ind in pop:
        comp_ind = toolbox.compile(ind)
        comp_ind_map[str(comp_ind)] = ind
        input_queue.put((str(comp_ind), comp_ind))
    
    for i in range(n_evals):
        print(i)
        comp_ind_str, fitness = output_queue.get()
        individual = comp_ind_map[comp_ind_str]
        individual.fitness.values = fitness
        
        # Add to population
        running_pop.append(individual)
        
        # Shrink population if needed        
        if len(running_pop) > P:
            running_pop.remove(min(running_pop, key = lambda i: i.fitness.values[0]))
        
        # Create new individual if needed - or do we just always queue?
        ind = toolbox.individual()
        comp_ind = toolbox.compile(ind)
        comp_ind_map[str(comp_ind)] = ind
        input_queue.put((str(comp_ind), comp_ind))
    
    return running_pop, None
        
    
    
    