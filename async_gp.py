import multiprocessing as mp
import threading

#from pathos.helpers.mp.process import Process
#import pathos.multiprocessing as pathosmp

def evaluator_daemon(input_queue, output_queue, fn):    
    while True:
        identifier, input_ = input_queue.get()
        output = fn(input_)
        output_queue.put((identifier, output))

def async_ea2(pop, toolbox, X, y, cxpb=0.2, mutpb=0.8, n_evals=300, verbose=True, halloffame=None):
    P = len(pop)
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
        if len(running_pop) > P:
            running_pop.remove(min(running_pop, key = lambda i: i.fitness.values[0]))
        
    return running_pop, None

def async_ea(pop, toolbox, X, y, cxpb=0.2, mutpb=0.8, n_evals=300, verbose=True, halloffame=None):
    mp_manager = mp.Manager()
    input_queue = mp_manager.Queue()
    output_queue = mp_manager.Queue()
    
    n_processes = 7
    P = len(pop)
    running_pop = []
    
    comp_ind_map = {}
    
    for _ in range(n_processes):
        p = mp.Process(target = evaluator_daemon, args = (input_queue, output_queue, toolbox.evaluate,))
        p.daemon = True
        p.start()
        
    for ind in pop:
        comp_ind = toolbox.compile(ind)
        comp_ind_map[str(comp_ind)] = ind
        input_queue.put((str(comp_ind), comp_ind))
    
    for i in range(n_evals):        
        #print(i)
        comp_ind_str, fitness = output_queue.get()
        individual = comp_ind_map[comp_ind_str]
        individual.fitness.values = fitness
        
        # Add to population
        running_pop.append(individual)
        halloffame.update([individual])
        
        # Shrink population if needed        
        if len(running_pop) > P:
            running_pop.remove(min(running_pop, key = lambda i: i.fitness.values[0]))
        
        # Create new individual if needed - or do we just always queue?
        ind = toolbox.individual()
        comp_ind = toolbox.compile(ind)
        comp_ind_map[str(comp_ind)] = ind
        input_queue.put((str(comp_ind), comp_ind))
    
    print('done')
    return running_pop, None
        
    
    
    