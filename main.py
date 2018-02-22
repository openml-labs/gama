"""
Created on Thu Jan 11 11:45:30 2018
@author: Pieter Gijsbers
"""
import deap
from deap import base, creator, tools
from deap import gp

import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score

from modifiedDEAP import gen_grow_safe
from configuration import configuration_with_preprocessing, tpot_config, new_config
from automl_gp import pset_from_config, compile_individual, pset_from_config_new

log_level = -3
def log_message(message, level=5):
    if level <= log_level:
        print(message)
        
def mut_replace_terminal(ind, pset):
    ind = toolbox.clone(ind)
    eligible = [i for i,el in enumerate(ind) if (issubclass(type(el), gp.Terminal) and len(pset.terminals[el.ret])>1)]
    #els = [el for i,el in enumerate(ind) if (issubclass(type(el), gp.Terminal) and len(pset.terminals[el.ret])>1)]
    if eligible == []:
        log_message('No way to mutate '+str(ind)+' was found.', level=4)
        return ind,
    
    to_change = np.random.choice(eligible)    
    ind[to_change] = np.random.choice(pset.terminals[ind[to_change].ret])
    return ind, 

def evaluate_pipeline(ind, X, y, cv = 5):
    log_message('evaluating '+str(ind), level=3)
    pl = toolbox.compile(ind)
    #if pl is None:
        # Failed to compile due to invalid hyperparameter configuration
    #    return (-float("inf"),)
    try:
        fitness = (np.mean(cross_val_score(pl, X, y, cv = cv)),)
    except:
        fitness = (-float("inf"),)
        
    return fitness

pset = pset_from_config(tpot_config)
pset, parameter_checks = pset_from_config_new(new_config)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

print('Registering toolbox')
toolbox = base.Toolbox()
def generate_valid(pset, min_, max_):
    for _ in range(50):
        ind = gen_grow_safe(pset, min_, max_)
        pl = toolbox.compile(ind)
        if pl is not None:
            return ind
    raise Exception('Failed')
        
toolbox.register("expr", generate_valid, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", compile_individual, pset=pset, parameter_checks=parameter_checks)

toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", mut_replace_terminal, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)

print('Loading data')
from sklearn.datasets import load_iris
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, stratify=iris.target, shuffle=True, random_state=42)
toolbox.register("evaluate", evaluate_pipeline, X=X_train, y=y_train)


print('Creating individual')
expr = toolbox.individual()
str(expr)

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)

pop = toolbox.population(n=100)
from deap.algorithms import eaSimple
pop, log = eaSimple(pop, toolbox, cxpb=0.2, mutpb=0.8, ngen=25, verbose=True, stats=mstats)
