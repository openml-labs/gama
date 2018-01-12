"""
Created on Thu Jan 11 11:45:30 2018

@author: Pieter Gijsbers
"""
from collections import defaultdict

import deap
from deap import base, creator, tools
from deap import gp

import numpy as np

import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

from modifiedDEAP import generate
from StackingTransformer import make_stacking_transformer
from configuration import configuration

log_level = 2
def log_message(message, level=5):
    if level <= log_level:
        print(message)

class Data(np.ndarray):
    pass 

class Predictions(np.ndarray):
    pass

def compile_individual(ind, pset):
    components = []
    name_counter = defaultdict(int)
    while(len(ind) > 0):
        log_message('compiling ' + str(ind), level = 4)
        prim, remainder = ind[0], ind[1:]
        if isinstance(prim, gp.Terminal):
            if len(remainder)>0:
                print([el.name for el in remainder])
                raise Exception
            break
        # See if all terminals have a value provided (except Data Terminal)
        required_provided = list(zip(reversed(prim.args[1:]), reversed(remainder)))
        if all(r==p.ret for (r,p) in required_provided):            
            log_message('compiling ' + str([p.name for r, p in required_provided]), level = 5)
            # If so, instantiate the pipeline component with given arguments.
            def extract_arg_name(terminal_name):
                equal_idx = terminal_name.rfind('=')
                return terminal_name[terminal_name.rfind('.',0,equal_idx)+1:equal_idx]
            args = {
                    extract_arg_name(p.name): pset.context[p.name]
                    for r, p in required_provided
                    }
            class_ = pset.context[prim.name]
            # All pipeline components must have a unique name
            name = prim.name + str(name_counter[prim.name])
            name_counter[prim.name] += 1
            components.append((name, class_(**args)))
            ind = ind[1:-len(args)]
        else:
            raise TypeError("Type is wrong or missing.")
            
    return Pipeline(list(reversed(components)))

def pset_from_config(config):
    pset = gp.PrimitiveSetTyped("pipeline",in_types=[Data], ret_type=Predictions)
    pset.renameArguments(ARG0="data")
    
    for path, hyperparameters in config.items():
        if '.' in path:
            module_path, class_ = path.rsplit('.', maxsplit=1)
            exec(f"from {module_path} import {class_}")
        else:
            class_ = path
            exec(f"import {class_}")
        
        hyperparameter_types = []
        for name, values in hyperparameters.items():
            # We construct a new type for each hyperparameter, so we can specify
            # it as terminal type, making sure it matches with expected
            # input of the operators. Moreover it automatically makes sure that
            # crossover only happens between same hyperparameters.
            hyperparameter_type = type(f"{class_}{name}",(object,), {})
            hyperparameter_types.append(hyperparameter_type)
            for value in values:
                # Escape string values with quotes otherwise they are variables
                value_str = f"'{value}'" if isinstance(value, str) else f"{value}"
                hyperparameter_str = f"{class_}.{name}={value_str}"            
                pset.addTerminal(value, hyperparameter_type, hyperparameter_str)
                
        class_type = eval(class_)
        pset.addPrimitive(class_type, [Data, *hyperparameter_types], Predictions)
        
        stacking_class = make_stacking_transformer(class_type)
        pset.addPrimitive(stacking_class, [Data, *hyperparameter_types], Data, name = class_ + stacking_class.__name__)
    
    return pset

def gen_grow_safe(pset, min_, max_, type_=None):
    """Generate an expression where each leaf might have a different depth between min_ and max_.
    Condition specifies if desired return type is not Data or Predictions, that a terminal be used.
    """

    def condition(height, depth, type_):
        """Stop when the depth is equal to height or when a node should be a terminal."""
        return (type_ not in [Data, Predictions]) or (depth == height)

    return generate(pset, min_, max_, condition, type_)

def mut_replace_terminal(ind, pset):
    eligible = [i for i,el in enumerate(ind) if (issubclass(type(el), gp.Terminal) and len(pset.terminals[el.ret])>1)]
    to_change = np.random.choice(eligible)
    ind[to_change] = np.random.choice(pset.terminals[ind[to_change].ret])
    return ind, 

def evaluate_pipeline(ind, X, y, cv = 5):
    log_message('evaluating '+str(ind), level=3)
    pl = toolbox.compile(ind)
    return (np.mean(cross_val_score(pl, X, y, cv = cv)),)

pset = pset_from_config(configuration)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

print('Registering toolbox')
toolbox = base.Toolbox()
toolbox.register("expr", gen_grow_safe, pset=pset, min_=2, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", compile_individual, pset=pset)

toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", mut_replace_terminal, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)

print('Loading data')
iris = sklearn.datasets.load_iris()
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
pop, log = deap.algorithms.eaSimple(pop, toolbox, cxpb=0.2, mutpb=0.8, ngen=50, verbose=True, stats=mstats)