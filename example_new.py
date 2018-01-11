"""
Created on Thu Jan 11 11:45:30 2018

@author: s105307
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from deap import base, creator, tools
from deap import gp
from deap.gp import PrimitiveSetTyped, PrimitiveTree, genFull
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ml_pipeline import MLPipeline

d ={
    'sklearn.tree.DecisionTreeClassifier': {
        'criterion': ["gini", "entropy"],
        #'max_depth': range(1, 11),
        #'min_samples_split': range(2, 21),
        #'min_samples_leaf': range(1, 21)
    },
    
    'sklearn.naive_bayes.BernoulliNB': {
            #'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
            'fit_prior': [True, False]
    }
}
    
class Data(np.ndarray):
    pass 

class Predictions(np.ndarray):
    pass


def compile_ind2(ind, pset):
    prim = ind[0]
    stack = [prim.args]
    
    while len(stack) > 0:
        ret_types = stack.pop()
        

def compile_ind_rec(ind, pset):
    prim, rest = ind[0], ind[1:]
    term_strings = []
    for arg in prim.args:
        if not rest[1].ret_type == arg:
            raise TypeError
        if isinstance(rest[1], gp.Primitive):
            comp_str = compile_ind_rec(rest, pset)
        elif isinstance(rest[1], gp.Primitive):
            comp_str = rest[1].name.split('.')[-1]
        else:
            raise TypeError
        term_strings.append(comp_str)
    return "{}({})".format(prim.name, ','.join(term_strings))

def compile_ind(ind, pset):
    # Recursively stringify pipeline
    primitive, remainder = ind[0], ind[1:]
    stack, remainder = [ind[0]], ind[1:]
     
    while len(stack) > 0:
        if isinstance(remainder[0], gp.Primitive):
            z=3   
        if isinstance(remainder[0], gp.Terminal):
            # Terminal are of shape "{class_}.{name}__{value}"
            remainder.name = remainder[0].value
            
    pass
    

def evalPipeline(ind, X_tr, X_te, y_tr, y_te):
    pipe = MLPipeline(ind)
    pipe.fit(X_tr,y_tr)
    return pipe.score(X_te, y_te),

# in = X out = y
# Define Primitives
pset = PrimitiveSetTyped("pipeline",in_types=[Data], ret_type=Predictions)
pset.renameArguments(ARG0="data")
#pset.addPrimitive(ProcessBernoulliNB, [Xtrain, Ytrain, Xtest, float, bool], Output)
for path, hyperparameters in d.items():
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
            hyperparameter_str = f"{class_}.{name}={value}"
            pset.addTerminal(value, hyperparameter_type, hyperparameter_str)
            
    pset.addPrimitive(eval(class_), [Data, *hyperparameter_types], Predictions)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

toolbox = base.Toolbox()
toolbox.register("expr", genFull, pset=pset, min_=1, max_=1)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("mate", gp.cxOnePoint)

def mut_replace_terminal(ind,pset):
    eligible = [i for i,el in enumerate(ind) if (issubclass(type(el), gp.Terminal) and len(pset.terminals[el.ret])>1)]
    to_change = np.random.choice(eligible)
    ind[to_change] = np.random.choice(pset.terminals[ind[to_change].ret])
    return ind, 

toolbox.register("mutate", mut_replace_terminal, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)

from sklearn.datasets import load_iris
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, stratify=iris.target, shuffle=True, random_state=42)
toolbox.register("evaluate", evalPipeline, X_tr=X_train, X_te=X_test, y_tr=y_train, y_te=y_test)

expr = toolbox.individual()
str(expr)

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)



pop = toolbox.population(n=5)
from deap.algorithms import eaSimple
pop, log = eaSimple(pop, toolbox, cxpb=0.2, mutpb=0.8, ngen=10, verbose=True, stats=mstats)