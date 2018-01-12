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

d ={
    'sklearn.tree.DecisionTreeClassifier': {
        'criterion': ["gini", "entropy"],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },
    
    'sklearn.naive_bayes.BernoulliNB': {
            'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
            'fit_prior': [True, False]
    }
}
    
class Xtrain(np.ndarray):
    pass 
class Ytrain(np.ndarray):
    pass 
class Xtest(np.ndarray):
    pass 
class Output(np.ndarray):
    pass 

def ProcessBernoulliNB(X_tr, X_te, y_tr, alpha, fit):
    bnb = BernoulliNB(alpha=alpha, fit_prior=fit)
    bnb.fit(X_tr, y_tr)
    return bnb.predict(X_te)

def ProcessDecisionTree(X_tr, X_te, y_tr, max_depth, max_features, min_samples_split):
    dt = DecisionTreeClassifier(max_depth=max_depth, max_features=max_features, min_samples_split=min_samples_split)
    dt.fit(X_tr, y_tr)
    return dt.predict(X_te)

def evalPipeline(ind, X_tr, X_te, y_tr, y_te):
    # pipe = MLPipeline(ind)
    # pipe.fit(X_tr,y_tr)
    # return pipe.score(X_te, y_te)
    func = toolbox.compile(expr=ind)
    preds = func(X_tr, X_te, y_tr)
    return accuracy_score(y_te, preds),

# in = X out = y
# Define Primitives
pset = PrimitiveSetTyped("pipeline",in_types=[Xtrain, Ytrain, Xtest], ret_type=Output)
pset.renameArguments(ARG0="X_tr")
pset.renameArguments(ARG1="X_te")
pset.renameArguments(ARG2="y_tr")
#pset.addPrimitive(ProcessBernoulliNB, [Xtrain, Ytrain, Xtest, float, bool], Output)
pset.addPrimitive(ProcessDecisionTree, [Xtrain, Ytrain, Xtest, int, float, float], Output)

# Define Terminals
pset.addTerminal(True, bool)
pset.addTerminal(False, bool)

for i in range(21,100):
    pset.addTerminal(1.0/(i-20), float)  
for i in range(1,10):
    pset.addTerminal(i, int)   

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

toolbox = base.Toolbox()
toolbox.register("expr", genFull, pset=pset, min_=1, max_=1)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

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