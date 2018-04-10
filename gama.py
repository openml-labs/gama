"""
Created on Thu Feb 22 14:30:52 2018

@author: P. Gijsbers
"""
import random
import numpy as np
from scipy.stats import mode

from deap import base, creator, tools, gp
from deap.algorithms import eaSimple

import stopit

from configuration import new_config
from modified_deap import cxOnePoint
import automl_gp
from automl_gp import compile_individual, pset_from_config, generate_valid, random_valid_mutation
from gama_exceptions import AttributeNotAssignedError
from gama_hof import HallOfFame

from async_gp import async_ea, async_ea2

STR_NO_OPTIMAL_PIPELINE = """Gama did not yet establish an optimal pipeline.
                          This can be because `fit` was not yet called, or
                          did not terminate successfully."""

class Gama(object):
    """ Wrapper for the DEAP toolbox logic surrounding the GP process. """
    
    def __init__(self, 
                 objective='accuracy',
                 config=new_config,
                 warm_start=False,
                 random_state=None,
                 pop_size = 10,
                 n_generations = 10,
                 max_total_time = None,
                 max_eval_time = 300):
        self._best_pipelines = None
        self._fitted_pipelines = {}
        self._warm_start = warm_start
        self._random_state = random_state
        self._pop_size = pop_size
        self._n_generations = n_generations
        self._max_total_time = max_total_time
        self._max_eval_time = max_eval_time
        
        self._evaluated_individuals = {}
        
        if self._random_state is not None:
            random.seed(self._random_state)
            np.random.seed(self._random_state)
        
        pset, parameter_checks = pset_from_config(config)
        
        self._pset = pset
        self._toolbox = base.Toolbox()
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)
        
        self._toolbox.register("expr", generate_valid, pset=pset, min_=1, max_=3, toolbox=self._toolbox)
        self._toolbox.register("individual", tools.initIterate, creator.Individual, self._toolbox.expr)
        self._toolbox.register("population", tools.initRepeat, list, self._toolbox.individual)
        self._toolbox.register("compile", compile_individual, pset=pset, parameter_checks=parameter_checks)
        
        self._toolbox.register("mate", cxOnePoint)
        
        self._toolbox.register("mutate", random_valid_mutation, pset=self._pset)
        self._toolbox.register("select", tools.selTournament, tournsize=3)  
    
    def predict(self, X, auto_ensemble_n=1):
        """ Predicts the target for input X. 
        
        Predict target for X, using the best found pipeline(s) during the `fit` call. 
        X must be of similar shape to the X value passed to `fit`.
        """
        if self._best_pipelines is None:
            raise AttributeNotAssignedError(STR_NO_OPTIMAL_PIPELINE)
        if len(self._best_pipelines) < auto_ensemble_n:
            print('Warning: Not enough pipelines evaluated. Continuing with less.')
        
        predictions  = np.zeros((len(X), auto_ensemble_n))
        for i, individual in enumerate(self._best_pipelines[:auto_ensemble_n]):
            if str(individual) in self._fitted_pipelines:
                pipeline = self._fitted_pipelines[str(individual)]
            else:
                Xt, yt = self._fit_data
                pipeline = self._fit_pipeline(individual, Xt, yt)
                self._fitted_pipelines[str(individual)] = pipeline
            predictions[:, i] = pipeline.predict(X)
        
        modes, counts = mode(predictions, axis=1)
        return modes
    
    def fit(self, X, y):
        """ Finds and fits a model to predict target y from X.
        
        Various possible machine learning pipelines will be fit to the (X,y) data.
        Using Genetic Programming, the pipelines chosen should lead to gradually
        better models. Pipelines will internally be validated using crossvalidation.
        
        After the search termination condition is met, the best found pipeline 
        configuration is then used to train a final model on all provided data.
        """
        self._fit_data = (X,y)
        
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        
        
        if self._warm_start:
            pop = self._best_pipelines
        else:
            pop = self._toolbox.population(n=self._pop_size)
            
        hof = HallOfFame('log.txt')
        
        self._toolbox.register("evaluate", self._compile_and_evaluate_individual, X=X, y=y, timeout=self._max_eval_time)
        self._toolbox.register("evaluate", automl_gp.evaluate_pipeline, X = X, y = y, timeout = self._max_eval_time)
        
        run_ea = lambda : eaSimple(pop, self._toolbox, cxpb=0.2, mutpb=0.8, ngen=self._n_generations, verbose=True, halloffame=hof)
        run_ea = lambda : async_ea(pop, self._toolbox, X, y, cxpb=0.2, mutpb=0.8, n_evals =self._n_generations*self._pop_size , verbose=True, halloffame=hof)
        
        if self._max_total_time is not None:
            try:
                with stopit.ThreadingTimeout(self._max_total_time) as c_mgr:
                    pop, log, sdp = run_ea()
            except KeyboardInterrupt:
                print('Keyboard Interrupt sent to outer with statement.')
            if not c_mgr:
                print('Terminated because maximum time has elapsed.')
        else:
            pop, log, sdp = run_ea()
        
        self._ = sdp
        
        
        if len(hof._pop) > 0:
            self._best_pipelines = sorted(hof._pop, key = lambda x: (-x.fitness.values[0], str(x)))
            best_individual = self._best_pipelines[0]
            self._fitted_pipelines[str(best_individual)] = self._fit_pipeline(best_individual, X, y)
        else:
            print('No pipeline evaluated.')
        
    def _fit_pipeline(self, individual, X, y):
        """ Compiles the individual representation and fit the data to it. """
        pipeline = self._toolbox.compile(individual)
        pipeline.fit(X,y)
        return pipeline      
    
    def _compile_and_evaluate_individual(self, ind, X, y, timeout, cv=5):
        if str(ind) in self._evaluated_individuals:
            print('using cache.')
            return self._evaluated_individuals[str(ind)]
        pl = self._toolbox.compile(ind)        
        #if pl is None:
            # Failed to compile due to invalid hyperparameter configuration
        #    return (-float("inf"),)
        fitness = automl_gp.evaluate_pipeline(pl, X, y, timeout)        
        self._evaluated_individuals[str(ind)] = fitness
        return fitness
        
    