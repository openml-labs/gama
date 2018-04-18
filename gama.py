import random
import numpy as np
import scipy.stats

from deap import base, creator, tools, gp
from deap.algorithms import eaMuPlusLambda

import stopit

from configuration import clf_config, reg_config
from modified_deap import cxOnePoint
import automl_gp
from automl_gp import compile_individual, pset_from_config, generate_valid, random_valid_mutation
from gama_exceptions import AttributeNotAssignedError
from gama_hof import HallOfFame

from async_gp import async_ea

STR_NO_OPTIMAL_PIPELINE = """Gama did not yet establish an optimal pipeline.
                          This can be because `fit` was not yet called, or
                          did not terminate successfully."""


class Gama(object):
    """ Wrapper for the DEAP toolbox logic surrounding the GP process. """

    def __init__(self, 
                 objectives=('accuracy', 'size'),
                 optimize_strategy=(1, -1),
                 config=None,
                 async=False,
                 random_state=None,
                 population_size=10,
                 generations=10,
                 max_total_time=None,
                 max_eval_time=300,
                 n_jobs=1):
        if len(objectives) != len(optimize_strategy):
            raise ValueError("Length of objectives should match length of optimize_strategy. "
                             "For each objective, an optimization strategy should be maximized.")
        if max_total_time is not None and max_total_time <= 0:
            raise ValueError("max_total_time should be greater than zero, or None.")
        if max_eval_time is not None and max_eval_time <= 0:
            raise ValueError("max_eval_time should be greater than zero, or None.")

        self._async_ea = async
        self._best_pipelines = None
        self._fitted_pipelines = {}
        self._random_state = random_state
        self._pop_size = population_size
        self._n_generations = generations
        self._max_total_time = max_total_time
        self._max_eval_time = max_eval_time
        self._fit_data = None
        self._n_threads = n_jobs
        self._scoring_function = objectives[0]
        self._hall_of_fame = None
        self._objectives = objectives
        
        self._evaluated_individuals = {}
        self._final_pop = None
        
        if self._random_state is not None:
            random.seed(self._random_state)
            np.random.seed(self._random_state)
        
        pset, parameter_checks = pset_from_config(config)
        
        self._pset = pset
        self._toolbox = base.Toolbox()
        
        creator.create("FitnessMax", base.Fitness, weights=optimize_strategy)
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

        self._toolbox.register("expr", generate_valid, pset=pset, min_=1, max_=3, toolbox=self._toolbox)
        self._toolbox.register("individual", tools.initIterate, creator.Individual, self._toolbox.expr)
        self._toolbox.register("population", tools.initRepeat, list, self._toolbox.individual)
        self._toolbox.register("compile", compile_individual, pset=pset, parameter_checks=parameter_checks)

        self._toolbox.register("mate", cxOnePoint)

        self._toolbox.register("mutate", self._random_valid_mutation_try_new)

        if len(self._objectives) == 1:
            self._toolbox.register("select", tools.selTournament, tournsize=3)
        elif len(self._objectives) == 2:
            self._toolbox.register("select", tools.selNSGA2)
        else:
            raise ValueError('Objectives must be a tuple of length at most 2.')

    def predict(self, X, auto_ensemble_n=1):
        """ Predicts the target for input X. 
        
        Predict target for X, using the best found pipeline(s) during the `fit` call. 
        X must be of similar shape to the X value passed to `fit`.
        """
        if len(self._hall_of_fame._pop) == 0:
            raise AttributeNotAssignedError(STR_NO_OPTIMAL_PIPELINE)
        if len(self._hall_of_fame._pop) < auto_ensemble_n:
            print('Warning: Not enough pipelines evaluated. Continuing with less.')
        
        predictions = np.zeros((len(X), auto_ensemble_n))
        for i, individual in enumerate(self._hall_of_fame.best_n(auto_ensemble_n)):
            print(str(individual), individual.fitness.values)
            if str(individual) in self._fitted_pipelines:
                pipeline = self._fitted_pipelines[str(individual)]
            else:
                Xt, yt = self._fit_data
                pipeline = self._fit_pipeline(individual, Xt, yt)
                self._fitted_pipelines[str(individual)] = pipeline
            predictions[:, i] = pipeline.predict(X)

        return self.merge_predictions(predictions)


    def fit(self, X, y, warm_start=False):
        """ Finds and fits a model to predict target y from X.
        
        Various possible machine learning pipelines will be fit to the (X,y) data.
        Using Genetic Programming, the pipelines chosen should lead to gradually
        better models. Pipelines will internally be validated using cross validation.
        
        After the search termination condition is met, the best found pipeline 
        configuration is then used to train a final model on all provided data.
        """
        self._fit_data = (X, y)
        
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        if warm_start and self._final_pop is not None:
            pop = self._final_pop
        else:
            if warm_start:
                print('Warning: Warm-start enabled but no earlier fit')
            pop = self._toolbox.population(n=self._pop_size)
            self._hall_of_fame = HallOfFame('log.txt')

        if self._async_ea:
            self._toolbox.register("evaluate", automl_gp.evaluate_pipeline, X=X, y=y, scoring=self._scoring_function, timeout=self._max_eval_time)

            def run_ea():
                return async_ea(self, self._n_threads, pop, self._toolbox, X, y, cxpb=0.2, mutpb=0.8, n_evals=self._n_generations*self._pop_size, verbose=True, halloffame=self._hall_of_fame)
        else:
            self._toolbox.register("evaluate", self._compile_and_evaluate_individual, X=X, y=y, scoring=self._scoring_function, timeout=self._max_eval_time)

            def run_ea():
                return eaMuPlusLambda(pop, self._toolbox, mu=len(pop), lambda_=len(pop), cxpb=0.2, mutpb=0.8, ngen=self._n_generations, verbose=True, halloffame=self._hall_of_fame)

        try:
            if self._max_total_time is not None:
                with stopit.ThreadingTimeout(self._max_total_time) as c_mgr:
                    final_pop, sdp = run_ea()
            else:
                final_pop, sdp = run_ea()
            self._final_pop = final_pop
            self._ = sdp
        except KeyboardInterrupt:
            print('Keyboard Interrupt sent to outer with statement.')

        if self._max_total_time is not None and not c_mgr:
            print('Terminated because maximum time has elapsed.')

        if len(self._hall_of_fame._pop) > 0:
            best_individual = self._hall_of_fame.best_n(n=1)[0]
            if str(best_individual) not in self._fitted_pipelines:
                # In the case of warm-starting, the pipeline might have been previously fit.
                self._fitted_pipelines[str(best_individual)] = self._fit_pipeline(best_individual, X, y)
        else:
            print('No pipeline evaluated.')
        
    def _fit_pipeline(self, individual, X, y):
        """ Compiles the individual representation and fit the data to it. """
        pipeline = self._toolbox.compile(individual)
        pipeline.fit(X, y)
        return pipeline
    
    def _compile_and_evaluate_individual(self, ind, X, y, timeout, scoring='accuracy', cv=5):
        if str(ind) in self._evaluated_individuals:
            print('using cache.')
            return self._evaluated_individuals[str(ind)]
        pl = self._toolbox.compile(ind)        
        if pl is None:
            # Failed to compile due to invalid hyperparameter configuration
            return -float("inf"), 1
        score, time = automl_gp.evaluate_pipeline(pl, X, y, timeout, scoring)
        length = automl_gp.pipeline_length(ind)

        if self._objectives[1] == 'size':
            fitness = (score, length)
        elif self._objectives[1] == 'time':
            fitness = (score, time)
        elif len(self._objectives) == 1:
            fitness = (score,)

        self._evaluated_individuals[str(ind)] = fitness
        return fitness

    def _random_valid_mutation_try_new(self, ind):
        """ Call `random_valid_mutation` until a new individual (that was not evaluated before) is created (at most 50x).
        """
        ind_copy = self._toolbox.clone(ind)
        for _ in range(50):
            new_ind, = random_valid_mutation(ind_copy, self._pset)
            if str(new_ind) not in self._evaluated_individuals:
                return new_ind,
        return new_ind,
