import random
import logging
import os
import re
from collections import defaultdict
import datetime
import shutil

import numpy as np
from deap import base, creator, tools, gp
from deap.algorithms import eaMuPlusLambda
from sklearn.preprocessing import Imputer

import stopit

from .ea.modified_deap import cxOnePoint
from .ea import automl_gp
from .ea.automl_gp import compile_individual, pset_from_config, generate_valid, random_valid_mutation
from .utilities.gama_exceptions import AttributeNotAssignedError
from .utilities.observer import Observer

from .ea.async_gp import async_ea
from .utilities.auto_ensemble import Ensemble

log = logging.getLogger(__name__)

STR_NO_OPTIMAL_PIPELINE = """Gama did not yet establish an optimal pipeline.
                          This can be because `fit` was not yet called, or
                          did not terminate successfully."""


class Gama(object):
    """ Wrapper for the DEAP toolbox logic surrounding the GP process. """

    def __init__(self, 
                 objectives=('neg_log_loss', 'size'),
                 optimize_strategy=(1, -1),
                 config=None,
                 async=False,
                 random_state=None,
                 population_size=10,
                 generations=10,
                 max_total_time=None,
                 max_eval_time=300,
                 n_jobs=1,
                 verbosity=None,
                 cache_dir=None):
        if len(objectives) != len(optimize_strategy):
            error_message = "Length of objectives should match length of optimize_strategy. " \
                             "For each objective, an optimization strategy should be maximized."
            log.error(error_message + " objectives: {}, optimize_strategy: {}".format(objectives, optimize_strategy))
            raise ValueError(error_message)
        if max_total_time is not None and max_total_time <= 0:
            error_message = "max_total_time should be greater than zero, or None."
            log.error(error_message + " max_total_time: {}".format(max_total_time))
            raise ValueError(error_message)
        if max_eval_time is not None and max_eval_time <= 0:
            error_message = "max_eval_time should be greater than zero, or None."
            log.error(error_message + " max_eval_time: {}".format(max_eval_time))
            raise ValueError(error_message)

        self._async_ea = async
        self._best_pipelines = None
        self._fitted_pipelines = {}
        self._random_state = random_state
        self._pop_size = population_size
        self._n_generations = generations
        self._max_total_time = max_total_time
        self._max_eval_time = max_eval_time
        self._fit_data = None
        self._n_jobs = n_jobs
        self._scoring_function = objectives[0]
        self._observer = None
        self._objectives = objectives
        self.ensemble = None

        default_cache_dir = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_GAMA"
        self._cache_dir = cache_dir if cache_dir is not None else default_cache_dir
        if not os.path.isdir(self._cache_dir):
            os.mkdir(self._cache_dir)

        self._imputer = Imputer(strategy="median")
        self._evaluated_individuals = {}
        self._final_pop = None
        self._subscribers = defaultdict(list)

        self._observer = Observer('log.txt')
        self.evaluation_completed(self._observer.update)
        
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

    def predict_proba(self, X):
        """ Predicts the target for input X. 
        
        Predict target for X, using the best found pipeline(s) during the `fit` call. 
        X must be of similar shape to the X value passed to `fit`.
        """
        if len(self._observer._individuals) == 0:
            raise AttributeNotAssignedError(STR_NO_OPTIMAL_PIPELINE)
        #if len(self._observer._individuals) < auto_ensemble_n:
        #    print('Warning: Not enough pipelines evaluated. Continuing with less.')
        if np.isnan(X).any():
            # This does not work if training data set did not have missing numbers.
            X = self._imputer.transform(X)

        if self.ensemble is not None:
            return self.ensemble.predict_proba(X)

    def predict(self, X):
        predictions = np.argmax(self.predict_proba(X), axis=1)
        return np.squeeze(predictions)

    def fit(self, X, y, warm_start=False, auto_ensemble_n=1):
        """ Finds and fits a model to predict target y from X.
        
        Various possible machine learning pipelines will be fit to the (X,y) data.
        Using Genetic Programming, the pipelines chosen should lead to gradually
        better models. Pipelines will internally be validated using cross validation.
        
        After the search termination condition is met, the best found pipeline 
        configuration is then used to train a final model on all provided data.

        TODO: determine how to cut down on the amount of if-else branching in this function.
        """

        # For now there is no support for semi-supervised learning, so remove all instances with unknown targets.
        nan_targets = np.isnan(y)
        if nan_targets.any():
            X = X[~nan_targets, :]
            y = y[~nan_targets]

        # For now we always impute if there are missing values, we always impute with median.
        # One should note that ideally imputation should not always be done since some methods work well without.
        # Secondly, the way imputation is done can also be dependent on the task. Median is generally not the best.
        if np.isnan(X).any():
            self._imputer.fit(X)
            X = self._imputer.transform(X)

        self._fit_data = (X, y)

        if warm_start and self._final_pop is not None:
            pop = self._final_pop
        else:
            if warm_start:
                print('Warning: Warm-start enabled but no earlier fit')
            pop = self._toolbox.population(n=self._pop_size)

        if self._async_ea:
            self._toolbox.register("evaluate", automl_gp.evaluate_pipeline, X=X, y=y, scoring=self._scoring_function, timeout=self._max_eval_time, cache_dir=self._cache_dir)

            def run_ea():
                return async_ea(self, self._n_jobs, pop, self._toolbox, X, y, cxpb=0.2, mutpb=0.8, n_evals=self._n_generations * self._pop_size, verbose=True, evaluation_callback=self._on_evaluation_completed)
        else:
            class DummyHoF:
                pass
            hof = DummyHoF()
            hof.update = self._on_generation_completed
            self._toolbox.register("evaluate", self._compile_and_evaluate_individual, X=X, y=y, scoring=self._scoring_function, timeout=self._max_eval_time)

            def run_ea():
                return eaMuPlusLambda(pop, self._toolbox, mu=len(pop), lambda_=len(pop), cxpb=0.2, mutpb=0.8, ngen=self._n_generations, verbose=True, halloffame=hof)

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

        if len(self._observer._individuals) > 0:
            self.ensemble = Ensemble(self._cache_dir, self._scoring_function, y)
            log.debug('Building ensemble.')
            self.ensemble.build(n_models_in_ensemble=auto_ensemble_n)
            log.debug('Fitting ensemble.')
            self.ensemble.fit(X, y)
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
        score, time = automl_gp.evaluate_pipeline(pl, X, y, timeout, scoring, cache_dir=self._cache_dir)
        length = automl_gp.pipeline_length(ind)

        if self._objectives[1] == 'size':
            fitness = (score, length)
        elif self._objectives[1] == 'time':
            fitness = (score, time)
        elif len(self._objectives) == 1:
            fitness = (score,)

        self._evaluated_individuals[str(ind)] = fitness
        ind.fitness.values = fitness
        self._on_evaluation_completed(ind)
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

    def delete_cache(self):
        if os.path.exists(self._cache_dir):
            shutil.rmtree(self._cache_dir)

    def _on_generation_completed(self, pop):
        for callback in self._subscribers['generation_completed']:
            callback(pop)

    def generation_completed(self, fn):
        """ Register a callback function that is called when new generation is completed.

        :param fn: Function to call when a pipeline is evaluated. Expected signature is: list: ind -> None
        """
        self._subscribers['generation_completed'].append(fn)

    def _on_evaluation_completed(self, ind):
        for callback in self._subscribers['evaluation_completed']:
            callback(ind)

    def evaluation_completed(self, fn):
        """ Register a callback function that is called when new evaluation is completed.

        :param fn: Function to call when a pipeline is evaluated. Expected signature is: ind -> None
        """
        self._subscribers['evaluation_completed'].append(fn)


def pretty_string_individual(individual):
    """ Creates a `pretty` version of the individual string, removing hyperparameter prefixes and the 'data' argument.

    :param individual: Individual of which to return a pretty string representation.
    :return: A string that represents the individual.
    """
    ugly_string = str(individual)
    # Remove the 'data' terminal
    terminal_signature = 'data,'
    terminal_idx = ugly_string.index(terminal_signature)
    pretty_string = ugly_string[:terminal_idx] + ugly_string[terminal_idx + len(terminal_signature):]
    # Remove hyperparameter prefixes
    pretty_string = re.sub('[ .+\.]', '', pretty_string)
    # Because some hyperparameters have a prefix and some don't (shared ones), we can't know where spaces are.
    # Remove all spaces and re-insert them only where wanted.
    pretty_string = pretty_string.replace(' ', '')
    pretty_string = pretty_string.replace(',', ', ')
    return pretty_string
