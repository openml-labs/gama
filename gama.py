import random
import logging
import os
from collections import defaultdict
import datetime
import shutil
from functools import partial
import time

import numpy as np
from deap import base, creator, tools, gp
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
__version__ = '0.1.0'


class Gama(object):
    """ Wrapper for the DEAP toolbox logic surrounding the GP process. """

    def __init__(self, 
                 objectives=('neg_log_loss', 'size'),
                 optimize_strategy=(1, -1),
                 config=None,
                 random_state=None,
                 population_size=50,
                 max_total_time=3600,
                 max_eval_time=300,
                 n_jobs=1,
                 verbosity=None,
                 cache_dir=None):
        log.info('Using GAMA version {}.'.format(__version__))
        log.info('{}({})'.format(
            self.__class__.__name__,
            ','.join(['{}={}'.format(k, v) for (k, v) in locals().items() if k not in ['self', 'config']])
        ))

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

        self._best_pipelines = None
        self._fitted_pipelines = {}
        self._random_state = random_state
        self._pop_size = population_size
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

        self._observer = Observer()
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

        create = partial(automl_gp.offspring_mate_and_mutate, toolbox=self._toolbox, cxpb=0.2, mutpb=0.8)
        self._toolbox.register("create", create)

        if len(self._objectives) == 1:
            self._toolbox.register("select", tools.selTournament, tournsize=3)
            self._toolbox.register("eliminate", automl_gp.eliminate_worst)
        elif len(self._objectives) == 2:
            self._toolbox.register("select", tools.selNSGA2)
            self._toolbox.register("eliminate", automl_gp.eliminate_NSGA)
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
        raise NotImplemented()

    def fit(self, X, y, warm_start=False, auto_ensemble_n=10, restart_=False):
        """ Finds and fits a model to predict target y from X.
        
        Various possible machine learning pipelines will be fit to the (X,y) data.
        Using Genetic Programming, the pipelines chosen should lead to gradually
        better models. Pipelines will internally be validated using cross validation.
        
        After the search termination condition is met, the best found pipeline 
        configuration is then used to train a final model on all provided data.
        """
        start_preprocessing = time.time()
        if hasattr(X, 'values') and hasattr(X, 'astype'):
            X = X.astype(np.float64).values
        if hasattr(y, 'values') and hasattr(y, 'astype'):
            y = y.astype(np.float64).values

        log.debug('fit(). Shapes X: {}({}), y: {}({}).'.format(X.shape, np.isnan(X).any(), y.shape, np.isnan(y).any()))

        # For now there is no support for semi-supervised learning, so remove all instances with unknown targets.
        nan_targets = np.isnan(y)
        if nan_targets.any():
            X = X[~nan_targets, :]
            y = y[~nan_targets]

        # For now we always impute if there are missing values, and we always impute with median.
        # One should note that ideally imputation should not always be done since some methods work well without.
        # Secondly, the way imputation is done can also be dependent on the task. Median is generally not the best.
        self._imputer.fit(X)
        if np.isnan(X).any():
            X = self._imputer.transform(X)

        log.debug('fit-data. Shapes X: {}({}), y: {}({}).'.format(X.shape, np.isnan(X).any(), y.shape, np.isnan(y).any()))
        self._fit_data = (X, y)

        if warm_start and self._final_pop is not None:
            pop = self._final_pop
        else:
            if warm_start:
                print('Warning: Warm-start enabled but no earlier fit')
            pop = self._toolbox.population(n=self._pop_size)

        self._toolbox.register("evaluate", automl_gp.evaluate_pipeline, X=X, y=y,
                               scoring=self._scoring_function, timeout=self._max_eval_time,
                               cache_dir=self._cache_dir)

        processing_time = time.time() - start_preprocessing
        log.info("Preprocessing data took {}s.".format(processing_time))
        try:
            ensemble_ratio = 0.1
            time_left = self._max_total_time - processing_time
            fit_time = int((1-ensemble_ratio)*time_left)
            ensemble_time = int(ensemble_ratio*time_left)
            with stopit.ThreadingTimeout(fit_time) as c_mgr:
                log.debug('Starting EA with max time of {} seconds.'.format(fit_time))

                def restart_criteria():
                    restart = self._observer._individuals_since_last_pareto_update > 400
                    if restart and restart_:
                        self._observer._individuals_since_last_pareto_update = 0
                        self._observer._pareto_front._front = []
                    return restart and restart_

                final_pop = async_ea(self._objectives,
                                     pop,
                                     self._toolbox,
                                     evaluation_callback=self._on_evaluation_completed,
                                     restart_callback=restart_criteria,
                                     n_evaluations=10000,
                                     n_jobs=self._n_jobs)
                self._final_pop = final_pop
        except KeyboardInterrupt:
            log.info('Search phase terminated because of Keyboard Interrupt.')

        if not c_mgr:
            log.info('Search phase terminated because maximum time has elapsed.')

        if len(self._observer._individuals) > 0:
            self._build_fit_ensemble(auto_ensemble_n, timeout=ensemble_time)
        else:
            print('No pipeline evaluated.')

    def _build_fit_ensemble(self, ensemble_size, timeout):
        start_build = time.time()
        X, y = self._fit_data
        self.ensemble = Ensemble(self._scoring_function, y, model_library_directory=self._cache_dir, n_jobs=self._n_jobs)
        log.debug('Building ensemble.')

        # Starting with more models in the ensemble should help against overfitting, but depending on the total
        # ensemble size, it might leave too little room to calibrate the weights or add new models. So we have
        # some adaptive defaults (for now).
        if ensemble_size <= 10:
            self.ensemble.build_initial_ensemble(1)
        else:
            self.ensemble.build_initial_ensemble(10)

        remainder = ensemble_size - self.ensemble._total_model_weights()
        if remainder > 0:
            self.ensemble.add_models(remainder)

        build_time = time.time() - start_build
        timeout = timeout - build_time
        log.info('Building ensemble took {}s. Fitting ensemble with timeout {}s.'.format(build_time, timeout))

        self.ensemble.fit(X, y, timeout=timeout)

        #log.info('Ensemble construction terminated because maximum time has elapsed.')

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
        """ Removes the cache folder and all files associated to this instance. """
        # would use shutil.rmtree(self._cache_dir) but it gives an error if a file is placed in the folder while
        # it is deleting files.
        if os.path.exists(self._cache_dir):
            is_deleted = False
            while not is_deleted:
                try:
                    for f in os.listdir(self._cache_dir):
                        os.remove(os.path.join(self._cache_dir, f))
                    os.rmdir(self._cache_dir)
                    is_deleted = True
                except OSError:
                    log.warning("Exception occurred while removing cache.", exc_info=True)

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
