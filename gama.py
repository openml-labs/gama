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
from .ea.evaluation import string_to_metric
from .utilities.gama_exceptions import AttributeNotAssignedError
from .utilities.observer import Observer

from .ea.async_gp import async_ea
from .utilities.auto_ensemble import Ensemble
from .utilities.stopwatch import Stopwatch
from .utilities import optimal_constant_predictor

log = logging.getLogger(__name__)

STR_NO_OPTIMAL_PIPELINE = """Gama did not yet establish an optimal pipeline.
                          This can be because `fit` was not yet called, or
                          did not terminate successfully."""
__version__ = '0.1.0'


class Gama(object):
    """ Wrapper for the DEAP toolbox logic surrounding the GP process. """

    def __init__(self, 
                 objectives=('filled_in_by_child_class', 'size'),
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

        self._observer = Observer(self._cache_dir)
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
        if (self.ensemble is None) or (sum(map(lambda pl_w: pl_w[1], self.ensemble._fit_models)) == 0):
            # Sum of weights of ensemble is 0, meaning no pipeline finished fitting.
            log.warning("Ensemble did not fit in time. Falling back on default predictions.")
            X_tr, y_tr = self._fit_data
            prediction = optimal_constant_predictor(y_tr, string_to_metric(self._scoring_function))
            return np.asarray([prediction for _ in range(len(X))])

        #if len(self._observer._individuals) < auto_ensemble_n:
        #    print('Warning: Not enough pipelines evaluated. Continuing with less.')
        if np.isnan(X).any():
            # This does not work if training data set did not have missing numbers.
            X = self._imputer.transform(X)

        if self.ensemble is not None:
            return self.ensemble.predict_proba(X)

    def predict(self, X):
        raise NotImplemented()

    def fit(self, X, y, warm_start=False, auto_ensemble_n=25, restart_=False):
        """ Finds and fits a model to predict target y from X.

        Various possible machine learning pipelines will be fit to the (X,y) data.
        Using Genetic Programming, the pipelines chosen should lead to gradually
        better models. Pipelines will internally be validated using cross validation.

        After the search termination condition is met, the best found pipeline
        configuration is then used to train a final model on all provided data.
        """

        ensemble_ratio = 0.1  # fraction of time left after preprocessing that reserved for postprocessing

        def restart_criteria():
            restart = self._observer._individuals_since_last_pareto_update > 400
            if restart and restart_:
                log.info("Continuing search with new population.")
                self._observer.reset_current_pareto_front()
            return restart and restart_

        with Stopwatch() as preprocessing_sw:
            X, y = self._preprocess_phase(X, y)
        log.info("Preprocessing took {:.4f}s. Moving on to search phase.".format(preprocessing_sw.elapsed_time))

        time_left = self._max_total_time - preprocessing_sw.elapsed_time
        fit_time = int((1 - ensemble_ratio) * time_left)

        with Stopwatch() as search_sw:
            self._search_phase(X, y, warm_start=False, restart_criteria=restart_criteria, timeout=fit_time)
        log.info("Search phase took {:.4f}s. Moving on to post processing.".format(search_sw.elapsed_time))

        time_left = self._max_total_time - search_sw.elapsed_time - preprocessing_sw.elapsed_time

        with Stopwatch() as post_sw:
            self._postprocess_phase(auto_ensemble_n, timeout=time_left)
        log.info("Postprocessing took {:.4f}s.".format(post_sw.elapsed_time))

    def _preprocess_phase(self, X, y):
        if hasattr(X, 'values') and hasattr(X, 'astype'):
            X = X.astype(np.float64).values
        if hasattr(y, 'values') and hasattr(y, 'astype'):
            y = y.astype(np.float64).values
        if isinstance(y, list):
            y = np.asarray(y)

        # For now there is no support for semi-supervised learning, so remove all instances with unknown targets.
        nan_targets = np.isnan(y)
        if nan_targets.any():
            log.info("Target vector y has been found to contain NaN-labels. All NaN entries will be ignored because "
                     "supervised learning is not (yet) supported.")
            X = X[~nan_targets, :]
            y = y[~nan_targets]

        # For now we always impute if there are missing values, and we always impute with median.
        # One should note that ideally imputation should not always be done since some methods work well without.
        # Secondly, the way imputation is done can also be dependent on the task. Median is generally not the best.
        self._imputer.fit(X)
        if np.isnan(X).any():
            log.info("Feature matrix X has been found to contain NaN-labels. Data will be imputed using median.")
            X = self._imputer.transform(X)

        self._fit_data = (X, y)
        return X, y

    def _search_phase(self, X, y, warm_start=False, restart_criteria=None, timeout=1e6):
        if warm_start and self._final_pop is not None:
            pop = self._final_pop
        else:
            if warm_start:
                log.warning('Warm-start enabled but no earlier fit. Using new generated population instead.')
            pop = self._toolbox.population(n=self._pop_size)

        self._toolbox.register("evaluate", automl_gp.evaluate_pipeline, X=X, y=y,
                               scoring=self._scoring_function, timeout=self._max_eval_time,
                               cache_dir=self._cache_dir)

        try:
            final_pop = async_ea(self._objectives,
                                 pop,
                                 self._toolbox,
                                 evaluation_callback=self._on_evaluation_completed,
                                 restart_callback=restart_criteria,
                                 max_time_seconds=timeout,
                                 n_jobs=self._n_jobs)
            self._final_pop = final_pop
        except KeyboardInterrupt:
            log.info('Search phase terminated because of Keyboard Interrupt.')

    def _postprocess_phase(self, n, timeout=1e6):
        self._build_fit_ensemble(n, timeout=timeout)

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
