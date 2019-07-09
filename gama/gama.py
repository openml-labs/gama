import random
import logging
import os
from collections import defaultdict, Iterable
import datetime
import multiprocessing
import shutil
from functools import partial
import sys
import time
import warnings
from typing import Tuple, Callable

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import stopit

import gama.genetic_programming.compilers.scikitlearn
from gama.genetic_programming.algorithms.metrics import Metric
from .utilities.observer import Observer

from gama.data import X_y_from_arff
from gama.genetic_programming.algorithms.async_ea import async_ea
from gama.genetic_programming.algorithms.asha import asha, evaluate_on_rung
from gama.utilities.generic.timekeeper import TimeKeeper
from gama.utilities.logging_utilities import TOKENS, log_parseable_event
from gama.utilities.preprocessing import define_preprocessing_steps, heuristic_numpy_to_dataframe
from gama.genetic_programming.mutation import random_valid_mutation_in_place, crossover
from gama.genetic_programming.selection import create_from_population, eliminate_from_pareto
from gama.genetic_programming.operations import create_random_expression
from gama.configuration.parser import pset_from_config
from gama.genetic_programming.operator_set import OperatorSet
from gama.genetic_programming.compilers.scikitlearn import compile_individual

# `gamalog` is for the entire gama module and submodules.
gamalog = logging.getLogger('gama')
gamalog.setLevel(logging.DEBUG)

log = logging.getLogger(__name__)

STR_NO_OPTIMAL_PIPELINE = """Gama did not yet establish an optimal pipeline.
                          This can be because `fit` was not yet called, or
                          did not terminate successfully."""
__version__ = '19.01.0'

for module_to_ignore in ["sklearn", "numpy"]:
    warnings.filterwarnings("ignore", module=module_to_ignore)


class Gama(object):
    """ Wrapper for the toolbox logic surrounding the GP process as well as ensemble construction.

    :param scoring: string, Metric or tuple.
        Specifies the/all metric(s) to optimize towards. A string will be converted to Metric. A tuple must
        specify each metric with the same type (i.e. all str or all Metric).

        The valid metrics depend on the type of task. Many scikit-learn metrics are available.
        For classification, the following metrics are available:
        'accuracy', 'roc_auc', 'average_precision', 'log_loss', 'precision_macro', 'precision_micro',
        'precision_weighted', 'precision_samples', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted',
        'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted'.
        For regression, the following metrics are available:
        'explained_variance', 'r2', 'median_absolute_error', 'mean_squared_error', 'mean_squared_log_error', 'mean_absolute_error'.
        Whether to minimize or maximize is determined automatically (though can be overwritten by `optimize_strategy`.
        However, you can instead also specify 'neg\_'+metric (e.g. 'neg_log_loss') as metric to make it explicit.

    :param regularize_length: bool.
        If True, add pipeline length as an optimization metric (preferring short over long).

    :param config: a dictionary which specifies available components and their valid hyperparameter settings
        For more information, see :ref:`search_space_configuration`.

    :param random_state:  integer or None (default=None)
        If an integer is passed, this will be the seed for the random number generators used in the process.
        However, with `n_jobs > 1`, there will be randomization introduced by multi-processing.
        For reproducible results, set this and use `n_jobs=1`.

    :param population_size: positive integer (default=50)
        Number of individuals to keep in the population at any one time.

    :param max_total_time: positive integer (default=3600)
        Time in seconds that can be used for the `fit` call.

    :param max_eval_time: positive integer or None (default=300)
        Time in seconds that can be used to evaluate any one single individual.

    :param n_jobs: integer (default=1)
        The amount of parallel processes that may be created to speed up `fit`. If this number
        is zero or negative, it will be set to the amount of cores.

    :param verbosity: integer (default=logging.WARNING)
        Sets the level of log messages to be automatically output to terminal.

    :param keep_analysis_log: str or False. (default='gama.log')
        If non-empty str, specifies the path (and name) where the log should be stored, e.g. /output/gama.log.
        If empty str or False, no log is stored.

    :param cache_dir: string or None (default=None)
        The directory in which to keep the cache during `fit`. In this directory,
        models and their evaluation results will be stored. This facilitates a quick ensemble construction.
    """

    def __init__(self,
                 scoring='filled_in_by_child_class',
                 regularize_length=True,
                 config=None,
                 random_state=None,
                 population_size=50,
                 max_total_time=3600,
                 max_eval_time=300,
                 n_jobs=1,
                 verbosity=logging.WARNING,
                 keep_analysis_log='gama.log',
                 cache_dir=None):

        if verbosity >= logging.DEBUG:
            stdout_streamhandler = logging.StreamHandler(sys.stdout)
            stdout_streamhandler.setLevel(verbosity)
            gamalog.addHandler(stdout_streamhandler)

        if keep_analysis_log:
            file_handler = logging.FileHandler(keep_analysis_log)
            file_handler.setLevel(logging.DEBUG)
            gamalog.addHandler(file_handler)

        log.info('Using GAMA version {}.'.format(__version__))
        log.info('{}({})'.format(
            self.__class__.__name__,
            ','.join(['{}={}'.format(k, v) for (k, v) in locals().items()
                      if k not in ['self', 'config', 'gamalog', 'file_handler', 'stdout_streamhandler']])
        ))

        if max_total_time is None or max_total_time <= 0:
            error_message = "max_total_time should be greater than zero."
            log.error(error_message + " max_total_time: {}".format(max_total_time))
            raise ValueError(error_message)
        if max_eval_time is not None and max_eval_time <= 0:
            error_message = "max_eval_time should be greater than zero, or None."
            log.error(error_message + " max_eval_time: {}".format(max_eval_time))
            raise ValueError(error_message)

        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        self._fitted_pipelines = {}
        self._random_state = random_state
        self._pop_size = population_size
        self._max_total_time = max_total_time
        self._max_eval_time = max_eval_time
        self._n_jobs = n_jobs
        self._regularize_length = regularize_length
        self._time_manager = TimeKeeper(max_total_time)
        self._best_pipeline = None
        self._observer = None
        self.ensemble = None
        self._ensemble_fit: bool = False
        self._metrics = self._scoring_to_metric(scoring)
        self._use_asha: bool = False
        self._X: pd.DataFrame = None
        self._y: pd.Series = None
        self._y_score = None

        default_cache_dir = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_GAMA"
        self._cache_dir = cache_dir if cache_dir is not None else default_cache_dir
        if not os.path.isdir(self._cache_dir):
            os.mkdir(self._cache_dir)

        self._imputer = None
        self._evaluated_individuals = {}
        self._final_pop = None
        self._subscribers = defaultdict(list)

        self._observer = Observer(self._cache_dir)
        self.evaluation_completed(self._observer.update)
        
        if self._random_state is not None:
            random.seed(self._random_state)
            np.random.seed(self._random_state)

        self._pset, parameter_checks = pset_from_config(config)
        self._operator_set = OperatorSet(
            mutate=partial(random_valid_mutation_in_place, primitive_set=self._pset),
            mate=crossover,
            create_from_population=partial(create_from_population, cxpb=0.2, mutpb=0.8),
            create_new=partial(create_random_expression, primitive_set=self._pset),
            compile_=compile_individual,
            eliminate=eliminate_from_pareto,
            evaluate_callback=self._on_evaluation_completed
        )

    def _scoring_to_metric(self, scoring):
        if isinstance(scoring, str):
            return tuple([Metric.from_string(scoring)])
        elif isinstance(scoring, Metric):
            return tuple([scoring])
        elif isinstance(scoring, Iterable):
            if all(isinstance(scorer, Metric) for scorer in scoring):
                return scoring
            elif all(isinstance(scorer, str) for scorer in scoring):
                return tuple([Metric.from_string(scorer) for scorer in scoring])
            else:
                raise ValueError("Iterable of mixed types for `scoring` currently not supported.")
        else:
            raise ValueError("scoring must be a string, Metric or Iterable (of strings or Metrics).")

    def _preprocess_predict_X(self, X=None, arff_file_path=None):
        if isinstance(arff_file_path, str):
            X, _ = X_y_from_arff(arff_file_path)
        elif isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            for col in self._X.columns:
                X[col] = X[col].astype(self._X[col].dtype)
        elif X is None:
            raise ValueError("Must specify either X or arff_file_path.")

        return X

    def predict(self, X=None, arff_file_path=None):
        raise NotImplemented('predict is implemented by base classes.')

    def score(self, X=None, y=None, arff_file_path=None):
        if arff_file_path:
            X, y = X_y_from_arff(arff_file_path)
        y_score = self._construct_y_score(y)

        predictions = self.predict_proba(X) if self._metrics[0].requires_probabilities else self.predict(X)
        return self._metrics[0].score(y_score, predictions)

    def _preprocess(self, X, y) -> Tuple[pd.DataFrame, pd.Series]:
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise TypeError("X must be either np.ndarray or pd.DataFrame.")
        if not isinstance(y, (np.ndarray, pd.Series)):
            raise TypeError("y must be either np.ndarray or pd.Series.")

        with self._time_manager.start_activity('preprocessing') as preprocessing_sw:
            # Internally X is always a pd.DataFrame and y is always a pd.Series
            if isinstance(X, np.ndarray):
                X = heuristic_numpy_to_dataframe(X)
            if hasattr(self, '_encode_labels'):
                # This will return a numpy array
                y = self._encode_labels(y)
            if isinstance(y, np.ndarray):
                y = pd.Series(y)

            if y.isnull().any():
                log.info("Target vector has been found to contain NaN-labels, these rows will be ignored.")
                X, y = X.loc[~y.isnull(), :], y[~y.isnull()]

            steps = define_preprocessing_steps(X, max_extra_features_created=None, max_categories_for_one_hot=10)
            self._operator_set._safe_compile = partial(compile_individual, preprocessing_steps=steps)

        log_parseable_event(log, TOKENS.PREPROCESSING_END, preprocessing_sw.elapsed_time)
        return X, y

    def fit_arff(self, arff_file_path: str, *args, **kwargs):
        """ Find and fit a model to predict the target column (last) from other columns.

        :param arff_file_path: string
            Path to an ARFF file containing the training data.
            The last column is always taken to be the target.
        """
        X, y = X_y_from_arff(arff_file_path)
        self.fit(X, y, *args, **kwargs)

    def fit(self, X=None, y=None, warm_start=False, auto_ensemble_n=25, restart_=False, keep_cache=False):
        """ Find and fit a model to predict target y from X.

        Various possible machine learning pipelines will be fit to the (X,y) data.
        Using Genetic Programming, the pipelines chosen should lead to gradually
        better models. Pipelines will internally be validated using cross validation.

        After the search termination condition is met, the best found pipeline
        configuration is then used to train a final model on all provided data.

        :param X: Numpy Array, shape = [n_samples, n_features]
            Training data. All elements must be able to be converted to float.
        :param y: Numpy Array, shape = [n_samples,]
            Target values.
        :param warm_start: bool. Indicates the optimization should continue using the last individuals of the
            previous `fit` call.
        :param auto_ensemble_n: positive integer. The number of models to include in the ensemble which is built
            after the optimizatio process.
        :param restart_: bool. Indicates whether or not the search should be restarted when a specific restart
            criteria is met.
        """
        ensemble_ratio = 0.3  # fraction of time left after preprocessing that reserved for postprocessing

        def restart_criteria():
            restart = self._observer._individuals_since_last_pareto_update > 400
            if restart and restart_:
                log.info("Continuing search with new population.")
                self._observer.reset_current_pareto_front()
            return restart and restart_

        self._X, self._y = self._preprocess(X, y)
        self._y_score = self._construct_y_score(y)

        fit_time = int((1 - ensemble_ratio) * self._time_manager.total_time_remaining)

        with self._time_manager.start_activity('search', time_limit=fit_time) as search_sw:
            self._search_phase(warm_start, restart_criteria=restart_criteria, timeout=fit_time)
        log_parseable_event(log, TOKENS.SEARCH_END, search_sw.elapsed_time)

        with self._time_manager.start_activity('postprocess',
                                               time_limit=int(self._time_manager.total_time_remaining)) as post_sw:
            self._postprocess_phase(auto_ensemble_n, timeout=self._time_manager.total_time_remaining)
        log_parseable_event(log, TOKENS.POSTPROCESSING_END, post_sw.elapsed_time)

        if not keep_cache:
            log.debug("Deleting cache.")
            self.delete_cache()

    def _construct_y_score(self, y: pd.Series):
        if any(metric.requires_probabilities for metric in self._metrics):
            return OneHotEncoder(categories='auto').fit_transform(y.reshape(-1, 1)).todense()
        return y

    def _search_phase(self, warm_start: bool=False, restart_criteria: Callable=None, timeout: int=1e6):
        """ Invoke the evolutionary algorithm, populate `final_pop` regardless of termination. """
        if warm_start and self._final_pop is not None:
            pop = self._final_pop
        else:
            if warm_start:
                log.warning('Warm-start enabled but no earlier fit. Using new generated population instead.')
            pop = [self._operator_set.individual() for _ in range(self._pop_size)]

        evaluate_args = dict(evaluate_pipeline_length=self._regularize_length, X=self._X, y_train=self._y, y_score=self._y_score,
                             timeout=self._max_eval_time, metrics=self._metrics, cache_dir=self._cache_dir)
        final_pop = []

        try:
            with stopit.ThreadingTimeout(timeout):
                if not self._use_asha:
                    self._operator_set.evaluate = partial(gama.genetic_programming.compilers.scikitlearn.evaluate_individual,
                                                          **evaluate_args)
                    final_pop = async_ea(self._operator_set, output=final_pop, start_population=pop,
                                         restart_callback=restart_criteria,
                                         max_time_seconds=timeout,
                                         n_jobs=self._n_jobs)
                else:
                    self._operator_set.evaluate = partial(evaluate_on_rung, **evaluate_args)
                    final_pop = asha(self._operator_set, output=final_pop,
                                     start_candidates=pop, maximum_resource=len(self._X))
                log.debug([str(i) for i in self._final_pop])
        except KeyboardInterrupt:
            log.info('Search phase terminated because of Keyboard Interrupt.')

        self._final_pop = final_pop
        log.info('Search phase evaluated {} individuals.'.format(len(final_pop)))

    def _postprocess_phase(self, n, timeout=1e6):
        """ Perform any necessary post processing, such as ensemble building. """
        self._best_individual = list(reversed(sorted(self._final_pop, key=lambda ind: ind.fitness.values)))[0]
        log.info("Best pipeline has fitness of {}".format(self._best_individual.fitness.values))
        self._best_pipeline = self._best_individual.pipeline
        log.info("Pipeline {}, steps: {}".format(self._best_pipeline, self._best_pipeline.steps))
        self._best_pipeline.fit(self._X, self._y)
        if n > 1:
            self._build_fit_ensemble(n, timeout=timeout)

    def _initialize_ensemble(self):
        raise NotImplementedError('_initialize_ensemble should be implemented by a child class.')

    def _build_fit_ensemble(self, ensemble_size, timeout):
        start_build = time.time()
        try:
            log.debug('Building ensemble.')
            self._initialize_ensemble()

            # Starting with more models in the ensemble should help against overfitting, but depending on the total
            # ensemble size, it might leave too little room to calibrate the weights or add new models. So we have
            # some adaptive defaults (for now).
            if ensemble_size <= 10:
                self.ensemble.build_initial_ensemble(1)
            else:
                self.ensemble.build_initial_ensemble(10)

            remainder = ensemble_size - self.ensemble._total_model_weights()
            if remainder > 0:
                self.ensemble.expand_ensemble(remainder)

            build_time = time.time() - start_build
            timeout = timeout - build_time
            log.info('Building ensemble took {}s. Fitting ensemble with timeout {}s.'.format(build_time, timeout))

            self.ensemble.fit(self._X, self._y, timeout=timeout)
            self._ensemble_fit = True
        except Exception as e:
            log.warning("Error during auto ensemble: {}".format(e))

    def delete_cache(self):
        """ Removes the cache folder and all files associated to this instance. """
        while os.path.exists(self._cache_dir):
            try:
                log.info("Attempting to delete {}".format(self._cache_dir))
                shutil.rmtree(self._cache_dir)
            except OSError as e:
                if "The directory is not empty" not in str(e):
                    log.warning("Did not delete due to:", exc_info=True)
                # else ignore silently. This can occur if an evaluation process writes to cache.

    def _safe_outside_call(self, fn):
        """ Calls fn and log any exception it raises without reraising, except for TimeoutException. """
        try:
            fn()
        except stopit.utils.TimeoutException:
            raise
        except Exception:
            # We actually want to catch any other exception here, because the callback code can be
            # arbitrary (it can be provided by users). This excuses the catch-all Exception.
            # Note that KeyboardInterrupts are not exceptions and get elevated to the caller.
            log.warning("Exception during callback.", exc_info=True)
            pass
        if self._time_manager.current_activity.exceeded_limit:
            log.info("Time exceeded during callback, but exception was swallowed.")
            raise stopit.utils.TimeoutException

    def _on_evaluation_completed(self, ind):
        for callback in self._subscribers['evaluation_completed']:
            self._safe_outside_call(partial(callback, ind))

    def evaluation_completed(self, fn):
        """ Register a callback function that is called when new evaluation is completed.

        :param fn: Function to call when a pipeline is evaluated. Expected signature is: ind -> None
        """
        self._subscribers['evaluation_completed'].append(fn)
