import random
import logging
import os
from collections import defaultdict
import datetime
import shutil
from functools import partial
import warnings
from typing import Callable, Union, List
import uuid

import pandas as pd
import numpy as np
import stopit

import gama.genetic_programming.compilers.scikitlearn
from gama.utilities.metrics import scoring_to_metric
from .utilities.observer import Observer

from gama.data import X_y_from_arff
from gama.search_methods.async_ea import async_ea
from gama.search_methods.asha import asha, evaluate_on_rung
from gama.utilities.generic.timekeeper import TimeKeeper
from gama.logging.utility_functions import register_stream_log, register_file_log
from gama.logging.machine_logging import TOKENS, log_event
from gama.utilities.preprocessing import define_preprocessing_steps, format_x_y
from gama.genetic_programming.mutation import random_valid_mutation_in_place, crossover
from gama.genetic_programming.selection import create_from_population, eliminate_from_pareto
from gama.genetic_programming.operations import create_random_expression
from gama.configuration.parser import pset_from_config
from gama.genetic_programming.operator_set import OperatorSet
from gama.genetic_programming.compilers.scikitlearn import compile_individual
from gama.postprocessing import post_process, FitBest, Ensemble, NoPostProcessing
from gama.utilities.generic.async_executor import AsyncExecutor

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

    :param n_jobs: integer (default=-1)
        The amount of parallel processes that may be created to speed up `fit`.
        Accepted values are positive integers or -1.
        If -1 is specified, multiprocessing.cpu_count() processes are created.

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
                 n_jobs=-1,
                 verbosity=logging.WARNING,
                 keep_analysis_log='gama.log',
                 cache_dir=None):

        register_stream_log(verbosity)
        if keep_analysis_log:
            register_file_log(keep_analysis_log)

        log.info('Using GAMA version {}.'.format(__version__))
        log.info('{}({})'.format(
            self.__class__.__name__,
            ','.join(['{}={}'.format(k, v) for (k, v) in locals().items()
                      if k not in ['self', 'config', 'gamalog', 'file_handler', 'stdout_streamhandler']])
        ))

        if max_total_time is None or max_total_time <= 0:
            raise ValueError(f"max_total_time should be integer greater than zero but is {max_total_time}.")
        if max_eval_time is None or max_eval_time <= 0:
            raise ValueError(f"max_eval_time should be integer greater than zero but is {max_eval_time}.")
        if n_jobs < -1 or n_jobs == 0:
            raise ValueError(f"n_jobs should be -1 or positive integer but is {n_jobs}.")
        elif n_jobs != -1:
            # AsyncExecutor defaults to using multiprocessing.cpu_count(), i.e. n_jobs=-1
            AsyncExecutor.n_jobs = n_jobs

        self._fitted_pipelines = {}
        self._random_state = random_state
        self._pop_size = population_size
        self._max_total_time = max_total_time
        self._max_eval_time = max_eval_time
        self._regularize_length = regularize_length
        self._time_manager = TimeKeeper(max_total_time)
        self._best_pipeline = None
        self._observer = None
        self.ensemble = None
        self._metrics = scoring_to_metric(scoring)
        self._use_asha: bool = False
        self._X: pd.DataFrame = None
        self._y: pd.DataFrame = None
        self._classes: List = []
        self.model: object = None

        default_cache_dir = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:4]}_GAMA"
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

    def _predict(self, x: pd.DataFrame):
        raise NotImplemented('_predict is implemented by base classes.')

    def predict(self, X: Union[pd.DataFrame, np.ndarray]):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            for col in self._X.columns:
                X[col] = X[col].astype(self._X[col].dtype)
        return self._predict(X)

    def predict_arff(self, arff_file_path: str):
        if not isinstance(arff_file_path, str):
            raise TypeError(f"`arff_file_path` must be of type `str` but is of type {type(arff_file_path)}")
        X, _ = X_y_from_arff(arff_file_path)
        return self._predict(X)

    def score(self, x: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        predictions = self.predict_proba(x) if self._metrics[0].requires_probabilities else self.predict(x)
        return self._metrics[0].score(y, predictions)

    def score_arff(self, arff_file_path: str):
        X, y = X_y_from_arff(arff_file_path)
        return self.score(X, y)

    def fit_arff(self, arff_file_path: str, *args, **kwargs):
        """ Find and fit a model to predict the target column (last) from other columns.

        :param arff_file_path: string
            Path to an ARFF file containing the training data.
            The last column is always taken to be the target.
        """
        X, y = X_y_from_arff(arff_file_path)
        self.fit(X, y, *args, **kwargs)

    def fit(self,
            x: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.DataFrame, pd.Series, np.ndarray],
            warm_start: bool=False,
            auto_ensemble_n: int=25,
            restart_: bool=False,
            keep_cache: bool=False):
        """ Find and fit a model to predict target y from X.

        Various possible machine learning pipelines will be fit to the (X,y) data.
        Using Genetic Programming, the pipelines chosen should lead to gradually
        better models. Pipelines will internally be validated using cross validation.

        After the search termination condition is met, the best found pipeline
        configuration is then used to train a final model on all provided data.

        :param x: pandas.DataFrame or numpy.ndarray, shape = [n_samples, n_features]
            Training data. All elements must be able to be converted to float.
        :param y: pandas.DataFrame, pandas.Series or numpy.ndarray, shape = [n_samples,]
            Target values. If a DataFrame is provided, it is assumed the first column contains target values.
        :param warm_start: bool. Indicates the optimization should continue using the last individuals of the
            previous `fit` call.
        :param auto_ensemble_n: positive integer. The number of models to include in the ensemble which is built
            after the optimizatio process.
        :param restart_: bool (default=False)
            Indicates whether or not the search should be restarted when a specific restart criteria is met.
        :param keep_cache: bool (default=False)
            If True, keep the cache directory and its content after fitting is complete. Otherwise delete it.
        """
        if auto_ensemble_n == 0:
            postprocessing = NoPostProcessing
        elif auto_ensemble_n == 1:
            postprocessing = FitBest
        else:
            postprocessing = Ensemble

        def restart_criteria():
            restart = self._observer._individuals_since_last_pareto_update > 400
            if restart and restart_:
                log.info("Continuing search with new population.")
                self._observer.reset_current_pareto_front()
            return restart and restart_

        with self._time_manager.start_activity('preprocessing') as preprocessing_sw:
            self._X, self._y = format_x_y(x, y)
            self._classes = list(set(self._y))
            steps = define_preprocessing_steps(self._X, max_extra_features_created=None, max_categories_for_one_hot=10)
            self._operator_set._safe_compile = partial(compile_individual, preprocessing_steps=steps)

        log_event(log, TOKENS.PREPROCESSING_END, preprocessing_sw.elapsed_time)

        fit_time = int((1 - postprocessing.time_fraction) * self._time_manager.total_time_remaining)

        with self._time_manager.start_activity('search', time_limit=fit_time) as search_sw:
            self._search_phase(warm_start, restart_criteria=restart_criteria, timeout=fit_time)
        log_event(log, TOKENS.SEARCH_END, search_sw.elapsed_time)

        with self._time_manager.start_activity('postprocess',
                                               time_limit=int(self._time_manager.total_time_remaining)) as post_sw:
            best_individual = list(reversed(sorted(self._final_pop, key=lambda ind: ind.fitness.values)))[0]
            self.model = post_process(self, postprocessing,
                                      ensemble_size=auto_ensemble_n,
                                      timeout=self._time_manager.total_time_remaining,
                                      best=best_individual)
        log_event(log, TOKENS.POSTPROCESSING_END, post_sw.elapsed_time)

        if not keep_cache:
            log.debug("Deleting cache.")
            self.delete_cache()

    def _search_phase(self, warm_start: bool=False, restart_criteria: Callable=None, timeout: int=1e6):
        """ Invoke the evolutionary algorithm, populate `final_pop` regardless of termination. """
        if warm_start and self._final_pop is not None:
            pop = self._final_pop
        else:
            if warm_start:
                log.warning('Warm-start enabled but no earlier fit. Using new generated population instead.')
            pop = [self._operator_set.individual() for _ in range(self._pop_size)]

        evaluate_args = dict(evaluate_pipeline_length=self._regularize_length, X=self._X, y_train=self._y,
                             timeout=self._max_eval_time, metrics=self._metrics, cache_dir=self._cache_dir)
        final_pop = []

        try:
            with stopit.ThreadingTimeout(timeout):
                if not self._use_asha:
                    self._operator_set.evaluate = partial(gama.genetic_programming.compilers.scikitlearn.evaluate_individual,
                                                          **evaluate_args)
                    final_pop = async_ea(self._operator_set, output=final_pop, start_population=pop,
                                         restart_callback=restart_criteria)
                else:
                    self._operator_set.evaluate = partial(evaluate_on_rung, **evaluate_args)
                    final_pop = asha(self._operator_set, output=final_pop,
                                     start_candidates=pop, maximum_resource=len(self._X))
        except KeyboardInterrupt:
            log.info('Search phase terminated because of Keyboard Interrupt.')

        self._final_pop = final_pop
        log.debug([str(i) for i in self._final_pop])
        log.info(f'Search phase evaluated {len(self._observer._individuals)} individuals.')

    def _initialize_ensemble(self):
        raise NotImplementedError('_initialize_ensemble should be implemented by a child class.')

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
