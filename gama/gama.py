from abc import ABC
from collections import defaultdict
import datetime
from functools import partial
import logging
import os
import random
import shutil
import time
from typing import Union, Tuple, Optional, Dict
import uuid
import warnings

import pandas as pd
import numpy as np
import stopit

import gama.genetic_programming.compilers.scikitlearn
from gama.genetic_programming.components import Individual
from gama.logging.machine_logging import log_event, TOKENS
from gama.search_methods.base_search import BaseSearch
from gama.utilities.generic.async_evaluation import AsyncEvaluator
from gama.utilities.metrics import scoring_to_metric
from .utilities.observer import Observer

from gama.data import X_y_from_arff
from gama.search_methods.async_ea import AsyncEA

from gama.utilities.generic.timekeeper import TimeKeeper
from gama.logging.utility_functions import register_stream_log, register_file_log

from gama.logging.machine_logging import TOKENS, log_event, MACHINE_LOG_LEVEL
from gama.utilities.preprocessing import define_preprocessing_steps, format_x_y
from gama.genetic_programming.mutation import random_valid_mutation_in_place
from gama.genetic_programming.mutation import random_valid_mutation_in_place
from gama.genetic_programming.crossover import random_crossover
from gama.genetic_programming.conformant_mutation import random_valid_mutation_in_place as conformant_mutation
from gama.genetic_programming.conformant_mutation import crossover as conformant_crossover
from gama.genetic_programming.selection import create_from_population, eliminate_from_pareto
from gama.genetic_programming.operations import create_random_expression, create_expression_by_rule
from gama.configuration.parser import pset_from_config
from gama.genetic_programming.operator_set import OperatorSet
from gama.genetic_programming.compilers.scikitlearn import compile_individual

from gama.d3m.metalearning import generate_warm_start_pop

#  `gamalog` is for the entire gama module and submodules.
gamalog = logging.getLogger('gama')
gamalog.setLevel(MACHINE_LOG_LEVEL)

from gama.postprocessing import BestFitPostProcessing, EnsemblePostProcessing, NoPostProcessing, BasePostProcessing
from gama.utilities.generic.async_evaluator import AsyncEvaluator
from gama.utilities.metrics import Metric


log = logging.getLogger(__name__)

STR_NO_OPTIMAL_PIPELINE = """Gama did not yet establish an optimal pipeline.
                          This can be because `fit` was not yet called, or
                          did not terminate successfully."""
__version__ = '19.01.0'

for module_to_ignore in ["sklearn", "numpy"]:
    warnings.filterwarnings("ignore", module=module_to_ignore)

class Gama(ABC):
    """ Wrapper for the toolbox logic surrounding executing the AutoML pipeline. """

    def __init__(self,
                 scoring: Union[str, Metric, Tuple[Union[str, Metric], ...]] = 'filled_in_by_child_class',
                 regularize_length: bool = True,
                 config: Dict = None,
                 random_state: int = None,
                 max_total_time: int = 3600,
                 max_eval_time: Optional[int] = None,
                 n_jobs: int = 1,
                 verbosity: int = logging.WARNING,
                 keep_analysis_log: Optional[str] = 'gama.log',
                 cache_dir: Optional[str] = None,
                 search_method: BaseSearch = AsyncEA(),
                 post_processing_method: BasePostProcessing = BestFitPostProcessing(),
                 grammar_file_name=None,
                 rule_name=None):
        """

        Parameters
        ----------
        scoring: str, Metric or Tuple
            Specifies the/all metric(s) to optimize towards. A string will be converted to Metric. A tuple must
            specify each metric with the same type (i.e. all str or all Metric). See :ref:`Metrics` for built-in
            metrics.

        regularize_length: bool
            If True, add pipeline length as an optimization metric (preferring short over long).

        config: a dictionary which specifies available components and their valid hyperparameter settings
            For more information, see :ref:`search_space_configuration`.

        random_state:  int or None (default=None)
            If an integer is passed, this will be the seed for the random number generators used in the process.
            However, with `n_jobs > 1`, there will be randomization introduced by multi-processing.
            For reproducible results, set this and use `n_jobs=1`.

        max_total_time: positive int (default=3600)
            Time in seconds that can be used for the `fit` call.

        max_eval_time: positive int, optional (default=None)
            Time in seconds that can be used to evaluate any one single individual.
            If None, set to 0.1 * max_total_time.

        n_jobs: int (default=1)
            The amount of parallel processes that may be created to speed up `fit`.
            Accepted values are positive integers or -1.
            If -1 is specified, multiprocessing.cpu_count() processes are created.

        verbosity: int (default=logging.WARNING)
            Sets the level of log messages to be automatically output to terminal.

        keep_analysis_log: str, optional (default='gama.log')
            If non-empty str, specifies the path (and name) where the log should be stored, e.g. /output/gama.log.
            If `None`, no log is stored.

        cache_dir: str or None (default=None)
            The directory in which to keep the cache during `fit`. In this directory,
            models and their evaluation results will be stored. This facilitates a quick ensemble construction.

        search_method: BaseSearch (default=AsyncEA())
            Search method to use to find good pipelines. Should be instantiated.

        post_processing_method: BasePostProcessing (default=BestFitPostProcessing())
            Post-processing method to create a model after the search phase. Should be instantiated.

        grammar_file_name: string, optional (default=None)
            The name of a grammar file to use in generating and validating individuals.

        rule_name: string, optional (default=None)
            The name of a grammar rule to use in generating and validating individuals.
        """
        register_stream_log(verbosity)
        if keep_analysis_log is not None:
            register_file_log(keep_analysis_log)

        arguments = ','.join(['{}={}'.format(k, v) for (k, v) in locals().items()
                              if k not in ['self', 'config', 'gamalog', 'file_handler', 'stdout_streamhandler']])
        log.info('Using GAMA version {}.'.format(__version__))
        log.info('{}({})'.format(self.__class__.__name__, arguments))
        log_event(log, TOKENS.INIT, [arguments])

        if max_total_time is None or max_total_time <= 0:
            raise ValueError(f"max_total_time should be integer greater than zero but is {max_total_time}.")
        if max_eval_time is not None and max_eval_time <= 0:
            raise ValueError(f"max_eval_time should be None or integer greater than zero but is {max_eval_time}.")
        if n_jobs < -1 or n_jobs == 0:
            raise ValueError(f"n_jobs should be -1 or positive integer but is {n_jobs}.")
        elif n_jobs != -1:
            # AsyncExecutor defaults to using multiprocessing.cpu_count(), i.e. n_jobs=-1
            AsyncEvaluator.n_jobs = n_jobs

        if max_eval_time is None:
            max_eval_time = 0.1 * max_total_time
        if max_eval_time > max_total_time:
            log.warning(f"max_eval_time ({max_eval_time}) > max_total_time ({max_total_time}) is not allowed. "
                        f"max_eval_time set to {max_total_time}.")
            max_eval_time = max_total_time

        self._max_eval_time = max_eval_time
        self._time_manager = TimeKeeper(max_total_time)
        self._metrics: Tuple[Metric] = scoring_to_metric(scoring)
        self._regularize_length = regularize_length
        self._search_method: BaseSearch = search_method
        self._post_processing = post_processing_method

        default_cache_dir = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:4]}_GAMA"
        self._cache_dir = cache_dir if cache_dir is not None else default_cache_dir
        if not os.path.isdir(self._cache_dir):
            os.mkdir(self._cache_dir)

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        self._X: Optional[pd.DataFrame] = None
        self._y: Optional[pd.DataFrame] = None
        self.model: object = None
        self._final_pop = None

        self._subscribers = defaultdict(list)
        self._observer = Observer(self._cache_dir)
        self.evaluation_completed(self._observer.update)

        self._pset, parameter_checks = pset_from_config(config)

        if grammar_file_name is not None:
            from gama.utilities.plgen.manager import Manager
            from gama.utilities.plgen.library import GamaPsetLibrary
            library = GamaPsetLibrary(self._pset)
            self._grammar_manager = Manager(library=library)
            self._grammar_manager.parse_file(grammar_file_name)
            self._grammar_rule_name = rule_name
            self._grammar_rule = self._grammar_manager.get_fa(rule_name)
            expression_creator = partial(create_expression_by_rule, primitive_set=self._pset, rule=self._grammar_rule)
            def individual_creator():
                return Individual(expression_creator(), )
            individual_creator = expression_creator
            # Turning off the grammar checking of individuals generated through reproduction
            # TODO: Debug grammar checking for reproduction
#            mutate = partial(conformant_mutation, primitive_set=self._pset, rule=self._grammar_rule)
#            mate = partial(conformant_crossover, rule=self._grammar_manager)
            mutate = partial(random_valid_mutation_in_place, primitive_set=self._pset)
            mate = random_crossover
        else:
            individual_creator = partial(create_random_expression, primitive_set=self._pset)
            mutate = partial(random_valid_mutation_in_place, primitive_set=self._pset)
            mate = random_crossover

        self._operator_set = OperatorSet(
            mutate=mutate,
            mate=mate,
            create_from_population=partial(create_from_population, cxpb=0.2, mutpb=0.8),
            create_new=individual_creator,
            compile_=compile_individual,
            eliminate=eliminate_from_pareto,
            evaluate_callback=self._on_evaluation_completed
        )

    def clean_pipeline_string(self, p):
        return str(p)

    def _predict(self, x: pd.DataFrame):
        raise NotImplemented('_predict is implemented by base classes.')

    def predict(self, x: Union[pd.DataFrame, np.ndarray]):
        """ Predict the target for input X.

        Parameters
        ----------
        x: pandas.DataFrame or numpy.ndarray
            A dataframe or array with the same number of columns as the input to `fit`.

        Returns
        -------
        numpy.ndarray
            array with predictions of shape (N,) where N is the length of the first dimension of X.
        """
        if isinstance(x, np.ndarray):
            x = pd.DataFrame(x)
            for col in self._X.columns:
                x[col] = x[col].astype(self._X[col].dtype)
        return self._predict(x)

    def predict_arff(self, arff_file_path: str):
        """ Predict the target for input found in the ARFF file.

        Parameters
        ----------
        arff_file_path: str
            An ARFF file with the same columns as the one that used in fit.
            The target column is ignored (but must be present).

        Returns
        -------
        numpy.ndarray
            array with predictions for each row in the ARFF file.
        """
        if not isinstance(arff_file_path, str):
            raise TypeError(f"`arff_file_path` must be of type `str` but is of type {type(arff_file_path)}")
        X, _ = X_y_from_arff(arff_file_path)
        return self._predict(X)

    def score(self, x: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """ Calculate the score of the model according to the `scoring` metric and input (x, y).

        Parameters
        ----------
        x: pandas.DataFrame or numpy.ndarray
            Data to predict target values for.
        y: pandas.Series or numpy.ndarray
            True values for the target.

        Returns
        -------
        float
            The score obtained on the given test data according to the `scoring` metric.
        """
        predictions = self.predict_proba(x) if self._metrics[0].requires_probabilities else self.predict(x)
        return self._metrics[0].score(y, predictions)

    def score_arff(self, arff_file_path: str) -> float:
        """ Calculate the score of the model according to the `scoring` metric and input in the ARFF file.

        Parameters
        ----------
        arff_file_path: string
            An ARFF file with which to calculate the score.

        Returns
        -------
        float
            The score obtained on the given test data according to the `scoring` metric.
        """
        X, y = X_y_from_arff(arff_file_path)
        return self.score(X, y)

    def fit_arff(self, arff_file_path: str, *args, **kwargs):
        """ Find and fit a model to predict the target column (last) from other columns.

        Parameters
        ----------
        arff_file_path: string
            Path to an ARFF file containing the training data.
            The last column is always taken to be the target.
        """
        X, y = X_y_from_arff(arff_file_path)
        self.fit(X, y, *args, **kwargs)

    def fit(self,
            x: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.DataFrame, pd.Series, np.ndarray],
            warm_start: bool = False,
            keep_cache: bool = False,
            d3m_mode: bool = False):
        """ Find and fit a model to predict target y from X.

        Various possible machine learning pipelines will be fit to the (X,y) data.
        Using Genetic Programming, the pipelines chosen should lead to gradually
        better models. Pipelines will internally be validated using cross validation.

        After the search termination condition is met, the best found pipeline
        configuration is then used to train a final model on all provided data.

        Parameters
        ----------
        x: pandas.DataFrame or numpy.ndarray, shape = [n_samples, n_features]
            Training data. All elements must be able to be converted to float.
        y: pandas.DataFrame, pandas.Series or numpy.ndarray, shape = [n_samples,]
            Target values. If a DataFrame is provided, it is assumed the first column contains target values.
        warm_start: bool (default=False)
            Indicates the optimization should continue using the last individuals of the
            previous `fit` call.
        keep_cache: bool (default=False)
            If True, keep the cache directory and its content after fitting is complete. Otherwise delete it.
        d3m_mode: bool (default=False)
            Signals whether fit is being run in D3M context.  If so, we assume that various
            kinds of preprocessing have already been performed and leave the inputs as D3M Dataframes.
        """
        with self._time_manager.start_activity('preprocessing', activity_meta=['default']):
            y_type = pd.DataFrame if d3m_mode else pd.Series
            self._X, self._y = format_x_y(x, y, y_type)
            self._classes = list(set(self._y))
            if not d3m_mode:
                steps = define_preprocessing_steps(self._X, max_extra_features_created=None, max_categories_for_one_hot=10)
                self._operator_set._safe_compile = partial(compile_individual, preprocessing_steps=steps)

        fit_time = int((1 - self._post_processing.time_fraction) * self._time_manager.total_time_remaining)

        with self._time_manager.start_activity('search', time_limit=fit_time,
                                               activity_meta=[self._search_method.__class__.__name__]):
            self._search_phase(warm_start, timeout=fit_time)

        with self._time_manager.start_activity('postprocess',
                                               time_limit=int(self._time_manager.total_time_remaining),
                                               activity_meta=[self._post_processing.__class__.__name__]):
            best_individuals = list(reversed(sorted(self._final_pop, key=lambda ind: ind.fitness.values)))
            self._post_processing.dynamic_defaults(self)
            self.model = self._post_processing.post_process(
                self._X, self._y, self._time_manager.total_time_remaining, best_individuals
            )
        if not keep_cache:
            log.debug("Deleting cache.")
            self.delete_cache()

    def _set_evaluator(self, timeout: int = 1e6):
        deadline = time.time() + timeout
        evaluate_args = dict(evaluate_pipeline_length=self._regularize_length, X=self._X, y_train=self._y,
                             metrics=self._metrics, cache_dir=self._cache_dir, timeout=self._max_eval_time,
                             deadline=deadline)
        self._operator_set.evaluate = partial(gama.genetic_programming.compilers.scikitlearn.evaluate_individual,
                                              **evaluate_args)

    def _search_phase(self, warm_start: bool = False, timeout: int = 1e6):
        """ Invoke the evolutionary algorithm, populate `final_pop` regardless of termination. """
        if warm_start and self._final_pop is not None:
            pop = self._final_pop
        else:
            if warm_start:
                log.warning('Warm-start enabled but no earlier fit. Using new generated population instead.')
            pop = [self._operator_set.individual() for _ in range(50)]

        self._set_evaluator(timeout)

        try:
            with stopit.ThreadingTimeout(timeout):
                self._search_method.dynamic_defaults(self._X, self._y, timeout)
                self._search_method.search(self._operator_set, start_candidates=pop)
        except KeyboardInterrupt:
            log.info('Search phase terminated because of Keyboard Interrupt.')

        self._final_pop = self._search_method.output
        log.debug([str(i) for i in self._final_pop[:100]])
        log.info(f'Search phase evaluated {len(self._observer._individuals)} individuals.')

    def _initialize_ensemble(self):
        raise NotImplementedError('_initialize_ensemble should be implemented by a child class.')

    def delete_cache(self):
        """ Removes the cache folder and all files associated to this instance.

        Returns
        -------
        None
        """
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

    def evaluation_completed(self, callback_function):
        """ Register a callback function that is called when new evaluation is completed.

        Parameters
        ----------
        callback_function:
            Function to call when a pipeline is evaluated, return values are ignored.
            Expected signature is: Individual -> Any
        """
        self._subscribers['evaluation_completed'].append(callback_function)

    def morris_sensitivity_chart(self, filename: str):
        """ Save a chart of Morris Sensitivity by feature as html file to `filename`. """
        from interpret import preserve
        from interpret.perf import ROC

        predict = self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict
        blackbox_perf = ROC(predict).explain_perf(self._X, self._y, name='Blackbox')
        preserve(blackbox_perf, file_name=filename)
