from abc import ABC
from collections import defaultdict
from functools import partial
import logging
import multiprocessing
import os
import random
import time
from typing import Union, Tuple, Optional, Dict, Type, List
import warnings

import pandas as pd
import numpy as np
import stopit
from sklearn.pipeline import Pipeline

import gama.genetic_programming.compilers.scikitlearn
from gama.logging.machine_logging import log_event, TOKENS
from gama.search_methods.base_search import BaseSearch
from gama.utilities.evaluation_library import EvaluationLibrary, Evaluation
from gama.utilities.metrics import scoring_to_metric

from gama.__version__ import __version__
from gama.data import X_y_from_arff, format_x_y
from gama.search_methods.async_ea import AsyncEA
from gama.utilities.generic.timekeeper import TimeKeeper
from gama.logging.utility_functions import register_stream_log, register_file_log
from gama.utilities.preprocessing import define_preprocessing_steps, basic_encoding, basic_pipeline_extension
from gama.genetic_programming.mutation import random_valid_mutation_in_place
from gama.genetic_programming.crossover import random_crossover
from gama.genetic_programming.selection import create_from_population, eliminate_from_pareto
from gama.genetic_programming.operations import create_random_expression
from gama.configuration.parser import pset_from_config
from gama.genetic_programming.operator_set import OperatorSet
from gama.genetic_programming.compilers.scikitlearn import compile_individual
from gama.postprocessing import BestFitPostProcessing, BasePostProcessing, EnsemblePostProcessing
from gama.utilities.generic.async_evaluator import AsyncEvaluator
from gama.utilities.metrics import Metric

log = logging.getLogger(__name__)

STR_NO_OPTIMAL_PIPELINE = """Gama did not yet establish an optimal pipeline.
                          This can be because `fit` was not yet called, or
                          did not terminate successfully."""

for module_to_ignore in ["sklearn", "numpy"]:
    warnings.filterwarnings("ignore", module=module_to_ignore)


class Gama(ABC):
    """ Wrapper for the toolbox logic surrounding executing the AutoML pipeline. """

    def __init__(self,
                 scoring: Union[str, Metric, Tuple[Union[str, Metric], ...]] = 'filled_in_by_child_class',
                 regularize_length: bool = True,
                 max_pipeline_length: Optional[int] = None,
                 config: Dict = None,
                 random_state: Optional[int] = None,
                 max_total_time: int = 3600,
                 max_eval_time: Optional[int] = None,
                 n_jobs: Optional[int] = None,
                 verbosity: int = logging.WARNING,
                 keep_analysis_log: Optional[str] = 'gama.log',
                 search_method: BaseSearch = AsyncEA(),
                 post_processing_method: BasePostProcessing = BestFitPostProcessing()):
        """

        Parameters
        ----------
        scoring: str, Metric or Tuple
            Specifies the/all metric(s) to optimize towards. A string will be converted to Metric. A tuple must
            specify each metric with the same type (i.e. all str or all Metric). See :ref:`Metrics` for built-in
            metrics.

        regularize_length: bool
            If True, add pipeline length as an optimization metric (preferring short over long).

        max_pipeline_length: int, optional (default=None)
            If set, limit the maximum number of steps in any evaluated pipeline. Encoding and imputation are excluded.

        config: a dictionary which specifies available components and their valid hyperparameter settings
            For more information, see :ref:`search_space_configuration`.

        random_state:  int, optional (default=None)
            If an integer is passed, this will be the seed for the random number generators used in the process.
            However, with `n_jobs > 1`, there will be randomization introduced by multi-processing.
            For reproducible results, set this and use `n_jobs=1`.

        max_total_time: positive int (default=3600)
            Time in seconds that can be used for the `fit` call.

        max_eval_time: positive int, optional (default=None)
            Time in seconds that can be used to evaluate any one single individual.
            If None, set to 0.1 * max_total_time.

        n_jobs: int, optional (default=None)
            The amount of parallel processes that may be created to speed up `fit`.
            Accepted values are positive integers, -1 or None.
            If -1 is specified, multiprocessing.cpu_count() processes are created.
            If None is specified, multiprocessing.cpu_count() / 2 processes are created.

        verbosity: int (default=logging.WARNING)
            Sets the level of log messages to be automatically output to terminal.

        keep_analysis_log: str, optional (default='gama.log')
            If non-empty str, specifies the path (and name) where the log should be stored, e.g. /output/gama.log.
            If `None`, no log is stored.

        search_method: BaseSearch (default=AsyncEA())
            Search method to use to find good pipelines. Should be instantiated.

        post_processing_method: BasePostProcessing (default=BestFitPostProcessing())
            Post-processing method to create a model after the search phase. Should be instantiated.

        """
        register_stream_log(verbosity)
        if keep_analysis_log is not None:
            register_file_log(keep_analysis_log)

        if keep_analysis_log is not None and not os.path.isabs(keep_analysis_log):
            keep_analysis_log = os.path.abspath(keep_analysis_log)

        arguments = ','.join(['{}={}'.format(k, v) for (k, v) in locals().items()
                              if k not in ['self', 'config', 'gamalog', 'file_handler', 'stdout_streamhandler']])
        log.info('Using GAMA version {}.'.format(__version__))
        log.info('{}({})'.format(self.__class__.__name__, arguments))
        log_event(log, TOKENS.INIT, arguments)

        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count() // 2
            log.debug('n_jobs defaulted to %d', n_jobs)

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

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        self._X: Optional[pd.DataFrame] = None
        self._y: Optional[pd.DataFrame] = None
        self._basic_encoding_pipeline: Optional[Pipeline] = None
        self._inferred_dtypes: List[Type] = []
        self.model: object = None
        self._final_pop = None

        self._subscribers = defaultdict(list)
        if isinstance(post_processing_method, EnsemblePostProcessing):
            self._evaluation_library = EvaluationLibrary(
                m=post_processing_method.hyperparameters['max_models'],
                n=post_processing_method.hyperparameters['hillclimb_size'],
            )
        else:
            # Don't keep memory-heavy evaluation meta-data (predictions, estimators)
            self._evaluation_library = EvaluationLibrary(m=0)
        self.evaluation_completed(self._evaluation_library.save_evaluation)

        self._pset, parameter_checks = pset_from_config(config)

        max_start_length = 3 if max_pipeline_length is None else max_pipeline_length
        self._operator_set = OperatorSet(
            mutate=partial(random_valid_mutation_in_place, primitive_set=self._pset, max_length=max_pipeline_length),
            mate=partial(random_crossover, max_length=max_pipeline_length),
            create_from_population=partial(create_from_population, cxpb=0.2, mutpb=0.8),
            create_new=partial(create_random_expression, primitive_set=self._pset, max_length=max_start_length),
            compile_=compile_individual,
            eliminate=eliminate_from_pareto,
            evaluate_callback=self._on_evaluation_completed
        )

    def _np_to_matching_dataframe(self, x: np.ndarray) -> pd.DataFrame:
        """ Format the numpy array to a dataframe whose column types match the training data. """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"Expected x to be of type 'numpy.ndarray' not {type(x)}.")

        x = pd.DataFrame(x)
        for i, dtype in enumerate(self._inferred_dtypes):
            x[i] = x[i].astype(dtype)
        return x

    def _prepare_for_prediction(self, x):
        if isinstance(x, np.ndarray):
            x = self._np_to_matching_dataframe(x)
        x = self._basic_encoding_pipeline.transform(x)
        return x

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
        x = self._prepare_for_prediction(x)
        return self._predict(x)

    def predict_arff(self,
                     arff_file_path: str,
                     target_column: Optional[str] = None,
                     encoding: Optional[str] = None) -> np.ndarray:
        """ Predict the target for input found in the ARFF file.

        Parameters
        ----------
        arff_file_path: str
            An ARFF file with the same columns as the one that used in fit.
            Target column must be present in file, but its values are ignored (can be '?').
        target_column: str, optional (default=None)
            Specifies which column the model should predict.
            If left None, the last column is taken to be the target.
        encoding: str, optional
            Encoding of the ARFF file.

        Returns
        -------
        numpy.ndarray
            array with predictions for each row in the ARFF file.
        """
        x, _ = X_y_from_arff(arff_file_path, split_column=target_column, encoding=encoding)
        x = self._prepare_for_prediction(x)
        return self._predict(x)

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

    def score_arff(self,
                   arff_file_path: str,
                   target_column: Optional[str] = None,
                   encoding: Optional[str] = None
                   ) -> float:
        """ Calculate the score of the model according to the `scoring` metric and input in the ARFF file.

        Parameters
        ----------
        arff_file_path: str
            An ARFF file with which to calculate the score.
        target_column: str, optional (default=None)
            Specifies which column the model should predict.
            If left None, the last column is taken to be the target.
        encoding: str, optional
            Encoding of the ARFF file.

        Returns
        -------
        float
            The score obtained on the given test data according to the `scoring` metric.
        """
        X, y = X_y_from_arff(arff_file_path, split_column=target_column, encoding=encoding)
        return self.score(X, y)

    def fit_arff(self,
                 arff_file_path: str,
                 target_column: Optional[str] = None,
                 encoding: Optional[str] = None,
                 *args, **kwargs
                 ) -> None:
        """ Find and fit a model to predict the target column (last) from other columns.

        Parameters
        ----------
        arff_file_path: str
            Path to an ARFF file containing the training data.
        target_column: str, optional (default=None)
            Specifies which column the model should predict.
            If left None, the last column is taken to be the target.
        encoding: str, optional
            Encoding of the ARFF file.

        """
        X, y = X_y_from_arff(arff_file_path, split_column=target_column, encoding=encoding)
        self.fit(X, y, *args, **kwargs)

    def fit(self,
            x: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.DataFrame, pd.Series, np.ndarray],
            warm_start: bool = False) -> None:
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
        """

        with self._time_manager.start_activity('preprocessing', activity_meta=['default']):
            x, self._y = format_x_y(x, y)
            self._inferred_dtypes = x.dtypes
            self._X, self._basic_encoding_pipeline = basic_encoding(x)
            steps = basic_pipeline_extension(self._X)
            #  steps = define_preprocessing_steps(self._X, max_extra_features_created=None, max_categories_for_one_hot=10)
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

    def _search_phase(self, warm_start: bool = False, timeout: int = 1e6):
        """ Invoke the evolutionary algorithm, populate `final_pop` regardless of termination. """
        if warm_start and self._final_pop is not None:
            pop = [ind for ind in self._final_pop]
        else:
            if warm_start:
                log.warning('Warm-start enabled but no earlier fit. Using new generated population instead.')
            pop = [self._operator_set.individual() for _ in range(50)]

        deadline = time.time() + timeout

        evaluate_pipeline = partial(gama.genetic_programming.compilers.scikitlearn.evaluate_pipeline,
                                    X=self._X, y_train=self._y, metrics=self._metrics)
        self._operator_set.evaluate = partial(gama.genetic_programming.compilers.scikitlearn.evaluate_individual,
                                              evaluate_pipeline=evaluate_pipeline,
                                              timeout=self._max_eval_time, deadline=deadline,
                                              add_length_to_score=self._regularize_length)

        try:
            with stopit.ThreadingTimeout(timeout):
                self._search_method.dynamic_defaults(self._X, self._y, timeout)
                self._search_method.search(self._operator_set, start_candidates=pop)
        except KeyboardInterrupt:
            log.info('Search phase terminated because of Keyboard Interrupt.')

        self._final_pop = self._search_method.output
        log.debug([str(i) for i in self._final_pop[:100]])
        log.info(f'Search phase evaluated {len(self._evaluation_library.evaluations)} individuals.')

    def export_script(self, file: str = 'gama_pipeline.py', raise_if_exists: bool = False):
        """ Exports a Python script which sets up the best found pipeline. Can only be called after `fit`.

        Example
        -------
        After the AutoML search process has completed (i.e. `fit` has been called), the model which
        has been found by GAMA may be exported to a Python file. The Python file will define the found
        pipeline or ensemble.

        .. code-block:: python

            automl = GamaClassifier()
            automl.fit(X, y)
            automl.export_script('my_pipeline_script.py')

        The resulting script will define a variable `pipeline` or `ensemble`, depending on the post-processing
        method that was used after search.

        Parameters
        ----------
        file: str (default='gama_pipeline.py')
            Desired filename of the exported Python script.
        raise_if_exists: bool (default=False)
            If True, raise an error if the file already exists.
            If False, overwrite `file` if it already exists.
        """
        if self.model is None:
            raise RuntimeError(STR_NO_OPTIMAL_PIPELINE)
        if raise_if_exists and os.path.isfile(file):
            raise FileExistsError(f"File {file} already exists.")

        script_text = self._post_processing.to_code(self._basic_encoding_pipeline)
        with open(file, 'w') as fh:
            fh.write(script_text)

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

    def _on_evaluation_completed(self, evaluation: Evaluation):
        for callback in self._subscribers['evaluation_completed']:
            self._safe_outside_call(partial(callback, evaluation))

    def evaluation_completed(self, callback_function):
        """ Register a callback function that is called when new evaluation is completed.

        Parameters
        ----------
        callback_function:
            Function to call when a pipeline is evaluated, return values are ignored.
            Expected signature is: Evaluation -> Any
        """
        self._subscribers['evaluation_completed'].append(callback_function)
