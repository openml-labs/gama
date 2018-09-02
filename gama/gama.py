import random
import logging
import os
from collections import defaultdict
import datetime
import shutil
from functools import partial
import time

import arff
import pandas as pd
import numpy as np
from deap import base, creator, tools, gp
from sklearn.preprocessing import Imputer, OneHotEncoder

import gama.ea.evaluation
from .ea.modified_deap import cxOnePoint
from .ea import automl_gp
from .ea.automl_gp import compile_individual, pset_from_config, generate_valid
from gama.ea.mutation import random_valid_mutation
from .ea.metrics import Metric
from .utilities.observer import Observer

from .ea.async_ea import async_ea
from gama.utilities.generic.stopwatch import Stopwatch
from .utilities import TOKENS, log_parseable_event
from gama.utilities.preprocessing import define_preprocessing_steps

log = logging.getLogger(__name__)

STR_NO_OPTIMAL_PIPELINE = """Gama did not yet establish an optimal pipeline.
                          This can be because `fit` was not yet called, or
                          did not terminate successfully."""
__version__ = '0.1.0'


class Gama(object):
    """ Wrapper for the DEAP toolbox logic surrounding the GP process as well as ensemble construction. """

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
        """

        :param objectives: a tuple which specifies towards which objectives to optimize. The valid metrics depend on
            the type of task. Many scikit-learn metrics are available. Two additional metrics can also be chosen: `size`
            which represents the number of components in the pipeline, and `time` which specifies the time it takes to
            train and validate a model. Currently the maximum arity is 2. Example: ('f1_macro', 'size') or ('f1',)
        :param optimize_strategy: a tuple of the same arity as objectives. Specifies for each objective whether you
            want to maximize (1) or minimize (-1) the objective. Example: (1, -1).
        :param config: a dictionary which specifies available components and their valid hyperparameter settings. For
            more informatio, see `docs\configuration`.
        :param random_state: integer or None. If an integer is passed, this will be the seed for the random number
            generators used in the process. However, with `n_jobs > 1`, there will be randomization introduced by
            multi-processing. For reproducible results, set this and use `n_jobs=1`.
        :param population_size: positive integer. The number of individuals to keep in the population at any one time.
        :param max_total_time: positive integer. The time in seconds that can be used for the `fit` call.
        :param max_eval_time: positive integer or None. THe time in seconds that can be used to evaluate any one single
            individual.
        :param n_jobs: integer. The amount of parallel processes that may be created to speed up `fit`. If this number
            is zero or negative, it will be set to the amount of cores.
        :param verbosity: integer. Does nothing right now. Follow progress of optimization by tracking the log.
        :param cache_dir: string or None. The directory in which to keep the cache during `fit`. In this directory,
            models and their evaluation results will be stored. This facilitates a quick ensemble construction.
        """
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
        if max_total_time is None or max_total_time <= 0:
            error_message = "max_total_time should be greater than zero."
            log.error(error_message + " max_total_time: {}".format(max_total_time))
            raise ValueError(error_message)
        if max_eval_time is not None and max_eval_time <= 0:
            error_message = "max_eval_time should be greater than zero, or None."
            log.error(error_message + " max_eval_time: {}".format(max_eval_time))
            raise ValueError(error_message)

        self._best_pipeline = None
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

    def _preprocess_predict_X(self, X):
        if hasattr(X, 'values') and hasattr(X, 'astype'):
            X = X.astype(np.float64).values
        if np.isnan(X).any():
            log.info("Feature matrix X has been found to contain NaN-labels. Data will be imputed using median.")
            X = self._imputer.transform(X)
        return X

    def predict(self, X):
        """ Predict the target for input X.

        :param X: a 2d numpy array with the length of the second dimension is equal to that of X of `fit`.
        :return: a numpy array with predictions. The array is of shape (N,) where N is the length of the
            first dimension of X.
        """
        raise NotImplemented('predict is implemented by base classes.')

    def fit_arff(self, arff_file_path, *args, **kwargs):
        # load arff
        with open(arff_file_path, 'r') as arff_file:
            arff_dict = arff.load(arff_file)

        attribute_names, data_types = zip(*arff_dict['attributes'])
        data = pd.DataFrame(arff_dict['data'], columns=attribute_names)
        for attribute_name, dtype in arff_dict['attributes']:
            # if dtype.lower() in ['real', 'numeric']:  probably interpreted correctly.
            if isinstance(dtype, list):
                data[attribute_name] = data[attribute_name].astype('category')
            # TODO: add date support

        X, y = data.iloc[:, :-1], data.iloc[:, -1]
        steps = define_preprocessing_steps(X, max_extra_features_created=None, max_categories_for_one_hot=10)
        self._toolbox.register("compile", compile_individual, pset=self._pset, preprocessing_steps=steps)
        self.fit(X, y, skip_preprocess=True, *args, **kwargs)

    def fit(self, X, y, skip_preprocess=False, warm_start=False, auto_ensemble_n=25, restart_=False, keep_cache=False):
        """ Find and fit a model to predict target y from X.

        Various possible machine learning pipelines will be fit to the (X,y) data.
        Using Genetic Programming, the pipelines chosen should lead to gradually
        better models. Pipelines will internally be validated using cross validation.

        After the search termination condition is met, the best found pipeline
        configuration is then used to train a final model on all provided data.

        :param X:
        :param y:
        :param warm_start: bool. Indicates the optimization should continue using the last individuals of the
            previous `fit` call.
        :param auto_ensemble_n: positive integer. The number of models to include in the ensemble which is built
            after the optimizatio process.
        :param restart_: bool. Indicates whether or not the search should be restarted when a specific restart
            criteria is met.
        """

        ensemble_ratio = 0.1  # fraction of time left after preprocessing that reserved for postprocessing

        def restart_criteria():
            restart = self._observer._individuals_since_last_pareto_update > 400
            if restart and restart_:
                log.info("Continuing search with new population.")
                self._observer.reset_current_pareto_front()
            return restart and restart_

        if not skip_preprocess:
            print('preprocess anyway')
            with Stopwatch() as preprocessing_sw:
                X, y = self._preprocess_phase(X, y)
            log.info("Preprocessing took {:.4f}s. Moving on to search phase.".format(preprocessing_sw.elapsed_time))
            log_parseable_event(log, TOKENS.PREPROCESSING_END, preprocessing_sw.elapsed_time)
            time_left = self._max_total_time - preprocessing_sw.elapsed_time
        else:
            print('assigning')
            self._fit_data = (X, y)
            self.X = X
            self.y_train = y
            self._construct_y_score(y)
            time_left = self._max_total_time

        fit_time = int((1 - ensemble_ratio) * time_left)

        with Stopwatch() as search_sw:
            self._search_phase(X, y, warm_start, restart_criteria=restart_criteria, timeout=fit_time)
        log.info("Search phase took {:.4f}s. Moving on to post processing.".format(search_sw.elapsed_time))
        log_parseable_event(log, TOKENS.SEARCH_END, search_sw.elapsed_time)

        time_left = time_left - search_sw.elapsed_time

        with Stopwatch() as post_sw:
            self._postprocess_phase(auto_ensemble_n, timeout=time_left)
        log.info("Postprocessing took {:.4f}s.".format(post_sw.elapsed_time))
        log_parseable_event(log, TOKENS.POSTPROCESSING_END, post_sw.elapsed_time)

        if not keep_cache:
            log.debug("Deleting cache.")
            self.delete_cache()

    def _preprocess_phase(self, X, y):
        """  Preprocess X and y such that scikit-learn pipelines can be evaluated on it.

        Preprocessing currently transforms the input into float64 numpy arrays.
        Any row that has a NaN y-value gets removed. Any remaining NaNs in X get imputed.

        :param X: Input data, DataFrame or numpy array with shape (sample, features)
        :param y: True labels for each sample
        :return: Preprocessed versions of X and y.
        """
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
        # This helps us use a wider variety of algorithms without constructing a grammar.
        # One should note that ideally imputation should not always be done since some methods work well without.
        # Secondly, the way imputation is done can also be dependent on the task. Median is generally not the best.
        self._imputer.fit(X)
        if np.isnan(X).any():
            log.info("Feature matrix X has been found to contain NaN-labels. Data will be imputed using median.")
            X = self._imputer.transform(X)

        self.X = X
        self.y_train = y
        self._construct_y_score(y)

        return X, y

    def _construct_y_score(self, y):
        if Metric(self._scoring_function).requires_probabilities:
            self.y_score = OneHotEncoder().fit_transform(y.reshape(-1, 1)).todense()
        else:
            self.y_score = y

    def _search_phase(self, X, y, warm_start=False, restart_criteria=None, timeout=1e6):
        """ Invoke the evolutionary algorithm, populate `final_pop` regardless of termination. """
        if warm_start and self._final_pop is not None:
            pop = self._final_pop
        else:
            if warm_start:
                log.warning('Warm-start enabled but no earlier fit. Using new generated population instead.')
            pop = self._toolbox.population(n=self._pop_size)

        self._toolbox.register("evaluate", gama.ea.evaluation.evaluate_pipeline,
                               X=self.X, y_train=self.y_train, y_score=self.y_score,
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
        """ Perform any necessary post processing, such as ensemble building. """
        #self._best_pipeline = list(reversed(sorted(self._final_pop, key=lambda ind: ind.fitness.wvalues)))[0]
        #print(self._best_pipeline.fitness.wvalues)
        #self._best_pipeline = self._toolbox.compile(self._best_pipeline)
        #X, y = self._fit_data
        #self._best_pipeline.fit(X, self.y_train)
        self._build_fit_ensemble(n, timeout=timeout)

    def _initialize_ensemble(self):
        raise NotImplementedError('_initialize_ensemble should be implemented by a child class.')

    def _build_fit_ensemble(self, ensemble_size, timeout):
        start_build = time.time()
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

        X, y = self._fit_data
        self.ensemble.fit(X, y, timeout=timeout)

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
