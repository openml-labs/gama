import logging
import os
import pickle
import time
import uuid
from datetime import datetime

import stopit
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

from gama.genetic_programming.algorithms.metrics import Metric

from gama.genetic_programming.components import Individual, PrimitiveNode
from gama.utilities.logging_utilities import MultiprocessingLogger, log_parseable_event, TOKENS

log = logging.getLogger(__name__)


def primitive_node_to_sklearn(primitive_node: PrimitiveNode) -> object:
    hyperparameters = {terminal.output: terminal.value for terminal in primitive_node._terminals}
    return primitive_node._primitive._identifier(**hyperparameters)


def compile_individual(individual: Individual, parameter_checks=None, preprocessing_steps=None) -> Pipeline:
    steps = [(str(i), primitive_node_to_sklearn(primitive)) for i, primitive in enumerate(individual.primitives)]
    if preprocessing_steps:
        steps = steps + [(str(i), step) for (i, step) in enumerate(preprocessing_steps, start=len(steps))]
    return Pipeline(list(reversed(steps)))


def cross_val_predict_score(estimator, X, y_train, y_score, groups=None, scoring=None, cv=None, n_jobs=1, verbose=0,
                            fit_params=None, pre_dispatch='2*n_jobs'):
    """ Return both the predictions and score of the estimator trained on the data given the cv strategy.
    # TODO: Add reference to underlying sklearn cross_val_predict for parameter descriptions.

    :param estimator: the estimator to evaluate
    :param X:
    :param y_train: target in appropriate format for training (typically (N,))
    :param y_score: target in appropriate format for scoring (typically (N,K) for metrics based on class probabilities,
        (N,) otherwise).
    :param groups:
    :param scoring:
    :param cv:
    :param n_jobs:
    :param verbose:
    :param fit_params:
    :param pre_dispatch:
    :return:
    """
    if isinstance(scoring, Metric):
        metric = scoring
    elif isinstance(scoring, str):
        metric = Metric(scoring)
    else:
        raise ValueError('Parameter `scoring` must be an instance of `str` or `gama.ea.metrics.Metric`, is {}.'
                         .format(type(scoring)))

    method = 'predict_proba' if metric.requires_probabilities else 'predict'
    predictions = cross_val_predict(estimator, X, y_train, groups, cv, n_jobs, verbose, fit_params, pre_dispatch, method)
    score = metric.maximizable_score(y_score, predictions)
    return predictions, score


def object_is_valid_pipeline(o):
    """ Determines if object behaves like a valid scikit-learn pipeline (it must have `fit`, `predict` and `steps`). """
    return (o is not None and
            hasattr(o, 'fit') and
            hasattr(o, 'predict') and
            hasattr(o, 'steps'))


def evaluate_pipeline(pl, X, y_train, y_score, timeout, scoring='accuracy', cv=5, cache_dir=None, logger=None):
    """ Evaluates a pipeline used k-Fold CV. """
    if not logger:
        logger = log

    if not object_is_valid_pipeline(pl):
        return ValueError('Pipeline is not valid. Must not be None and have `fit`, `predict` and `steps`.')

    start_datetime = datetime.now()
    start = time.process_time()
    with stopit.ThreadingTimeout(timeout) as c_mgr:
        try:
            prediction, score = cross_val_predict_score(pl, X, y_train, y_score, cv=cv, scoring=scoring)
        except stopit.TimeoutException:
            # score not actually unused, because exception gets caught by the context manager.
            score = float('-inf')
            raise
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if isinstance(logger, MultiprocessingLogger):
                logger.info('{} encountered while evaluating pipeline.'.format(type(e)))
            else:
                logger.info('{} encountered while evaluating pipeline.'.format(type(e)), exc_info=True)

            single_line_pipeline = str(pl).replace('\n', '')
            log_parseable_event(logger, TOKENS.EVALUATION_ERROR, start_datetime, single_line_pipeline, type(e), e)
            score = -float("inf")

    if cache_dir and score != -float("inf"):
        pl_filename = str(uuid.uuid4())

        try:
            with open(os.path.join(cache_dir, pl_filename + '.pkl'), 'wb') as fh:
                pickle.dump((pl, prediction, score), fh)
        except FileNotFoundError:
            log.warning("File not found while saving predictions. This can happen in the multi-process case if the "
                        "cache gets deleted within `max_eval_time` of the end of the search process.", exc_info=True)

    evaluation_time = time.process_time() - start
    pipeline_length = len(pl.steps)

    if c_mgr.state == c_mgr.INTERRUPTED:
        # A TimeoutException was raised, but not by the context manager.
        # This indicates that the outer context manager (the ea) timed out.
        logger.info("Outer-timeout during evaluation of {}".format(pl))
        raise stopit.utils.TimeoutException()

    if not c_mgr:
        # For now we treat a eval timeout the same way as e.g. NaN exceptions.
        fitness_values = (-float("inf"), start_datetime, timeout, pipeline_length)
        logger.info('Timeout encountered while evaluating pipeline.')

        single_line_pipeline = ''.join(str(pl).split('\n'))
        log_parseable_event(logger, TOKENS.EVALUATION_TIMEOUT, start_datetime, single_line_pipeline)
        logger.debug("Timeout after {}s: {}".format(timeout, pl))
    else:
        fitness_values = (score, start_datetime, evaluation_time, pipeline_length)

    return fitness_values