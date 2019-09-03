import logging
import os
import pickle
import time
from typing import Iterable
import uuid
from datetime import datetime

import stopit
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_predict, ShuffleSplit
from sklearn.pipeline import Pipeline

from gama.utilities.metrics import Metric
from gama.genetic_programming.components import Individual, PrimitiveNode, Fitness
from gama.logging.utility_functions import MultiprocessingLogger
from gama.logging.machine_logging import TOKENS, log_event

log = logging.getLogger(__name__)


def primitive_node_to_sklearn(primitive_node: PrimitiveNode) -> object:
    hyperparameters = {terminal.output: terminal.value for terminal in primitive_node._terminals}
    return primitive_node._primitive.identifier(**hyperparameters)


def compile_individual(individual: Individual, parameter_checks=None, preprocessing_steps: Iterable[object]=None) -> Pipeline:
    steps = [(str(i), primitive_node_to_sklearn(primitive)) for i, primitive in enumerate(individual.primitives)]
    if preprocessing_steps:
        steps = steps + [(str(i), step) for (i, step) in enumerate(reversed(preprocessing_steps), start=len(steps))]
    return Pipeline(list(reversed(steps)))


def cross_val_predict_score(estimator, X, y_train, metrics=None, **kwargs):
    """ Return both the predictions and score of the estimator trained on the data given the cv strategy.

    :param y_train: target in appropriate format for training (typically (N,))
    """
    if not all(isinstance(metric, Metric) for metric in metrics):
        raise ValueError('All `metrics` must be an instance of `metrics.Metric`, is {}.'
                         .format([type(metric) for metric in metrics]))

    predictions_are_probabilities = any(metric.requires_probabilities for metric in metrics)
    method = 'predict_proba' if predictions_are_probabilities else 'predict'
    predictions = cross_val_predict(estimator, X, y_train, method=method, **kwargs)

    if predictions.ndim == 2 and predictions.shape[1] == 1:
        predictions = predictions.squeeze()

    scores = []
    for metric in metrics:
        if metric.requires_probabilities:
            # `predictions` are of shape (N,K) and the ground truth should be formatted accordingly
            y_ohe = OneHotEncoder().fit_transform(y_train.values.reshape(-1, 1)).toarray()
            scores.append(metric.maximizable_score(y_ohe, predictions))
        elif predictions_are_probabilities:
            # Metric requires no probabilities, but probabilities were predicted.
            scores.append(metric.maximizable_score(y_train, predictions.argmax(axis=1)))
        else:
            # No metric requires probabilities, so `predictions` is an array of labels.
            scores.append(metric.maximizable_score(y_train, predictions))

    return predictions, scores


def object_is_valid_pipeline(o):
    """ Determines if object behaves like a valid scikit-learn pipeline (it must have `fit`, `predict` and `steps`). """
    return (o is not None and
            hasattr(o, 'fit') and
            hasattr(o, 'predict') and
            hasattr(o, 'steps'))


def evaluate_individual(individual: Individual, evaluate_pipeline_length, *args, **kwargs):
    (scores, start_datetime, wallclock_time, process_time) = evaluate_pipeline(individual.pipeline, *args, **kwargs)
    if evaluate_pipeline_length:
        scores = (*scores, -len(individual.primitives))
    individual.fitness = Fitness(scores, start_datetime, wallclock_time, process_time)
    return individual


def evaluate_pipeline(pl, X, y_train, timeout, metrics='accuracy', cv=5, cache_dir=None, logger=None, subsample=None):
    """ Evaluates a pipeline used k-Fold CV. """
    if not logger:
        logger = log

    if not object_is_valid_pipeline(pl):
        return ValueError('Pipeline is not valid. Must not be None and have `fit`, `predict` and `steps`.')

    draw_subsample = (isinstance(subsample, int) and subsample < len(y_train))
    scores = tuple([float('-inf')] * len(metrics))
    start_datetime = datetime.now()
    start = time.process_time()
    with stopit.ThreadingTimeout(timeout) as c_mgr:
        try:
            if draw_subsample:
                idx, _ = next(ShuffleSplit(n_splits=1, train_size=subsample, random_state=0).split(X))
                X, y_train = X.iloc[idx, :], y_train[idx]

            prediction, scores = cross_val_predict_score(pl, X, y_train, cv=cv, metrics=metrics)
        except stopit.TimeoutException:
            # score not actually unused, because exception gets caught by the context manager.
            raise
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if isinstance(logger, MultiprocessingLogger):
                logger.debug('{} encountered while evaluating pipeline.'.format(type(e)))
            else:
                logger.debug('{} encountered while evaluating pipeline.'.format(type(e)), exc_info=True)

            single_line_pipeline = str(pl).replace('\n', '')
            log_event(logger, TOKENS.EVALUATION_ERROR, start_datetime, single_line_pipeline, type(e), e)

    if cache_dir and -float("inf") not in scores and not draw_subsample:
        pl_filename = str(uuid.uuid4())
        try:
            with open(os.path.join(cache_dir, pl_filename + '.pkl'), 'wb') as fh:
                pickle.dump((pl, prediction, scores), fh)
        except FileNotFoundError:
            log.debug("File not found while saving predictions. This can happen in the multi-process case if the "
                      "cache gets deleted within `max_eval_time` of the end of the search process.", exc_info=True)

    process_time = time.process_time() - start
    wallclock_time = (datetime.now() - start_datetime).total_seconds()

    if c_mgr.state == c_mgr.INTERRUPTED:
        # A TimeoutException was raised, but not by the context manager.
        # This indicates that the outer context manager (the ea) timed out.
        logger.info("Outer-timeout during evaluation of {}".format(pl))
        raise stopit.utils.TimeoutException()

    if not c_mgr:
        # For now we treat an eval timeout the same way as e.g. NaN exceptions and use the default score.
        logger.info('Timeout encountered while evaluating pipeline.')
        single_line_pipeline = ''.join(str(pl).split('\n'))
        log_event(logger, TOKENS.EVALUATION_TIMEOUT, start_datetime, single_line_pipeline)
        logger.debug("Timeout after {}s: {}".format(timeout, pl))

    fitness_values = (scores, start_datetime, wallclock_time, process_time)
    return fitness_values
