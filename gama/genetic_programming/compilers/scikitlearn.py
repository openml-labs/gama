from datetime import datetime
import logging
import os
import pickle
import time
from typing import Iterable
import uuid
import numpy as np
import scipy.sparse as sp

import pandas as pd
import stopit
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import cross_val_predict, ShuffleSplit, check_cv
from sklearn.model_selection._validation import _fit_and_predict
from sklearn.pipeline import Pipeline
from sklearn.utils import indexable
from sklearn.utils._joblib import Parallel, delayed
from sklearn.base import clone, is_classifier

from gama.utilities.metrics import Metric
from gama.genetic_programming.components import Individual, PrimitiveNode, Fitness
from gama.logging.utility_functions import MultiprocessingLogger
from gama.logging.machine_logging import TOKENS, log_event

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)

# print("Local scikitlearn module loaded")


def primitive_node_to_sklearn(primitive_node: PrimitiveNode) -> object:
    hyperparameters = {terminal.output: terminal.value for terminal in primitive_node._terminals}
    return primitive_node._primitive.identifier(**hyperparameters)


def compile_individual(individual: Individual, parameter_checks=None, preprocessing_steps: Iterable[object]=None) -> Pipeline:
    steps = [(str(i), primitive_node_to_sklearn(primitive)) for i, primitive in enumerate(individual.primitives)]
    if preprocessing_steps:
        steps = steps + [(str(i), step) for (i, step) in enumerate(reversed(preprocessing_steps), start=len(steps))]
    return Pipeline(list(reversed(steps)))


def cross_val_predict_score(estimator, X, y_train, metrics=None, cvpredict=cross_val_predict, **kwargs):
    """ Return both the predictions and score of the estimator trained on the data given the cv strategy.

    :param y_train: target in appropriate format for training (typically (N,))
    """
    if not all(isinstance(metric, Metric) for metric in metrics):
        raise ValueError('All `metrics` must be an instance of `metrics.Metric`, is {}.'
                         .format([type(metric) for metric in metrics]))

    predictions_are_probabilities = any(metric.requires_probabilities for metric in metrics)
    method = 'predict_proba' if predictions_are_probabilities else 'predict'
    result = cvpredict(estimator, X, y_train, method=method, **kwargs)

    # Ugly hack to support evaluation for time
    if isinstance(result, tuple):
        predictions, y_train = result
    else:
        predictions = result

    if predictions.ndim == 2 and predictions.shape[1] == 1:
        predictions = predictions.squeeze()

    scores = []
    if isinstance(y_train, pd.DataFrame):
        # Metrics want to work with Series, for d3m the target vector is a pd.DataFrame
        y_train = y_train[y_train.columns[0]]

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


def cross_val_predict_timeseries(estimator, X, y=None, groups=None, cv='warn',
                                 verbose=0, fit_params=None, method='predict'):
    X, y, groups = indexable(X, y, groups)
    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    if method in ['decision_function', 'predict_proba', 'predict_log_proba']:
        le = LabelEncoder()
        y = le.fit_transform(y)

    prediction_blocks = []
    y_part = []
    for train, test in cv.split(X, y, groups):
        preds = _fit_and_predict(clone(estimator), X, y, train, test, verbose, fit_params, method)
        prediction_blocks.append(preds)
        y_part.append(test)

    # Concatenate the predictions
    predictions = [pred_block_i for pred_block_i, _ in prediction_blocks]

    # Check for sparse predictions
    if sp.issparse(predictions[0]):
        predictions = sp.vstack(predictions, format=predictions[0].format)
    else:
        predictions = np.concatenate(predictions)
    y_train = np.concatenate(y_part)
    return predictions, y.iloc[y_train, :]


def object_is_valid_pipeline(o):
    """ Determines if object behaves like a valid scikit-learn pipeline (it must have `fit`, `predict` and `steps`). """
    return (o is not None and
            hasattr(o, 'fit') and
            hasattr(o, 'predict') and
            hasattr(o, 'steps'))


def evaluate_individual(individual: Individual, evaluate_pipeline_length, *args, **kwargs):
    log.info("Evaluating individual: %s" % individual)
    (scores, start_datetime, wallclock_time, process_time) = evaluate_pipeline(individual.pipeline, *args, **kwargs)
    if evaluate_pipeline_length:
        scores = (*scores, -len(individual.primitives))
    individual.fitness = Fitness(scores, start_datetime, wallclock_time, process_time)
    return individual


def evaluate_pipeline(pl, X, y_train, timeout, deadline, metrics='accuracy', cv=5, cache_dir=None, logger=None,
                      subsample=None, cvpredict=cross_val_predict):
    """ Evaluates a pipeline used k-Fold CV. """
    if not logger:
        logger = log

    if not object_is_valid_pipeline(pl):
        return ValueError('Pipeline is not valid. Must not be None and have `fit`, `predict` and `steps`.')

    draw_subsample = (isinstance(subsample, int) and subsample < len(y_train))
    scores = tuple([float('-inf')] * len(metrics))
    start_datetime = datetime.now()
    start = time.process_time()

    time_to_deadline = deadline - time.time()
    timeout = min(timeout, time_to_deadline)
    with open('evals.txt', 'a') as fh:
        fh.write(f'{timeout}->{str(pl)}\n')
    with stopit.ThreadingTimeout(timeout) as c_mgr:
        try:
            if draw_subsample:
                idx, _ = next(ShuffleSplit(n_splits=1, train_size=subsample, random_state=0).split(X))
                X, y_train = X.iloc[idx, :], y_train[idx]

            prediction, scores = cross_val_predict_score(pl, X, y_train, cv=cv, metrics=metrics, cvpredict=cvpredict)
            print(scores)
        except stopit.TimeoutException:
            # score not actually unused, because exception gets caught by the context manager.
            raise
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if isinstance(logger, MultiprocessingLogger):

                logger.debug('{} encountered while evaluating pipeline: {}'.format(type(e), str(e)))
                # log.debug('{} encountered while evaluating pipeline: {}'.format(type(e), str(e)), exc_info=True)
            else:
                logger.debug('{} encountered while evaluating pipeline.'.format(type(e)), exc_info=True)
                # log.debug('{} encountered while evaluating pipeline.'.format(type(e)), exc_info=True)

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
