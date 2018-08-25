import os
import pickle
import time
import uuid

import stopit
from sklearn.model_selection import cross_val_predict

from gama.ea.automl_gp import log
from gama.ea.metrics import Metric
from gama.utilities import log_parseable_event, TOKENS
from gama.utilities.mp_logger import MultiprocessingLogger


def cross_val_predict_score(estimator, X, y_train, y_score, groups=None, scoring=None, cv=None, n_jobs=1, verbose=0,
                            fit_params=None, pre_dispatch='2*n_jobs'):
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

            single_line_pipeline = str(pl).replace(old='\n', new='')
            log_parseable_event(logger, TOKENS.EVALUATION_ERROR, single_line_pipeline, type(e), e)
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
        fitness_values = (-float("inf"), timeout, pipeline_length)
        logger.info('Timeout encountered while evaluating pipeline.')

        single_line_pipeline = ''.join(str(pl).split('\n'))
        log_parseable_event(logger, TOKENS.EVALUATION_TIMEOUT, single_line_pipeline)
        logger.debug("Timeout after {}s: {}".format(timeout, pl))
    else:
        fitness_values = (score, evaluation_time, pipeline_length)

    return fitness_values
