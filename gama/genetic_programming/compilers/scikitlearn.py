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


def cross_val_train_predict(estimator, x, y, method: str = 'predict', cv: int = 5):
    """ Perform (Stratified)KFold returning the trained estimators and predictions of each fold. """
    from sklearn.base import clone, is_classifier
    from sklearn.model_selection._split import check_cv
    from sklearn.utils.metaestimators import _safe_split
    import numpy as np

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    estimators = []
    predictions = None
    for train, test in cv.split(x, y):
        x_train, y_train = _safe_split(estimator, x, y, train)
        x_test, _ = _safe_split(estimator, x, y, test, train)

        fold_estimator = clone(estimator)
        fold_predict = getattr(fold_estimator, method)

        fold_estimator.fit(x_train, y_train)
        estimators.append(fold_estimator)
        fold_prediction = fold_predict(x_test)

        if predictions is None:
            if fold_prediction.ndim == 2:
                predictions = np.empty(shape=(len(y), fold_prediction.shape[1]))
            else:
                predictions = np.empty(shape=(len(y),))

        predictions[test] = fold_prediction

    return {'predictions': predictions, 'estimators': estimators}


def cross_val_predict_score(estimator, X, y_train, metrics=None, cvpredict=cross_val_train_predict, **kwargs):
    """ Return both the predictions and score of the estimator trained on the data given the cv strategy.

    :param y_train: target in appropriate format for training (typically (N,))
    """
    if not all(isinstance(metric, Metric) for metric in metrics):
        raise ValueError('All `metrics` must be an instance of `metrics.Metric`, is {}.'
                         .format([type(metric) for metric in metrics]))

    predictions_are_probabilities = any(metric.requires_probabilities for metric in metrics)
    method = 'predict_proba' if predictions_are_probabilities else 'predict'
    result = cvpredict(estimator, X, y_train, method=method, **kwargs)
    predictions = result['predictions']
    y_train = result.get('y_train', y_train)
    estimators = result.get('estimators', None)
    # time series = predictions, y_train
    # other  = predictions, estimators

    # Ugly hack to support evaluation for time
    # if isinstance(result, tuple):
    #     predictions, y_train = result
    # else:
    #     predictions = result
    # predictions = cross_val_predict(estimator, X, y_train, method=method, **kwargs)

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

    return predictions, scores, estimators


def cross_val_predict_timeseries(estimator, X, y=None, groups=None, cv='warn',
                                 n_jobs=None, verbose=0, fit_params=None,
                                 pre_dispatch='2*n_jobs', method='predict'):
    """Generate cross-validated estimates for each input data point

    It is not appropriate to pass these predictions into an evaluation
    metric. Use :func:`cross_validate` to measure generalization error.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.

    X : array-like
        The data to fit. Can be, for example a list, or an array at least 2d.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like, with shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.20
            ``cv`` default value if None will change from 3-fold to 5-fold
            in v0.22.

    n_jobs : int or None, optional (default=None)
        The number of CPUs to use to do the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : integer, optional
        The verbosity level.

    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    method : string, optional, default: 'predict'
        Invokes the passed method name of the passed estimator. For
        method='predict_proba', the columns correspond to the classes
        in sorted order.

    Returns
    -------
    predictions : ndarray
        This is the result of calling ``method``
    y
        The subset of y tested against

    See also
    --------
    cross_val_score : calculate score for each CV split

    cross_validate : calculate one or more scores and timings for each CV split

    Notes
    -----
    In the case that one or more classes are absent in a training portion, a
    default score needs to be assigned to all instances for that class if
    ``method`` produces columns per class, as in {'decision_function',
    'predict_proba', 'predict_log_proba'}.  For ``predict_proba`` this value is
    0.  In order to ensure finite output, we approximate negative infinity by
    the minimum finite float value for the dtype in other cases.

    """
    X, y, groups = indexable(X, y, groups)
    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    if method in ['decision_function', 'predict_proba', 'predict_log_proba']:
        le = LabelEncoder()
        y = le.fit_transform(y)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    prediction_blocks = []
    y_part = []
    for train, test in cv.split(X, y, groups):
        preds = _fit_and_predict(clone(estimator), X, y, train, test, verbose, fit_params, method)
        prediction_blocks.append(preds)
        y_part.append(test)
#    prediction_blocks = parallel(delayed(_fit_and_predict)(
#            clone(estimator), X, y, train, test, verbose, fit_params, method)
#        for train, test in cv.split(X, y, groups))

    # Concatenate the predictions
    predictions = [pred_block_i for pred_block_i, _ in prediction_blocks]
#    test_indices = np.concatenate([indices_i
#                                   for _, indices_i in prediction_blocks])

    #inv_test_indices = np.empty(len(test_indices), dtype=int)
    #inv_test_indices[test_indices] = np.arange(len(test_indices))

    # Check for sparse predictions
    if sp.issparse(predictions[0]):
        predictions = sp.vstack(predictions, format=predictions[0].format)
    else:
        predictions = np.concatenate(predictions)
    y_train = np.concatenate(y_part)
    return {'predictions': predictions, 'y_train': y.iloc[y_train, :]}


def object_is_valid_pipeline(o):
    """ Determines if object behaves like a valid scikit-learn pipeline (it must have `fit`, `predict` and `steps`). """
    return (o is not None and
            hasattr(o, 'fit') and
            hasattr(o, 'predict') and
            hasattr(o, 'steps'))


def evaluate_individual(individual: Individual, evaluate_pipeline_length, *args, **kwargs):
    (scores, start_datetime, wallclock_time, process_time) = evaluate_pipeline(individual, *args, **kwargs)
    if evaluate_pipeline_length:
        scores = (*scores, -len(individual.primitives))
    individual.fitness = Fitness(scores, start_datetime, wallclock_time, process_time)
    return individual


def evaluate_pipeline(individual, X, y_train, timeout, deadline, metrics='accuracy', cv=5, cache_dir=None, logger=None,
                      subsample=None, cvpredict=cross_val_predict):
    """ Evaluates a pipeline used k-Fold CV. """
    pl = individual.pipeline
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
            prediction, scores, estimators = cross_val_predict_score(pl, X, y_train, cv=cv, metrics=metrics, cvpredict=cvpredict)
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
                pickle.dump((individual, estimators, prediction, scores), fh)
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
        single_line_pipeline = ''.join(str(pl).split('\n'))
        log_event(logger, TOKENS.EVALUATION_TIMEOUT, start_datetime, single_line_pipeline)
        logger.debug("Timeout after {}s: {}".format(timeout, pl))

    fitness_values = (scores, start_datetime, wallclock_time, process_time)
    return fitness_values
