from datetime import datetime
import logging
import time
from typing import Iterable, Callable, Tuple, Optional

import stopit
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline

from gama.utilities.evaluation_library import Evaluation
from gama.utilities.generic.stopwatch import Stopwatch
from gama.utilities.metrics import Metric
from gama.genetic_programming.components import Individual, PrimitiveNode, Fitness
from gama.logging.utility_functions import MultiprocessingLogger
from gama.logging.machine_logging import TOKENS, log_event

log = logging.getLogger(__name__)


def primitive_node_to_sklearn(primitive_node: PrimitiveNode) -> object:
    hyperparameters = {
        terminal.output: terminal.value for terminal in primitive_node._terminals
    }
    return primitive_node._primitive.identifier(**hyperparameters)


def compile_individual(
    individual: Individual,
    parameter_checks=None,
    preprocessing_steps: Iterable[object] = None,
) -> Pipeline:
    steps = [
        (str(i), primitive_node_to_sklearn(primitive))
        for i, primitive in enumerate(individual.primitives)
    ]
    if preprocessing_steps:
        steps = steps + [
            (str(i), step)
            for (i, step) in enumerate(reversed(preprocessing_steps), start=len(steps))
        ]
    return Pipeline(list(reversed(steps)))


def cross_val_train_predict(
    estimator, x, y, predict_method: str = "predict", cv: int = 5
):
    """ Return fit estimators and predictions of each (Stratified) fold. """
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
        fold_predict = getattr(fold_estimator, predict_method)

        fold_estimator.fit(x_train, y_train)
        estimators.append(fold_estimator)
        fold_prediction = fold_predict(x_test)

        if predictions is None:
            if fold_prediction.ndim == 2:
                predictions = np.empty(shape=(len(y), fold_prediction.shape[1]))
            else:
                predictions = np.empty(shape=(len(y),))

        predictions[test] = fold_prediction

    return predictions, estimators


def cross_val_predict_score(estimator, X, y_train, metrics=None, **kwargs):
    """ Return scores, fit estimators and predictions of each (Stratified) fold.

    :param y_train: target in appropriate format for training (typically (N,))
    """
    if not all(isinstance(metric, Metric) for metric in metrics):
        raise ValueError(
            f"All `metrics` must be an instance of `metrics.Metric`, "
            f"is {[type(metric) for metric in metrics]}."
        )

    predictions_are_probabilities = any(m.requires_probabilities for m in metrics)
    method = "predict_proba" if predictions_are_probabilities else "predict"
    # predictions = cross_val_predict(estimator, X, y_train, method=method, **kwargs)
    predictions, estimators = cross_val_train_predict(
        estimator, X, y_train, predict_method=method
    )

    if predictions.ndim == 2 and predictions.shape[1] == 1:
        predictions = predictions.squeeze()

    scores = []
    for metric in metrics:
        if metric.requires_probabilities:
            # `predictions` are of shape (N,K), ground truth to be formatted accordingly
            y_ohe = (
                OneHotEncoder().fit_transform(y_train.values.reshape(-1, 1)).toarray()
            )
            scores.append(metric.maximizable_score(y_ohe, predictions))
        elif predictions_are_probabilities:
            # Metric requires no probabilities, but probabilities were predicted.
            scores.append(metric.maximizable_score(y_train, predictions.argmax(axis=1)))
        else:
            # No metric requires probabilities, so `predictions` is an array of labels.
            scores.append(metric.maximizable_score(y_train, predictions))

    return predictions, scores, estimators


def object_is_valid_pipeline(o):
    """ Determines if object behaves like a scikit-learn pipeline. """
    return (
        o is not None
        and hasattr(o, "fit")
        and hasattr(o, "predict")
        and hasattr(o, "steps")
    )


def evaluate_pipeline(
    pipeline,
    x,
    y_train,
    timeout: float,
    metrics="accuracy",
    cv=5,
    logger=None,
    subsample=None,
) -> Tuple:
    """ Score `pipeline` with k-fold CV according to `metrics` on (a subsample of) X, y

    Returns
    -------
    Tuple:
        prediction: np.ndarray if successful, None if not
        scores: tuple with one float per metric, each value is -inf on fail.
        estimators: list of fitted pipelines if successful, None if not
        error: None if successful, otherwise an Exception
    """
    if not logger:
        logger = log
    if not object_is_valid_pipeline(pipeline):
        raise TypeError(f"Pipeline must not be None and requires fit, predict, steps.")

    start_datetime = datetime.now()
    prediction, estimators = None, None
    # default score for e.g. timeout or failure
    scores = tuple([float("-inf")] * len(metrics))

    with stopit.ThreadingTimeout(timeout) as c_mgr:
        try:
            if isinstance(subsample, int) and subsample < len(y_train):
                sampler = ShuffleSplit(n_splits=1, train_size=subsample, random_state=0)
                idx, _ = next(sampler.split(x))
                x, y_train = x.iloc[idx, :], y_train[idx]

            prediction, scores, estimators = cross_val_predict_score(
                pipeline, x, y_train, cv=cv, metrics=metrics
            )
        except stopit.TimeoutException:
            # This exception is handled by the ThreadingTimeout context manager.
            raise
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if isinstance(logger, MultiprocessingLogger):
                logger.debug(f"{type(e)} raised during evaluation.")
            else:
                logger.debug(f"{type(e)} raised during evaluation.", exc_info=True)

            single_line_pipeline = str(pipeline).replace("\n", "")
            log_event(
                logger,
                TOKENS.EVALUATION_ERROR,
                start_datetime,
                single_line_pipeline,
                type(e),
                e,
            )
            return prediction, scores, None, e

    if c_mgr.state == c_mgr.INTERRUPTED:
        # A TimeoutException was raised, but not by the context manager.
        # This indicates that the outer context manager (the ea) timed out.
        logger.info("Outer-timeout during evaluation of {}".format(pipeline))
        raise stopit.utils.TimeoutException()

    if not c_mgr:
        # For now we treat an eval timeout the same way as
        # e.g. NaN exceptions and use the default score.
        single_line_pipeline = str(pipeline).replace("\n", "")
        log_event(
            logger, TOKENS.EVALUATION_TIMEOUT, start_datetime, single_line_pipeline
        )
        logger.debug(f"Timeout after {timeout}s: {pipeline}")

    return prediction, scores, estimators, None


def evaluate_individual(
    individual: Individual,
    evaluate_pipeline: Callable,
    timeout: float = 1e6,
    deadline: Optional[float] = None,
    add_length_to_score: bool = True,
    **kwargs,
) -> Evaluation:
    """ Evaluate the pipeline specified by individual, and record

    Parameters
    ----------
    individual: Individual
        Blueprint for the pipeline to evaluate.
    evaluate_pipeline: Callable
        Function which takes the pipeline and produces validation predictions,
        scores, estimators and errors.
    timeout: float (default=1e6)
        Maximum time in seconds that the evaluation is allowed to take.
        Don't depend on high accuracy.
        A shorter timeout is imposed if `deadline` is in less than `timeout` seconds.
    deadline: float, optional
        A time in seconds since epoch.
        Cut off evaluation at `deadline` even if `timeout` seconds have not yet elapsed.
    add_length_to_score: bool (default=True)
        Add the length of the individual to the score result of the evaluation.
    **kwargs: Dict, optional (default=None)
        Passed to `evaluate_pipeline` function.

    Returns
    -------
    Evaluation

    """
    result = Evaluation(individual)
    result.start_time = datetime.now()

    if deadline is not None:
        time_to_deadline = deadline - time.time()
        timeout = min(timeout, time_to_deadline)

    with Stopwatch() as wall_time, Stopwatch(time.process_time) as process_time:
        evaluation = evaluate_pipeline(individual.pipeline, timeout=timeout, **kwargs)
        result.predictions, result.score, result.estimators, result.error = evaluation
    result.duration = wall_time.elapsed_time

    if add_length_to_score:
        result.score = (*result.score, -len(individual.primitives))
    individual.fitness = Fitness(
        result.score,
        result.start_time,
        wall_time.elapsed_time,
        process_time.elapsed_time,
    )

    return result
