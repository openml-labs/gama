from datetime import datetime
import logging
import os
import time
from typing import Callable, Tuple, Optional, Sequence

import stopit
from sklearn.base import TransformerMixin, is_classifier
from sklearn.model_selection import ShuffleSplit, cross_validate, check_cv
from sklearn.pipeline import Pipeline

from gama.utilities.evaluation_library import Evaluation
from gama.utilities.generic.stopwatch import Stopwatch
import numpy as np
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
    preprocessing_steps: Sequence[Tuple[str, TransformerMixin]] = None,
) -> Pipeline:
    steps = [
        (str(i), primitive_node_to_sklearn(primitive))
        for i, primitive in enumerate(individual.primitives)
    ]
    if preprocessing_steps:
        steps = steps + list(reversed(preprocessing_steps))
    return Pipeline(list(reversed(steps)))


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
    metrics: Tuple[Metric],
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
    if len(metrics) > 1:
        raise NotImplementedError("Multiple metrics temporarily disabled.")

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

            splitter = check_cv(cv, y_train, is_classifier(pipeline))
            result = cross_validate(
                pipeline,
                x,
                y_train,
                cv=splitter,
                return_estimator=True,
                scoring=[m.name for m in metrics],
                error_score="raise",
            )
            scores = tuple([np.mean(result[f"test_{m.name}"]) for m in metrics])
            estimators = result["estimator"]

            for (estimator, (_, test)) in zip(estimators, splitter.split(x, y_train)):
                if any([m.requires_probabilities for m in metrics]):
                    fold_pred = estimator.predict_proba(x.iloc[test, :])
                else:
                    fold_pred = estimator.predict(x.iloc[test, :])

                if prediction is None:
                    if fold_pred.ndim == 2:
                        prediction = np.empty(shape=(len(y_train), fold_pred.shape[1]))
                    else:
                        prediction = np.empty(shape=(len(y_train),))
                prediction[test] = fold_pred

            # prediction, scores, estimators = cross_val_predict_score(
            #     pipeline, x, y_train, cv=cv, metrics=metrics
            # )
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
            return prediction, scores, estimators, e

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
        return prediction, scores, estimators, stopit.TimeoutException()

    return prediction, tuple(scores), estimators, None


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
    result = Evaluation(individual, pid=os.getpid())
    result.start_time = datetime.now()

    if deadline is not None:
        time_to_deadline = deadline - time.time()
        timeout = min(timeout, time_to_deadline)

    with Stopwatch() as wall_time, Stopwatch(time.process_time) as process_time:
        evaluation = evaluate_pipeline(individual.pipeline, timeout=timeout, **kwargs)
        result._predictions, result.score, result._estimators, result.error = evaluation
    result.duration = wall_time.elapsed_time

    if add_length_to_score:
        result.score = result.score + (-len(individual.primitives),)
    individual.fitness = Fitness(
        result.score,
        result.start_time,
        wall_time.elapsed_time,
        process_time.elapsed_time,
    )

    return result
