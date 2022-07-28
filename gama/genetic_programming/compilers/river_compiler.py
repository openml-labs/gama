from datetime import datetime
import logging
import os
import time
from typing import Callable, Tuple, Optional, Sequence

import stopit
#from sklearn.base import TransformerMixin, is_classifier
from river.base import Classifier
from river import evaluate
from gama.utilities.river_metrics import get_metric
#from sklearn.model_selection import ShuffleSplit, cross_validate, check_cv
from river.compose.pipeline import Pipeline #River pipeline instead
from river import compose
# we dont have cross valudate checkcv and is classifier equivalent in river, we do have metrics.workswith though
# shuffle split is also just suffle streamer and then there isa splitter
from river import stream
from river import metrics


from gama.utilities.evaluation_library import Evaluation
from gama.utilities.generic.stopwatch import Stopwatch
import numpy as np
from gama.utilities.metrics import Metric
from gama.genetic_programming.components import Individual, PrimitiveNode, Fitness

log = logging.getLogger(__name__)

# Use progressive_val_score instead of cross validate

def primitive_node_to_sklearn(primitive_node: PrimitiveNode) -> object:
    hyperparameters = {
        terminal.output: terminal.value for terminal in primitive_node._terminals
    }
    return primitive_node._primitive.identifier(**hyperparameters)


def compile_individual(
    individual: Individual,
    parameter_checks=None,
    preprocessing_steps: Sequence[Tuple[str, Classifier
    ]] = None,
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
        and hasattr(o, "learn_one")
        and hasattr(o, "predict_one")
        and hasattr(o, "steps")
    )

def evaluate_pipeline(
    pipeline, x, y_train, timeout: float,metrics: str = 'accuracy', cv=5, subsample=None,
) -> Tuple:
    """ Score `pipeline` with online holdout evaluation according to `metrics` on (a subsample of) X, y

    Returns
    -------
    Tuple:
        prediction: np.ndarray if successful, None if not
        scores: tuple with one float per metric, each value is -inf on fail.
        estimators: list of fitted pipelines if successful, None if not
        error: None if successful, otherwise an Exception
    """
    if not object_is_valid_pipeline(pipeline):
        raise TypeError(f"Pipeline must not be None and requires learn_one, predict_one, steps.")
    if not timeout > 0:
        raise ValueError(f"`timeout` must be greater than 0, is {timeout}.")
    prediction, estimators = None, None
    # default score for e.g. timeout or failure
    scores = tuple([float("-inf")])
    river_metric = get_metric(metrics)

    with stopit.ThreadingTimeout(timeout) as c_mgr:
        try:
            dataset = []
            for a, b in stream.iter_pandas(x, y_train):
                dataset.append((a,b))
            steps = list(pipeline.steps.values())

            for i in range(len(steps[0])):
                if i == 0:
                    river_model = steps[0][i][1]
                else:
                    river_model |= steps[0][i][1]

            result = evaluate.progressive_val_score(
                dataset = dataset,
                model = river_model,
                metric = river_metric,
            )

            scores = tuple([result.get()])
            estimators = river_model

            prediction = np.empty(shape=(len(y_train),))
            y_pred = []
            for a, b in stream.iter_pandas(x, y_train):
                y_pred.append(river_model.predict_one(a))
                river_model = river_model.learn_one(a, b)

            prediction = np.asarray(y_pred)


        except stopit.TimeoutException:
            # This exception is handled by the ThreadingTimeout context manager.
            raise
        except KeyboardInterrupt:
            raise
        except Exception as e:
            return prediction, scores, estimators, e

    if c_mgr.state == c_mgr.INTERRUPTED:
        # A TimeoutException was raised, but not by the context manager.
        # This indicates that the outer context manager (the ea) timed out.
        raise stopit.utils.TimeoutException()

    if not c_mgr:
        # For now we treat an eval timeout the same way as
        # e.g. NaN exceptions and use the default score.
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
        result._predictions, result.score, result._estimators, error = evaluation

        if error is not None:
            result.error = f"{type(error)} {str(error)}"
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