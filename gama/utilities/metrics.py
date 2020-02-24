""" This module defines the built-in metrics for GAMA.

The valid metrics depend on the type of task. Many scikit-learn metrics are available.
For classification, the following metrics are available:

    - accuracy, roc_auc, average_precision, log_loss
    - micro, macro, weighted and sample averages for precision, recall and f1

For regression, the following metrics are available:

    explained_variance, r2, median_absolute_error, mean_squared_error,
    mean_squared_log_error, mean_absolute_error

Whether to minimize or maximize is determined automatically.
"""
from enum import Enum
from functools import partial
from typing import Callable, Iterable, Tuple, Union, Dict

import numpy as np
import pandas as pd
from sklearn import metrics

"""
Scikit-learn does not have an option to return predictions and score at the same time. 
Furthermore, the only string interpretation of scoring functions automatically make 
'scorers' which train the model internally, also throwing away any predictions.
So we need to make our own conversion of scoring string to function, predict, score,
and return both. Construction of metric_strings copied with minor modifications 
from SCORERS of scikit-learn. See also:

1. https://github.com/scikit-learn/scikit-learn/blob/a24c8b464d094d2c468a16ea9f8bf8d42d949f84/sklearn/metrics/scorer.py#L530
2. https://stackoverflow.com/questions/41003897/scikit-learn-cross-validates-score-and-predictions-at-one-go
"""

# name: (Score function, bool: requires_probabilities, bool: maximizing optimizes)
classification_metrics = dict(
    accuracy=(metrics.accuracy_score, False, True),
    roc_auc=(metrics.roc_auc_score, True, True),
    average_precision=(metrics.average_precision_score, True, True),
    log_loss=(metrics.log_loss, True, False),
    neg_log_loss=(metrics.log_loss, True, False),
)

# Below is also based on scikit-learn code:
for name, score_fn in [
    ("precision", metrics.precision_score),
    ("recall", metrics.recall_score),
    ("f1", metrics.f1_score),
]:
    classification_metrics[name] = (score_fn, False, True)
    for average in ["macro", "micro", "samples", "weighted"]:
        qualified_name = "{0}_{1}".format(name, average)
        qualified_score_fn = partial(score_fn, average=average)
        classification_metrics[qualified_name] = (qualified_score_fn, False, True)

regression_metrics: Dict[str, Tuple[Callable, bool, bool]] = dict(
    explained_variance=(metrics.explained_variance_score, False, True),
    r2=(metrics.r2_score, False, True),
    neg_mean_absolute_error=(metrics.mean_absolute_error, False, False),
    mean_absolute_error=(metrics.mean_absolute_error, False, False),
    neg_mean_squared_log_error=(metrics.mean_squared_log_error, False, False),
    mean_squared_log_error=(metrics.mean_squared_log_error, False, False),
    neg_median_absolute_error=(metrics.median_absolute_error, False, False),
    median_absolute_error=(metrics.median_absolute_error, False, False),
    neg_mean_squared_error=(metrics.mean_squared_error, False, False),
    mean_squared_error=(metrics.mean_squared_error, False, False),
)

all_metrics = {**classification_metrics, **regression_metrics}


class MetricType(Enum):
    """ Metric types supported by GAMA. """

    CLASSIFICATION: int = 1  #: discrete target
    REGRESSION: int = 2  #: continuous target


class Metric:
    """ A wrapper for a scoring function to provide additional meta-data. """

    def __init__(
        self,
        metric_name: str,
        score_function: Callable,
        requires_probabilities: bool,
        maximize: bool,
        task_type: MetricType,
    ):
        """
        Parameters
        ----------
        metric_name: str
            Name of the metric.
        score_function: Callable
            Takes either predicted targets or predicted probabilities and
            the true target values to calculate a score.
        requires_probabilities: bool
            For classification metrics only,
            describes if the metric should be computed with predicted probabilities.
        maximize: bool
            If True, indicates that maximizing the score corresponds to better models.
        task_type: MetricType
            Type of the machine learning task.
        """
        self.name = metric_name
        self._score_function = score_function
        self.requires_probabilities = requires_probabilities
        self._optimize_modifier = 1 if maximize else -1
        self.task_type = task_type

    def score(self, y_true, predictions) -> float:
        """ Score the predictions based on the metric.

        :param y_true:
            array of shape (N,K) if metric relies on class probabilities, (N,) otherwise
        :param predictions:
            array of shape (N,K) if metric relies on class probabilities, (N,) otherwise
        :return: score of predictions according to the metric.
        """
        # Scikit-learn metrics can be very flexible with their input,
        # interpreting a list as class labels for one metric,
        # while interpreting it as class probability for the positive class for another.
        # We want to force clear and early errors
        # to avoid accidentally working with the wrong data/interpretation.
        if not isinstance(y_true, (np.ndarray, pd.Series, pd.DataFrame)):
            raise TypeError(
                "y_true must be a numpy array, pandas series or pandas dataframe."
            )
        if not isinstance(predictions, (np.ndarray, pd.Series, pd.DataFrame)):
            raise TypeError(
                "predictions must be a numpy array, pandas series or pandas dataframe."
            )
        return self._score_function(y_true, predictions)

    def maximizable_score(self, y_true, predictions):
        """ The score, possibly made negative, s.t. maximizing is always better. """
        return self._optimize_modifier * self.score(y_true, predictions)

    @classmethod
    def from_string(cls, metric_name: str) -> "Metric":
        if metric_name in regression_metrics:
            task_type = MetricType.REGRESSION
        elif metric_name in classification_metrics:
            task_type = MetricType.CLASSIFICATION
        else:
            raise ValueError(f"Metric not known: {metric_name}.")

        score_fn, requires_probabilities, should_maximize = all_metrics[metric_name]
        return cls(
            metric_name, score_fn, requires_probabilities, should_maximize, task_type,
        )


def scoring_to_metric(
    scoring: Union[str, Metric, Iterable[str], Iterable[Metric]]
) -> Tuple[Metric]:
    if isinstance(scoring, str):
        return tuple([Metric.from_string(scoring)])
    if isinstance(scoring, Metric):
        return tuple([scoring])

    if isinstance(scoring, Iterable):
        if all([isinstance(scorer, (Metric, str)) for scorer in scoring]):
            converted_metrics = [
                scorer if isinstance(scorer, Metric) else Metric.from_string(scorer)
                for scorer in scoring
            ]
            return tuple(converted_metrics)

    raise TypeError("scoring must be str, Metric or Iterable (of str or Metric).")
