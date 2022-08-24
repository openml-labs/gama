from enum import Enum
from typing import Iterable, Tuple, Union

from sklearn.metrics import get_scorer
from sklearn.metrics._scorer import _ProbaScorer, _BaseScorer, SCORERS

from river import metrics

classification_metrics = {"accuracy", "roc_auc", "average_precision", "neg_log_loss"}

for metric in ["precision", "recall", "f1"]:
    for average in ["macro", "micro", "samples", "weighted"]:
        classification_metrics.add(f"{metric}_{average}")

regression_metrics = {
    "explained_variance",
    "r2",
    "neg_mean_absolute_error",
    "neg_mean_squared_log_error",
    "neg_median_absolute_error",
    "neg_mean_squared_error",
}
river_metrics = {"r_accuracy", "r_balanced_accuracy", "r_f1", "r_roc_auc"}
all_metrics = {*classification_metrics, *regression_metrics, *river_metrics}
reversed_scorers = {v: k for k, v in SCORERS.items()}


class MetricType(Enum):
    """Metric types supported by GAMA."""

    CLASSIFICATION: int = 1  #: discrete target
    REGRESSION: int = 2  #: continuous target
    ONLINE: int = 3  #: continuous metric for online learning


class Metric:
    """A thin layer around the `scorer` class of scikit-learn."""

    def __init__(self, scorer: Union[_BaseScorer, str]):
        if isinstance(scorer, str):
            if scorer in river_metrics:
                self.name = scorer
                scorer = get_river_metric(scorer)
            else:
                scorer = get_scorer(scorer)
        if not isinstance(scorer, _BaseScorer):
            if not isinstance(scorer, metrics.base.ClassificationMetric):
                raise ValueError(
                    "Scorer was not a valid scorer or could not be converted to one."
                )
        self.scorer = scorer
        if not isinstance(scorer, metrics.base.ClassificationMetric):
            self.name = reversed_scorers[scorer]

        self.requires_probabilities = (
            isinstance(scorer, _ProbaScorer) or self.name == "roc_auc"
        )
        if self.name in classification_metrics:
            self.task_type = MetricType.CLASSIFICATION
        elif self.name in regression_metrics:
            self.task_type = MetricType.REGRESSION
        elif self.name in river_metrics:
            self.task_type = MetricType.ONLINE
        else:
            raise ValueError(
                "Not sure which type of metric this is. Please raise an issue."
            )
        if self.name not in river_metrics:
            self.score = self.scorer._score_func

    def __call__(self, *args, **kwargs):
        return self.scorer(*args, **kwargs)

    def maximizable_score(self, *args, **kwargs):
        return self.scorer._sign * self.score(*args, **kwargs)


def scoring_to_metric(
    scoring: Union[str, Metric, Iterable[str], Iterable[Metric]]
) -> Tuple[Metric, ...]:
    if isinstance(scoring, str):
        return tuple([Metric(scoring)])
    if isinstance(scoring, Metric):
        return tuple([scoring])

    if isinstance(scoring, Iterable):
        if all([isinstance(scorer, (Metric, str)) for scorer in scoring]):
            converted_metrics = [
                scorer if isinstance(scorer, Metric) else Metric(scorer)
                for scorer in scoring
            ]
            return tuple(converted_metrics)

    raise TypeError("scoring must be str, Metric or Iterable (of str or Metric).")


def get_river_metric(
    scoring: Union[str, Metric, Iterable[str], Iterable[Metric]]
) -> Tuple[Metric, ...]:
    if scoring == "r_accuracy":
        metric = metrics.Accuracy()
    elif scoring == "r_balanced_accuracy":
        metric = metrics.BalancedAccuracy()
    elif scoring == "r_f1":
        metric = metrics.F1()
    elif scoring == "r_roc_auc":
        metric = metrics.ROCAUC()
    return metric
