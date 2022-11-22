from enum import Enum
from typing import Iterable, Tuple, Union
import numpy as np

from sklearn.metrics import get_scorer, make_scorer
from sklearn.metrics._scorer import _ProbaScorer, _BaseScorer, SCORERS

# scores that ground truth labels are not needed
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, silhouette_samples

SCORERS['silhouette'] = make_scorer(silhouette_score)
SCORERS['calinski_harabasz'] = make_scorer(calinski_harabasz_score)
SCORERS['davies_bouldin'] = make_scorer(davies_bouldin_score)

classification_metrics = {"accuracy", "roc_auc", "average_precision", "neg_log_loss"}
for metric in ["precision", "recall", "f1"]:
    for average in ["macro", "micro", "samples", "weighted"]:
        classification_metrics.add(f"{metric}_{average}")

regression_metrics = {"explained_variance",
                      "r2",
                      "neg_mean_absolute_error",
                      "neg_mean_squared_log_error",
                      "neg_median_absolute_error",
                      "neg_mean_squared_error",
                      }

clustering_metrics = {'silhouette',
                      'calinski_harabasz',
                      'davies_bouldin',
                      'adjusted_mutual_info_score',
                      'adjusted_rand_score',
                      'completeness_score',
                      'fowlkes_mallows_score',
                      'homogeneity_score',
                      'mutual_info_score',
                      'normalized_mutual_info_score',
                      'rand_score',
                      'v_measure_score',
                      'jaccard'
                      }

all_metrics = {*classification_metrics, *regression_metrics, *clustering_metrics}
reversed_scorers = {v: k for k, v in SCORERS.items()}


class MetricType(Enum):
    """ Metric types supported by GAMA. """

    CLASSIFICATION: int = 1  #: discrete target
    REGRESSION: int = 2  #: continuous target
    CLUSTERING: int = 3  #: unknown target


class Metric:
    """ A thin layer around the `scorer` class of scikit-learn. """

    def __init__(self, scorer: Union[_BaseScorer, str]):
        if isinstance(scorer, str):
            scorer = get_scorer(scorer)
        if not isinstance(scorer, _BaseScorer):
            raise ValueError(
                "Scorer was not a valid scorer or could not be converted to one."
            )
        self.scorer = scorer
        self.name = reversed_scorers[scorer]
        self.requires_probabilities = (
            isinstance(scorer, _ProbaScorer) or self.name == "roc_auc"
        )

        if self.name in classification_metrics:
            self.task_type = MetricType.CLASSIFICATION
        elif self.name in regression_metrics:
            self.task_type = MetricType.REGRESSION
        elif self.name in clustering_metrics:
            self.task_type = MetricType.CLUSTERING
        else:
            raise ValueError(
                "Not sure which type of metric this is. Please raise an issue."
            )

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
