import heapq
from typing import Tuple, List, Optional, Iterable

import numpy as np

from gama.genetic_programming.components import Individual


class Evaluation:
    """ Record relevant evaluation data of an individual. """
    __slots__ = ['individual', 'score', 'predictions', 'estimators', 'start_time', 'duration', 'error']

    def __init__(self, individual: Individual):
        self.individual: Individual = individual
        self.score: Tuple[float, ...] = ()
        self.predictions: Optional[np.ndarray] = None
        self.estimators: List[object] = []
        self.start_time: float = -1
        self.duration: float = -1
        self.error = None

    # Is there a better way to do this? Assignment in __init__ is not preferred even if it saves lines.
    def __lt__(self, other): return self.score.__lt__(other.score)
    def __le__(self, other): return self.score.__le__(other.score)
    def __eq__(self, other): return self.score.__eq__(other.score)
    def __ne__(self, other): return self.score.__ne__(other.score)
    def __gt__(self, other): return self.score.__gt__(other.score)
    def __ge__(self, other): return self.score.__ge__(other.score)


class EvaluationLibrary:
    """ Maintains an in-memory record of the top n evaluations. """

    def __init__(self, max_number_of_models: Optional[int] = 200, prediction_sample=None):
        # some mask to index the predictions
        # standardize mask available for dataframe, predictions and np.ndarray??
        self.best_pipelines = []
        self._max_n_models = max_number_of_models
        self._sample = prediction_sample

    def save_evaluation(self, evaluation: Evaluation) -> None:
        if self._sample is not None:
            evaluation.predictions = evaluation.predictions[self._sample]

        if self._max_n_models is None or self._max_n_models > len(self.best_pipelines):
            heapq.heappush(self.best_pipelines, evaluation)
        else:
            heapq.heappushpop(self.best_pipelines, evaluation)

    def n_best(self, n: int = 5) -> Iterable[Evaluation]:
        return [e for e in heapq.nlargest(n, self.best_pipelines) if e.predictions is not None]
