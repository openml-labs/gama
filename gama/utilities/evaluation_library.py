import heapq
from typing import Tuple, List, Optional, Iterable, Union

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

    def __init__(
            self,
            max_number_of_evaluations: Optional[int] = 200,
            prediction_sample: Optional[Union[int, np.ndarray]] = None
    ):
        """
        
        Parameters
        ----------
        max_number_of_evaluations: int, optional (default=200)
            Maximum number of evaluations to keep in memory with predictions and fitted pipelines.
        prediction_sample: int or np.ndarray, optional (default=None)
            Allows downsampling of predictions to a select number before storing the evaluation.
            This is useful if you don't plan on using all predictions anyway, as it lowers memory usage.
            If it is set with an int, `prediction_sample` is the number of predictions to keep of each evaluation.
            If it is set with a numpy array, it specifies the indices of the predictions to keep.
            Set with an array if it matters which predictions to keep (e.g. class stratified samples).
        """
        self.top_evaluations = []
        self._max_n_evaluations = max_number_of_evaluations
        self._sample = prediction_sample

    def save_evaluation(self, evaluation: Evaluation) -> None:
        self._downsample_predictions(evaluation)

        if self._max_n_evaluations is None or self._max_n_evaluations > len(self.top_evaluations):
            heapq.heappush(self.top_evaluations, evaluation)
        else:
            heapq.heappushpop(self.top_evaluations, evaluation)

    def _downsample_predictions(self, evaluation: Evaluation):
        """ Downsample predictions if possible and specified through `self._sample`. """
        if evaluation.predictions is None or self._sample is None:
            return

        # On initialization, `self._sample` may be set as int, if so, select indices for all future downsampling.
        if isinstance(self._sample, int):
            if self._sample >= len(evaluation.predictions):
                self._sample = None  # Sample size exceeds number of predictions, store all predictions, don't sample.
                return
            self._sample = np.random.choice(range(len(evaluation.predictions)), size=self._sample, replace=False)

        evaluation.predictions = evaluation.predictions[self._sample]

    def n_best(self, n: int = 5) -> List[Evaluation]:
        return [e for e in heapq.nlargest(n, self.top_evaluations) if e.predictions is not None]
