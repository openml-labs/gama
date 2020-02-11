import heapq
from typing import Tuple, List, Optional, Iterable, Union

import numpy as np
import pandas as pd

from gama.genetic_programming.components import Individual


class Evaluation:
    """ Record relevant evaluation data of an individual. """
    __slots__ = ['individual', 'score', 'predictions', 'estimators', 'start_time', 'duration', 'error']

    def __init__(
            self,
            individual: Individual,
            predictions: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None,
            score: Tuple[float, ...] = None,
            estimators: Optional[List] = None,
            start_time: float = -1,
            duration: float = -1,
            error: str = None
    ):
        self.individual: Individual = individual
        self.score = () if score is None else score
        self.estimators = [] if estimators is None else estimators
        self.start_time = start_time
        self.duration = duration
        self.error = error

        if isinstance(predictions, (pd.Series, pd.DataFrame)):
            predictions = predictions.values
        self.predictions: Optional[np.ndarray] = predictions

    # Is there a better way to do this? Assignment in __init__ is not preferred even if it saves lines.
    def __lt__(self, other): return self.score.__lt__(other.score)
    def __le__(self, other): return self.score.__le__(other.score)
    def __eq__(self, other): return self.score.__eq__(other.score)
    def __ne__(self, other): return self.score.__ne__(other.score)
    def __gt__(self, other): return self.score.__gt__(other.score)
    def __ge__(self, other): return self.score.__ge__(other.score)


class EvaluationLibrary:
    """ Maintains an in-memory record of evaluations. """

    def __init__(
            self,
            m: Optional[int] = 200,
            prediction_sample: Optional[Union[int, np.ndarray]] = None
    ):
        """
        
        Parameters
        ----------
        m: int, optional (default=200)
            Evaluations outside of the top `m` have their predictions and estimators discarded to reduce memory usage.
            The best `m` evaluations can be queried quickly through `n_best(int)`.
            If set to `None` all evaluations remain sorted and with predictions and fit pipelines.
        prediction_sample: int or np.ndarray, optional (default=None)
            Allows downsampling of predictions to a select number before storing the evaluation.
            This is useful if you don't plan on using all predictions anyway, as it lowers memory usage.
            If it is set with an int, `prediction_sample` is the number of predictions to keep of each evaluation.
            If it is set with a numpy array, it specifies the indices of the predictions to keep.
            Set with an array if it matters which predictions to keep (e.g. class stratified samples).
        """
        self.top_evaluations = []
        self.other_evaluations = []
        self._m = m
        self._sample = prediction_sample

    @property
    def evaluations(self):
        return self.top_evaluations + self.other_evaluations

    def save_evaluation(self, evaluation: Evaluation) -> None:
        self._downsample_predictions(evaluation)

        if self._m is None or self._m > len(self.top_evaluations):
            heapq.heappush(self.top_evaluations, evaluation)
        else:
            removed = heapq.heappushpop(self.top_evaluations, evaluation)
            removed.predictions, removed.estimators = None, None
            self.other_evaluations.append(removed)

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
        """ Return the best `n` pipelines. Slower if `n` exceeds `m` given on initialization. """
        if n <= self._m:
            return [e for e in heapq.nlargest(n, self.top_evaluations) if e.predictions is not None]
        else:
            return list(reversed(sorted(self.evaluations)))[:n]
