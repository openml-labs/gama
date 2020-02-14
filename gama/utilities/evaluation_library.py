import heapq
import logging
from typing import Tuple, List, Optional, Iterable, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from gama.genetic_programming.components import Individual

log = logging.getLogger(__name__)


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
    """ Maintains an in-memory record of evaluations.

    The main function of the EvaluationLibrary is to maintain a fast lookup for
    the best evaluations, and to discard meta-data of Evaluations which are not
    good enough.

    Specifically for the top `m` evaluations the estimators and predictions are kept.
    All but `n` predictions of each evaluation are discarded.
    As soon as an evaluation is no longer in the top `m`,
    its estimators and the sampled predictions are also discarded.
    Other evaluation meta-data (e.g. scores, evaluation time, errors) is not discarded.

    This discarding is useful to reduce memory usage when you know which meta-data
    is used later. E.g.:
    Ensembling selects from the best `m` models and uses `n` points for hill-climbing,
    but general score information is relevant for all evaluated models.
    """

    def __init__(
            self,
            m: Optional[int] = 200,
            n: Optional[int] = 10_000,
            sample: Optional[np.ndarray] = None
    ):
        """ Create an EvaluationLibrary for in-memory record of evaluations.

        Parameters
        ----------
        m: int, optional (default=200)
            Evaluations outside of the top `m` have their predictions and estimators discarded to reduce memory usage.
            The best `m` evaluations can be queried quickly through `n_best(int)`.
            If set to `None` all evaluations remain sorted and with predictions and fit pipelines.
        n: int, optional (default=10_000)
            Instead of storing all predictions of the 'top m' evaluations,
            store only 'n' predictions. If left `None` all predictions will be kept.
            This parameter is ignored when 'sample' is also provided.
            Call `determine_sample_indices` before saving the first evaluation,
            if you want to ensure a class stratified sample.
        sample: np.ndarray, optional (default=None)
            Instead of storing all predictions of the 'top m' evaluations,
            store only those with indices specified in this array.
        """
        self.top_evaluations = []
        self.other_evaluations = []
        self._m = m
        self._sample_n = n

        if sample is not None:
            self._sample = sample
        elif n is None:
            self._sample = None
        else:
            self._sample = "not set"

    @property
    def evaluations(self):
        return self.top_evaluations + self.other_evaluations

    def determine_sample_indices(
            self,
            n: Optional[int] = None,
            prediction_size: Optional[int] = None,
            stratify: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None
    ) -> None:
        """ Set `self._sample` to an array for sampling predictions or `None`.

        The sample indices can be class stratified if `stratify` is set.
        If `prediction_size` or `len(stratify)` is smaller than `n`,
        predictions will not be sampled and all predictions are saved.

        Parameters
        ----------
        n: int, optional (default=None)
            Number of predictions to keep for each evaluation.
            If `None`, use `n` set at the library's initialisation.
        prediction_size: int, optional (default=None)
            Number of predictions one can sample from.
        stratify: np.ndarray, pd.Series or pd.DataFrame, optional (default=None)
            The target variable of a supervised classification problem.
            The sample will be class stratified according to this variable.
        """
        if not ((prediction_size is None) ^ (stratify is None)):
            raise ValueError("Exactly one of `prediction_size` and `stratify` must be set.")
        if len(self.top_evaluations) > 0:
            log.warning("New subsample not used for already stored evaluations.")
        n = self._sample_n if n is None else n

        if n is not None:
            if prediction_size is not None and n < prediction_size:
                # Subsample is to be chosen uniformly random.
                self._sample = np.random.choice(range(prediction_size), size=n, replace=False)
            elif stratify is not None and n < len(stratify):
                splitter = StratifiedShuffleSplit(n_splits=1, train_size=n)
                self._sample, _ = next(splitter.split(np.zeros(len(stratify)), stratify))
            else:
                # Specified sample size exceeds size of predictions
                self._sample = None
        else:
            # No n was provided here nor set on initialization
            self._sample = None

    def _process_predictions(self, evaluation: Evaluation):
        """ Downsample evaluation predictions if required. """
        if self._sample_n == 0:
            evaluation.predictions = None
        if evaluation.predictions is None:
            return  # Predictions either not provided or removed because sample_n is 0.

        if isinstance(self._sample, str) and self._sample == "not set":
            # Happens only for the first evaluation with predictions.
            self.determine_sample_indices(self._sample_n, len(evaluation.predictions))

        if self._sample is not None:
            evaluation.predictions = evaluation.predictions[self._sample]

    def save_evaluation(self, evaluation: Evaluation) -> None:
        self._process_predictions(evaluation)

        if evaluation.error is not None:
            self.other_evaluations.append(evaluation)
        elif self._m is None or self._m > len(self.top_evaluations):
            heapq.heappush(self.top_evaluations, evaluation)
        else:
            removed = heapq.heappushpop(self.top_evaluations, evaluation)
            removed.predictions, removed.estimators = None, None
            self.other_evaluations.append(removed)

    def n_best(self, n: int = 5) -> List[Evaluation]:
        """ Return the best `n` pipelines. Slower if `n` exceeds `m` given on initialization. """
        if self._m is None or n <= self._m:
            return [e for e in heapq.nlargest(n, self.top_evaluations) if e.predictions is not None]
        else:
            return list(reversed(sorted(self.evaluations)))[:n]
