from abc import ABC
from typing import List, Union

import pandas as pd


from gama.genetic_programming.components import Individual


class BasePostProcessing(ABC):

    def __init__(self, time_fraction: float):
        """

        :param time_fraction: float
            The fraction of total time that should be reserved for this post-processing step.
        """
        self.time_fraction: float = time_fraction

    def dynamic_defaults(self, gama: 'Gama'):
        pass

    def post_process(
            self,
            x: pd.DataFrame,
            y: Union[pd.DataFrame, pd.Series],
            timeout: float,
            selection: List[Individual]) -> 'model':
        """

        :param x: Union[pd.DataFrame]
            all training features
        :param y: Union[pd.DataFrame, pd.Series]
            all training labels
        :param timeout: float
            allowed time in seconds for post-processing
        :param selection: List[Individual]
            individuals selected by the search space, ordered best first
        :return:
            a model with `predict` and optionally `predict_proba`
        """
        raise NotImplementedError("Method must be implemented by child class.")
