from abc import ABC
from typing import List, Union

import pandas as pd


from gama.genetic_programming.components import Individual


class BasePostProcessing(ABC):
    """ All post-processing methods should be derived from this class.
    This class should not be directly used to configure GAMA.
    """

    def __init__(self, time_fraction: float):
        """

        Parameters
        ----------
        time_fraction: float
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
        Parameters
        ----------
        x: pd.DataFrame
            all training features
        y: Union[pd.DataFrame, pd.Series]
            all training labels
        timeout: float
            allowed time in seconds for post-processing
        selection: List[Individual]
            individuals selected by the search space, ordered best first

        Returns
        -------
        Any
            A model with `predict` and optionally `predict_proba`.
        """
        raise NotImplementedError("Method must be implemented by child class.")
