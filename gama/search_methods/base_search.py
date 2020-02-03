from abc import ABC
from typing import List, Dict, Tuple, Any, Union, Callable

import pandas as pd

from gama.genetic_programming.operator_set import OperatorSet
from gama.genetic_programming.components import Individual
from gama.utilities.generic.timekeeper import TimeKeeper


class BaseSearch(ABC):
    """ All search methods should be derived from this class.
    This class should not be directly used to configure GAMA.
    """

    def __init__(self):
        # hyperparameters can be used to safe/process search hyperparameters
        self.hyperparameters: Dict[str, Tuple[Any, Any]] = dict()
        self.output: List[Individual] = []

    def dynamic_defaults(
            self,
            x: pd.DataFrame,
            y: Union[pd.DataFrame, pd.Series],
            time_limit: int) -> None:
        """ Set hyperparameter defaults taking into account dataset characteristics and time-constraints.

        Called before `search`.

        Parameters
        ----------
        x: pandas.DataFrame
            Features of the data.
        y: pandas.DataFrame or pandas.Series
            Labels of the data.
        time_limit: int
            Time in seconds available for search and selecting dynamic defaults.
            There is no need to adhere to this explicitly, a `stopit.utils.TimeoutException` will be raised.
            However, the time-limit might be an important factor in setting hyperparameter values.
        """
        # updates self.hyperparameters defaults
        raise NotImplementedError("Must be implemented by child class.")

    def search(self, operations: OperatorSet, start_candidates: List[Individual]):
        """ Execute search as configured. Set `output` field of this class to the best Individuals.

        Parameters
        ----------
        operations: OperatorSet
            Has methods to create new individuals, evaluate individuals and more.
        start_candidates: List[Individual]
            A list of individuals to be considered before all others.
        """
        raise NotImplementedError("Must be implemented by child class.")


def _check_base_search_hyperparameters(
        toolbox,
        output: List[Individual],
        start_candidates: List[Individual]
) -> None:
    """ Checks that search hyperparameters are valid.

    :param toolbox:
    :param output:
    :param start_candidates:
    :return:
    """
    if not isinstance(start_candidates, list):
        raise TypeError(f"'start_population' must be a list but was {type(start_candidates)}")
    if not all(isinstance(x, Individual) for x in start_candidates):
        raise TypeError(f"Each element in 'start_population' must be Individual.")
