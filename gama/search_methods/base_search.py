from abc import ABC
from typing import List, Dict, Tuple, Any, Union

import pandas as pd

from gama.genetic_programming.operator_set import OperatorSet
from gama.genetic_programming.components import Individual
from gama.logging.evaluation_logger import EvaluationLogger


class BaseSearch(ABC):
    """ All search methods should be derived from this class.
    This class should not be directly used to configure GAMA.
    """

    def __init__(self):
        # hyperparameters can be used to safe/process search hyperparameters
        self._hyperparameters: Dict[str, Tuple[Any, Any]] = dict()
        self.output: List[Individual] = []
        self.logger = EvaluationLogger

    def __str__(self):
        # Not sure if I should report actual used hyperparameters (i.e. include default)
        # or only those set by user.
        user_set_hps = {
            parameter: set_value
            for parameter, (set_value, default) in self._hyperparameters.items()
            if set_value is not None
        }
        hp_configuration = ",".join(
            [f"{name}={value}" for (name, value) in user_set_hps.items()]
        )
        return f"{self.__class__.__name__}({hp_configuration})"

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        """ Hyperparameter (name, value) pairs as set/determined dynamically/default.

         Values may have been set directly, through dynamic defaults or static defaults.
         This is also the order in which the value of a hyperparameter is checked,
         i.e. a user set value wil overwrite any other value, and a dynamic default
         will overwrite a static one.
         Dynamic default values only considered if `dynamic_defaults` has been called.
         """
        return {
            parameter: set_value if set_value is not None else default
            for parameter, (set_value, default) in self._hyperparameters.items()
        }

    def _overwrite_hyperparameter_default(self, hyperparameter: str, value: Any):
        set_value, default_value = self._hyperparameters[hyperparameter]
        self._hyperparameters[hyperparameter] = (set_value, value)

    def dynamic_defaults(
        self, x: pd.DataFrame, y: Union[pd.DataFrame, pd.Series], time_limit: float
    ) -> None:
        """ Set hyperparameter defaults based on the dataset and time-constraints.

        Should be called before `search`.

        Parameters
        ----------
        x: pandas.DataFrame
            Features of the data.
        y: pandas.DataFrame or pandas.Series
            Labels of the data.
        time_limit: float
            Time in seconds available for search and selecting dynamic defaults.
            There is no need to adhere to this explicitly,
            a `stopit.utils.TimeoutException` will be raised.
            The time-limit might be an important factor in setting hyperparameter values
        """
        # updates self.hyperparameters defaults
        raise NotImplementedError("Must be implemented by child class.")

    def search(self, operations: OperatorSet, start_candidates: List[Individual]):
        """ Execute search as configured.

        Sets `output` field of this class to the best Individuals.

        Parameters
        ----------
        operations: OperatorSet
            Has methods to create new individuals, evaluate individuals and more.
        start_candidates: List[Individual]
            A list of individuals to be considered before all others.
        """
        raise NotImplementedError("Must be implemented by child class.")


def _check_base_search_hyperparameters(
    toolbox, output: List[Individual], start_candidates: List[Individual]
) -> None:
    """ Checks that search hyperparameters are valid.

    :param toolbox:
    :param output:
    :param start_candidates:
    :return:
    """
    if not isinstance(start_candidates, list):
        raise TypeError(
            f"'start_population' must be a list but was {type(start_candidates)}"
        )
    if not all(isinstance(x, Individual) for x in start_candidates):
        raise TypeError(f"Each element in 'start_population' must be Individual.")
