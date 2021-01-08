import logging
from typing import List, Optional

from dask.distributed import Client, as_completed
import pandas as pd
import stopit

from gama.genetic_programming.components import Individual
from gama.genetic_programming.operator_set import OperatorSet
from gama.search_methods.base_search import (
    BaseSearch,
    _check_base_search_hyperparameters,
)

log = logging.getLogger(__name__)


class RandomSearch(BaseSearch):
    """ Perform random search over all possible pipelines. """

    def dynamic_defaults(self, x: pd.DataFrame, y: pd.DataFrame, time_limit: float):
        pass

    def search(self, operations: OperatorSet, start_candidates: List[Individual]):
        random_search(operations, self.output, start_candidates)


def random_search(
    operations: OperatorSet,
    output: List[Individual],
    start_candidates: List[Individual],
    max_evaluations: Optional[int] = None,
) -> List[Individual]:
    """ Perform random search over all possible pipelines.

    Parameters
    ----------
    operations: OperatorSet
        An operator set with `evaluate` and `individual` functions.
    output: List[Individual]
        A list which contains the found individuals during search.
    start_candidates: List[Individual]
        A list with candidate individuals to evaluate first.
    max_evaluations: int, optional (default=None)
        If specified, only a maximum of `max_evaluations` individuals are evaluated.
        If None, the algorithm will be run indefinitely.

    Returns
    -------
    List[Individual]
        All evaluated individuals.
    """
    _check_base_search_hyperparameters(operations, output, start_candidates)

    with Client() as client:
        try:
            futures = client.map(operations.evaluate, start_candidates)
            ac = as_completed(futures, with_results=True)
            for future, result in ac:
                if result.error is None:
                    output.append(result.individual)
                new_future = client.submit(operations.evaluate, operations.individual())
                ac.add(new_future)
                if (max_evaluations is not None) and (len(output) >= max_evaluations):
                    log.info("Stopping due to maximum number of evaluations performed.")
                    break
        except stopit.TimeoutException:
            pass
        finally:
            ac.clear()

    return output
