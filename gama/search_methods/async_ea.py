import logging
from functools import partial
from typing import Optional, Any, Tuple, Dict, List, Callable

import pandas as pd

from gama.genetic_programming.components import Individual
from gama.genetic_programming.operator_set import OperatorSet
from gama.logging.evaluation_logger import EvaluationLogger
from gama.search_methods.base_search import BaseSearch
from gama.utilities.generic.async_evaluator import AsyncEvaluator

log = logging.getLogger(__name__)


class AsyncEA(BaseSearch):
    """ Perform asynchronous evolutionary optimization.

    Parameters
    ----------
    population_size: int, optional (default=50)
        Maximum number of individuals in the population at any time.

    max_n_evaluations: int, optional (default=None)
        If specified, only a maximum of `max_n_evaluations` individuals are evaluated.
        If None, the algorithm will be run until interrupted by the user or a timeout.

    restart_callback: Callable[[], bool], optional (default=None)
        Function which takes no arguments and returns True if search restart.
    """

    def __init__(
        self,
        population_size: Optional[int] = None,
        max_n_evaluations: Optional[int] = None,
        restart_callback: Optional[Callable[[], bool]] = None,
    ):
        super().__init__()
        # maps hyperparameter -> (set value, default)
        self._hyperparameters: Dict[str, Tuple[Any, Any]] = dict(
            population_size=(population_size, 50),
            restart_callback=(restart_callback, None),
            max_n_evaluations=(max_n_evaluations, None),
        )
        self.output = []

        def get_parent(evaluation, n) -> str:
            """ retrieves the nth parent if it exists, '' otherwise. """
            if len(evaluation.individual.meta.get("parents", [])) > n:
                return evaluation.individual.meta["parents"][n]
            return ""

        self.logger = partial(
            EvaluationLogger,
            extra_fields=dict(
                parent0=partial(get_parent, n=0),
                parent1=partial(get_parent, n=1),
                origin=lambda e: e.individual.meta.get("origin", "unknown"),
            ),
        )

    def dynamic_defaults(self, x: pd.DataFrame, y: pd.DataFrame, time_limit: float):
        pass

    def search(self, operations: OperatorSet, start_candidates: List[Individual]):
        self.output = async_ea(
            operations, self.output, start_candidates, **self.hyperparameters
        )


def async_ea(
    ops: OperatorSet,
    output: List[Individual],
    start_candidates: List[Individual],
    restart_callback: Optional[Callable[[], bool]] = None,
    max_n_evaluations: Optional[int] = None,
    population_size: int = 50,
) -> List[Individual]:
    """ Perform asynchronous evolutionary optimization with given operators.

    Parameters
    ----------
    ops: OperatorSet
        Operator set with `evaluate`, `create`, `individual` and `eliminate` functions.
    output: List[Individual]
        A list which contains the set of best found individuals during search.
    start_candidates: List[Individual]
        A list with candidate individuals which should be used to start search from.
    restart_callback: Callable[[], bool], optional (default=None)
        Function which takes no arguments and returns True if search restart.
    max_n_evaluations: int, optional (default=None)
        If specified, only a maximum of `max_n_evaluations` individuals are evaluated.
        If None, the algorithm will be run indefinitely.
    population_size: int (default=50)
        Maximum number of individuals in the population at any time.

    Returns
    -------
    List[Individual]
        The individuals currently in the population.
    """
    if max_n_evaluations is not None and max_n_evaluations <= 0:
        raise ValueError(
            f"n_evaluations must be non-negative or None, is {max_n_evaluations}."
        )

    max_pop_size = population_size

    current_population = output
    n_evaluated_individuals = 0

    with AsyncEvaluator() as async_:
        should_restart = True
        while should_restart:
            should_restart = False
            current_population[:] = []
            log.info("Starting EA with new population.")
            for individual in start_candidates:
                async_.submit(ops.evaluate, individual)

            while (max_n_evaluations is None) or (
                n_evaluated_individuals < max_n_evaluations
            ):
                future = ops.wait_next(async_)
                if future.exception is None:
                    individual = future.result.individual
                    current_population.append(individual)
                    if len(current_population) > max_pop_size:
                        to_remove = ops.eliminate(current_population, 1)
                        current_population.remove(to_remove[0])

                if len(current_population) > 2:
                    new_individual = ops.create(current_population, 1)[0]
                    async_.submit(ops.evaluate, new_individual)

                should_restart = restart_callback is not None and restart_callback()
                n_evaluated_individuals += 1
                if should_restart:
                    log.info("Restart criterion met. Creating new random population.")
                    start_candidates = [ops.individual() for _ in range(max_pop_size)]
                    break

    return current_population
