import logging
from functools import partial
from typing import Optional, Any, Tuple, Dict, List, Callable

import pandas as pd

from gama.genetic_programming.components import Individual
from gama.genetic_programming.operator_set import OperatorSet
from gama.logging.machine_logging import TOKENS, log_event
from gama.logging.utility_functions import MultiprocessingLogger
from gama.search_methods.base_search import BaseSearch
from gama.utilities.generic.async_evaluation import AsyncEvaluator
from gama.utilities.generic.async_executor import AsyncExecutor

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

    restart_callback: Callable, optional (default=None)
        A function with signature () -> `bool` which returns `True` if search should be restarted.
    """

    def __init__(self,
                 population_size: Optional[int] = None,
                 max_n_evaluations: Optional[int] = None,
                 restart_callback: Optional[Callable] = None):
        super().__init__()
        # maps hyperparameter -> (set value, default)
        self.hyperparameters: Dict[str, Tuple[Any, Any]] = dict(
            population_size=(population_size, 50),
            restart_callback=(restart_callback, None),
            max_n_evaluations=(max_n_evaluations, None)
        )
        self.output = []

    def dynamic_defaults(self, x: pd.DataFrame, y: pd.DataFrame, time_limit: int):
        pass

    def search(self, operations: OperatorSet, start_candidates: List[Individual]):
        hyperparameters = {parameter: set_value if set_value is not None else default
                           for parameter, (set_value, default) in self.hyperparameters.items()}
        self.output = async_ea(operations, self.output, start_candidates, **hyperparameters)


def async_ea(
        operations: OperatorSet,
        output: List[Individual],
        start_candidates: List[Individual],
        restart_callback: Optional[Callable[[], bool]] = None,
        max_n_evaluations: Optional[int] = None,
        population_size: int = 50) -> List[Individual]:
    """ Perform asynchronous evolutionary optimization given the evolutionary operators in `operations`.

    Parameters
    ----------
    operations: OperatorSet
        An operator set with `evaluate`, `create`, `individual` and `eliminate` functions.
    output: List[Individual]
        A list which contains the set of best found individuals during search.
    start_candidates: List[Individual]
        A list with candidate individuals which should be used to start search from.
    restart_callback: Callable[[], bool], optional (default=None)
        A function with signature () -> bool which returns True if search should be restarted.
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
        raise ValueError("'n_evaluations' must be non-negative or None, but was {}.".format(max_n_evaluations))

    max_population_size = population_size
    logger = MultiprocessingLogger()

    evaluate_log = partial(operations.evaluate, logger=logger)
    futures = set()

    current_population = output
    n_evaluated_individuals = 0

    with AsyncEvaluator() as async_:
        should_restart = True
        while should_restart:
            should_restart = False
            current_population[:] = []
            log.info('Starting EA with new population.')
            for individual in start_candidates:
                futures.add(async_.submit(evaluate_log, individual))

            while (max_n_evaluations is None) or (n_evaluated_individuals < max_n_evaluations):
                done = operations.wait_next(async_)
                #logger.flush_to_log(log)
                #for future in done:
                individual = done.result
                current_population.append(individual)
                if len(current_population) > max_population_size:
                    to_remove = operations.eliminate(current_population, 1)
                    log_event(log, TOKENS.EA_REMOVE_IND, to_remove[0])
                    current_population.remove(to_remove[0])

                if len(current_population) > 1:
                    new_individual = operations.create(current_population, 1)[0]
                    futures.add(async_.submit(evaluate_log, new_individual))

                should_restart = (restart_callback is not None and restart_callback())
                n_evaluated_individuals += 1
                if should_restart:
                    log.info("Restart criterion met. Restarting with new random population.")
                    log_event(log, TOKENS.EA_RESTART, n_evaluated_individuals)
                    start_candidates = [operations.individual() for _ in range(max_population_size)]
                    break

    return current_population
