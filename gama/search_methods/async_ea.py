import logging
from functools import partial
from typing import Optional, Any, Tuple, Dict, List, Callable

import pandas as pd

from gama.genetic_programming.components import Individual
from gama.genetic_programming.operator_set import OperatorSet
from gama.search_methods import _check_base_search_hyperparameters
from gama.logging.machine_logging import TOKENS, log_event
from gama.logging.utility_functions import MultiprocessingLogger
from gama.search_methods.base_search import BaseSearch
from gama.utilities.generic.async_executor import AsyncExecutor

log = logging.getLogger(__name__)


class AsyncEA(BaseSearch):

    def __init__(self,
                 population_size: Optional[int] = None,
                 max_n_evaluations: Optional[int] = None,
                 restart_callback: Optional[Callable] = None):
        # maps hyperparameter -> (set value, default)
        self.hyperparameters: Dict[str, Tuple[Any, Any]] = dict(
            population_size=(population_size, 50),
            restart_callback=(restart_callback, None),
            max_n_evaluations=(max_n_evaluations, 10_000)
        )
        self.output = []

    def dynamic_defaults(self, x: pd.DataFrame, y: pd.DataFrame, time: int):
        pass

    def search(self, operations: OperatorSet, start_candidates: List[Individual]):
        hyperparameters = {parameter: set_value if set_value is not None else default
                           for parameter, (set_value, default) in self.hyperparameters.items()}
        self.output = async_ea(operations, self.output, start_candidates, **hyperparameters)


def async_ea(toolbox, output, start_candidates, restart_callback=None, max_n_evaluations=10000, population_size: int = 50):
    _check_base_search_hyperparameters(toolbox, output, start_candidates)
    if max_n_evaluations <= 0:
        raise ValueError("'n_evaluations' must be non-negative, but was {}.".format(max_n_evaluations))

    max_population_size = len(start_candidates)
    logger = MultiprocessingLogger()

    evaluate_log = partial(toolbox.evaluate, logger=logger)
    futures = set()

    current_population = output

    with AsyncExecutor() as async_:
        should_restart = True
        while should_restart:
            should_restart = False
            current_population[:] = []
            log.info('Starting EA with new population.')
            for individual in start_candidates:
                futures.add(async_.submit(evaluate_log, individual))

            for ind_no in range(max_n_evaluations):
                done, futures = toolbox.wait_first_complete(futures)
                logger.flush_to_log(log)
                for future in done:
                    individual = future.result()
                    should_restart = (restart_callback is not None and restart_callback())
                    if should_restart:
                        log.info("Restart criterion met. Restarting with new random population.")
                        log_event(log, TOKENS.EA_RESTART, ind_no)
                        start_candidates = [toolbox.individual() for _ in range(max_population_size)]
                        break

                    current_population.append(individual)
                    if len(current_population) > max_population_size:
                        to_remove = toolbox.eliminate(current_population, 1)
                        log_event(log, TOKENS.EA_REMOVE_IND, to_remove[0])
                        current_population.remove(to_remove[0])

                    if len(current_population) > 1:
                        new_individual = toolbox.create(current_population, 1)[0]
                        futures.add(async_.submit(evaluate_log, new_individual))

    return current_population
