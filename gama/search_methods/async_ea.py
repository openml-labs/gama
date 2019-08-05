import logging
from functools import partial

from gama.search_methods import _check_base_search_hyperparameters
from gama.logging.machine_logging import TOKENS, log_event
from gama.logging.utility_functions import MultiprocessingLogger
from gama.utilities.generic.async_executor import AsyncExecutor

log = logging.getLogger(__name__)


def async_ea(toolbox, output, start_population, restart_callback=None, max_n_evaluations=10000):
    _check_base_search_hyperparameters(toolbox, output, start_population)
    if max_n_evaluations <= 0:
        raise ValueError("'n_evaluations' must be non-negative, but was {}.".format(max_n_evaluations))

    max_population_size = len(start_population)
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
            for individual in start_population:
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
                        start_population = [toolbox.individual() for _ in range(max_population_size)]
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
