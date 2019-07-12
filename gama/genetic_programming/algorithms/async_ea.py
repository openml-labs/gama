import logging
from functools import partial

from gama.logging.machine_logging import TOKENS, log_parseable_event
from gama.logging.utility_functions import MultiprocessingLogger
from gama.utilities.generic.async_executor import AsyncExecutor

log = logging.getLogger(__name__)


def async_ea(toolbox, output, start_population, restart_callback=None, max_n_evaluations=10000, max_time_seconds=1e7, n_jobs=1):
    if max_time_seconds <= 0 or max_time_seconds > 3e6:
        raise ValueError("'max_time_seconds' must be greater than 0 and less than or equal to 3e6, but was {}."
                         .format(max_time_seconds))
    if max_n_evaluations <= 0:
        raise ValueError("'n_evaluations' must be non-negative, but was {}.".format(max_n_evaluations))
    if n_jobs <= 0:
        raise ValueError("'n_jobs' must be non-negative, but was {}.".format(n_jobs))

    max_population_size = len(start_population)
    logger = MultiprocessingLogger()

    evaluate_log = partial(toolbox.evaluate, logger=logger)
    futures = set()

    current_population = output

    with AsyncExecutor(n_jobs) as async_:
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
                    log_parseable_event(log, TOKENS.EVALUATION_RESULT, individual.fitness.start_time,
                                        individual.fitness.wallclock_time, individual.fitness.process_time,
                                        individual.fitness.values, individual._id, individual.pipeline_str())

                    should_restart = (restart_callback is not None and restart_callback())
                    if should_restart:
                        log.info("Restart criterion met. Restarting with new random population.")
                        log_parseable_event(log, TOKENS.EA_RESTART, ind_no)
                        start_population = [toolbox.individual() for _ in range(max_population_size)]
                        break

                    current_population.append(individual)
                    if len(current_population) > max_population_size:
                        to_remove = toolbox.eliminate(current_population, 1)
                        log_parseable_event(log, TOKENS.EA_REMOVE_IND, to_remove[0])
                        current_population.remove(to_remove[0])

                    if len(current_population) > 1:
                        new_individual = toolbox.create(current_population, 1)[0]
                        futures.add(async_.submit(evaluate_log, new_individual))

    return current_population
