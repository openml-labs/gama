import logging
from functools import partial
import time

import stopit
import pebble

from gama.utilities.generic.async_executor import AsyncExecutor
from gama.utilities.logging_utilities import TOKENS, log_parseable_event
from gama.utilities.logging_utilities import MultiprocessingLogger

log = logging.getLogger(__name__)


def _safe_outside_call(fn, timeout):
    """ Calls fn and log any exception it raises without reraising, except for TimeoutException. """
    try:
        fn()
    except stopit.utils.TimeoutException:
        raise
    except Exception:
        # We actually want to catch any other exception here, because the callback code can be
        # arbitrary (it can be provided by users). This excuses the catch-all Exception.
        # Note that KeyboardInterrupts are not exceptions and get elevated to the caller.
        log.warning("Exception during callback.", exc_info=True)
        pass
    if timeout():
        log.info("Time exceeded during callback, but exception was swallowed.")
        raise stopit.utils.TimeoutException


def async_ea(start_population, toolbox, evaluation_callback=None, restart_callback=None,
             elimination_callback=None, max_n_evaluations=10000, max_time_seconds=1e7, n_jobs=1):
    if max_time_seconds <= 0 or max_time_seconds > 3e6:
        raise ValueError("'max_time_seconds' must be greater than 0 and less than or equal to 3e6, but was {}."
                         .format(max_time_seconds))
    if max_n_evaluations <= 0:
        raise ValueError("'n_evaluations' must be non-negative, but was {}.".format(max_n_evaluations))
    if n_jobs <= 0:
        raise ValueError("'n_jobs' must be non-negative, but was {}.".format(n_jobs))

    start_time = time.time()
    max_population_size = len(start_population)
    logger = MultiprocessingLogger()

    def exceed_timeout():
        return (time.time() - start_time) > max_time_seconds

    evaluate_log = partial(toolbox.evaluate, logger=logger)
    futures = set()
    async = pebble.ProcessPool(n_jobs)
    with stopit.ThreadingTimeout(max_time_seconds) as c_mgr:
        should_restart = True
        while should_restart:
            should_restart = False
            current_population = []

            log.info('Starting EA with new population.')
            for individual in start_population:
                futures.add(async.schedule(evaluate_log, (individual,)))

            for ind_no in range(max_n_evaluations):
                completed, futures = AsyncExecutor.wait_first(futures)
                logger.flush_to_log(log)
                for individual in [future.result() for future in completed]:
                    log_parseable_event(log, TOKENS.EVALUATION_RESULT, individual.fitness.start_time,
                                        individual.fitness.wallclock_time, individual.fitness.process_time,
                                        individual.fitness.values, individual._id, individual.pipeline_str())

                    if evaluation_callback:
                        _safe_outside_call(partial(evaluation_callback, individual), exceed_timeout)

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
                        if elimination_callback:
                            _safe_outside_call(partial(elimination_callback, to_remove[0]), exceed_timeout)

                    if len(current_population) > 1:
                        new_individual = toolbox.create(current_population, 1)[0]
                        futures.add(async.schedule(evaluate_log, (new_individual,)))

    async.stop()
    #for future in futures:
    #    future.cancel()

    if not c_mgr:
        log.info('Asynchronous EA terminated because maximum time has elapsed.'
                 '{} individuals have been evaluated.'.format(ind_no))
    return current_population
