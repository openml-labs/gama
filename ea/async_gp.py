import logging
from functools import partial
import time

import stopit

from ..utilities.mp_logger import MultiprocessingLogger
from ..utilities.function_dispatcher import FunctionDispatcher

log = logging.getLogger(__name__)


def async_ea(objectives, start_population, toolbox, evaluation_callback=None, restart_callback=None, n_evaluations=10000, max_time_seconds=1e7, n_jobs=1):
    if max_time_seconds <= 0 or max_time_seconds > 3e6:
        raise ValueError("'max_time_seconds' must be greater than 0 and less than or equal to 3e6, but was {}."
                         .format(max_time_seconds))
    if n_evaluations <= 0:
        raise ValueError("'n_evaluations' must be non-negative, but was {}.".format(n_evaluations))
    if n_jobs <= 0:
        raise ValueError("'n_jobs' must be non-negative, but was {}.".format(n_jobs))
        
    with stopit.ThreadingTimeout(max_time_seconds) as c_mgr:
        logger = MultiprocessingLogger() if n_jobs > 1 else log
        evaluation_dispatcher = FunctionDispatcher(n_jobs, partial(toolbox.evaluate, logger=logger))

        max_population_size = len(start_population)
        queued_individuals_str = set()
        queued_individuals = {}
        start_time = time.time()

        evaluation_dispatcher.start()

        restart = True
        while restart:
            restart = False
            current_population = []

            log.info('Starting EA with new population.')
            for individual in start_population:
                if str(individual) not in queued_individuals_str:
                    queued_individuals_str.add(str(individual))
                    compiled_individual = toolbox.compile(individual)
                    if compiled_individual is not None:
                        identifier = evaluation_dispatcher.queue_evaluation(compiled_individual)
                        queued_individuals[identifier] = individual

            for ind_no in range(n_evaluations):
                identifier, output, _ = evaluation_dispatcher.get_next_result()
                individual = queued_individuals[identifier]

                score, evaluation_time, length = output
                if len(objectives) == 1:
                    individual.fitness.values = (score,)
                elif objectives[1] == 'time':
                    individual.fitness.values = (score, evaluation_time)
                elif objectives[1] == 'size':
                    individual.fitness.values = (score, length)
                individual.fitness.time = evaluation_time

                if n_jobs > 1:
                    logger.flush_to_log(log)

                if evaluation_callback:
                    try:
                        evaluation_callback(individual)
                    except stopit.utils.TimeoutException:
                        raise
                    except Exception:
                        # We actually want to catch any other exception here, because the callback code can be
                        # arbitrary (it can be provided by users). This excuses the catch-all Exception.
                        log.warning("Exception during callback.", exc_info=True)
                        pass
                    if time.time() - start_time > max_time_seconds:
                        log.warning("Time exceeded during callback.")
                        raise stopit.utils.TimeoutException

                if restart_callback is not None and restart_callback():
                    log.info("Restart criterion met. Restarting with new random population.")
                    restart = True
                    start_population = toolbox.population(n=max_population_size)
                    break

                current_population.append(individual)
                if len(current_population) > max_population_size:
                    to_remove = toolbox.eliminate(current_population, 1)
                    log.debug("Removed from population: {}".format(to_remove))
                    current_population.remove(to_remove[0])

                if len(current_population) > 1:
                    for _ in range(50):
                        new_individual = toolbox.create(current_population, 1)[0]
                        if str(new_individual) not in queued_individuals_str:
                            queued_individuals_str.add(str(new_individual))
                            compiled_individual = toolbox.compile(new_individual)
                            if compiled_individual is not None:
                                identifier = evaluation_dispatcher.queue_evaluation(compiled_individual)
                                queued_individuals[identifier] = new_individual
                                break
                    else:
                        log.warning('unable to create new individual.')

            evaluation_dispatcher.restart()

    # If the function is terminated early by way of a KeyboardInterrupt, there is no need to communicate to the
    # evaluation processes to shut down, since they handle the KeyboardInterrupt directly.
    # The function should not be terminated early by way of another exception, if it does, it should crash loud.
    evaluation_dispatcher.stop()
    if not c_mgr:
        log.info('Asynchronous EA terminated because maximum time has elapsed.'
                 '{} individuals have been evaluated.'.format(ind_no))
    return current_population
