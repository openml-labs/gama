import logging
from functools import partial

import stopit

from ..utilities.mp_logger import MultiprocessingLogger
from ..utilities.function_dispatcher import FunctionDispatcher

log = logging.getLogger(__name__)


def async_ea(objectives, start_population, toolbox, evaluation_callback=None, restart_callback=None, n_evaluations=10000, n_jobs=1):
    logger = MultiprocessingLogger()
    evaluation_dispatcher = FunctionDispatcher(n_jobs, partial(toolbox.evaluate, logger=logger))

    max_population_size = len(start_population)
    queued_individuals_str = set()
    queued_individuals = {}

    try:
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

            for _ in range(n_evaluations):
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

                logger.flush_to_log(log)
                if evaluation_callback:
                    evaluation_callback(individual)

                if restart_callback is not None and restart_callback():
                    log.info("Restart criterion met. Restarting with new random population.")
                    restart = True
                    start_population = toolbox.population(n=max_population_size)
                    break

                current_population.append(individual)
                if len(current_population) > max_population_size:
                    to_remove = toolbox.eliminate(current_population, 1)
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
    except stopit.utils.TimeoutException:
        log.info("Shutting down EA due to Timeout.")
        evaluation_dispatcher.stop()
        raise
    except KeyboardInterrupt:
        log.info('Shutting down EA due to KeyboardInterrupt.')
        # No need to communicate to processes since they also handle the KeyboardInterrupt directly.
    except Exception:
        log.error('Unexpected exception in asynchronous parallel algorithm.', exc_info=True)
        # Even in the event of an error we want the helper processes to shut down.
        evaluation_dispatcher.stop()
        raise

    evaluation_dispatcher.stop()
    return current_population
