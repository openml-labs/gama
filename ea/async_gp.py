import logging
from functools import partial

import stopit

from ..utilities.mp_logger import MultiprocessingLogger
from ..utilities.function_dispatcher import FunctionDispatcher

log = logging.getLogger(__name__)


def async_ea(objectives, population, toolbox, evaluation_callback=None, restart_callback=None, n_evaluations=10000, n_jobs=1):
    logger = MultiprocessingLogger()
    evaluation_dispatcher = FunctionDispatcher(n_jobs, partial(toolbox.evaluate, logger=logger), toolbox)
    max_population_size = len(population)

    try:
        evaluation_dispatcher.start()

        restart = True
        while restart:
            restart = False
            current_population = []

            log.info('Starting EA with new population.')
            for ind in population:
                evaluation_dispatcher.queue_evaluation(ind)

            for _ in range(n_evaluations):
                individual, output = evaluation_dispatcher.get_next_result()
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
                    population = toolbox.population(n=max_population_size)
                    break

                current_population.append(individual)
                if len(current_population) > max_population_size:
                    to_remove = toolbox.eliminate(current_population, 1)
                    current_population.remove(to_remove[0])

                if len(current_population) > 1:
                    new_individual = toolbox.create(current_population, 1)[0]
                    evaluation_dispatcher.queue_evaluation(new_individual)

            evaluation_dispatcher.cancel_all_evaluations()
    except stopit.utils.TimeoutException:
        log.info("Shutting down EA due to Timeout.")
        evaluation_dispatcher.shut_down()
        raise
    except KeyboardInterrupt:
        log.info('Shutting down EA due to KeyboardInterrupt.')
        # No need to communicate to processes since they also handle the KeyboardInterrupt directly.
    except Exception:
        log.error('Unexpected exception in asynchronous parallel algorithm.', exc_info=True)
        # Even in the event of an error we want the helper processes to shut down.
        evaluation_dispatcher.shut_down()
        raise

    evaluation_dispatcher.shut_down()
    return current_population
