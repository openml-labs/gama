import logging
import math
import time
from functools import partial

import stopit

from gama.utilities.generic.async_executor import AsyncExecutor, wait_first_complete
from gama.genetic_programming.compilers.scikitlearn import evaluate_individual

"""
TODO:
 - instead of list, use a min-heap by rung.
 - promoted pipelines as set and set-intersection to determine promotability?
"""

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


def asha(operations, start_candidates=None, max_time_seconds=300, evaluation_callback=None,  # General Search Hyperparameters
         reduction_factor=3, minimum_resource=100, maximum_resource=1700, minimum_early_stopping_rate=1):  # Algorithm Specific
    start_time = time.time()

    def exceed_timeout():
        return (time.time() - start_time) > max_time_seconds

    # Note that here we index the rungs by all possible rungs (0..ceil(log_eta(R/r))), and ignore the first
    # minimum_early_stopping_rate rungs. This contrasts the paper where rung 0 refers to the first used one.
    max_rung = math.ceil(math.log(maximum_resource/minimum_resource, reduction_factor))
    rungs = range(minimum_early_stopping_rate, max_rung + 1)
    resource_for_rung = {rung: min(minimum_resource * (reduction_factor ** rung), maximum_resource) for rung in rungs}

    # Should we just use lists of lists/heaps instead?
    individuals_by_rung = {rung: [] for rung in reversed(rungs)}  # Highest rungs first is how we typically access them
    promoted_individuals = {rung: [] for rung in reversed(rungs)}

    def get_job():
        for rung, individuals in list(individuals_by_rung.items())[1:]:
            # This is not in the paper code but is derived from fig 2b
            n_to_promote = math.floor(len(individuals) / reduction_factor)
            if n_to_promote - len(promoted_individuals[rung]) > 0:
                # Problem: equal loss falls back on comparison of individual
                candidates = list(sorted(individuals, key=lambda t: t[0], reverse=True))[:n_to_promote]
                promotable = [candidate for candidate in candidates if candidate not in promoted_individuals[rung]]
                if len(promotable) > 0:
                    promoted_individuals[rung].append(promotable[0])
                    return promotable[0][1], rung + 1

        if start_candidates is not None and len(start_candidates) > 0:
            return start_candidates.pop(), minimum_early_stopping_rate
        else:
            return operations.individual(), minimum_early_stopping_rate

    futures = set()
    with AsyncExecutor() as async_, stopit.ThreadingTimeout(max_time_seconds) as timer:
        def start_new_job():
            individual, rung = get_job()
            futures.add(async_.submit(operations.evaluate, individual, rung, subsample=resource_for_rung[rung]))

        for _ in range(8):
            start_new_job()

        while sum(map(len, individuals_by_rung.values())) < 100:
            done, futures = wait_first_complete(futures)
            for loss, individual, rung in [future.result() for future in done]:
                individuals_by_rung[rung].append((loss, individual))
                start_new_job()
                if evaluation_callback is not None:
                    _safe_outside_call(partial(evaluation_callback, individual), exceed_timeout)

    highest_rung_reached = max(rung for rung, individuals in individuals_by_rung.items() if individuals != [])
    for rung, individuals in individuals_by_rung.items():
        log.info('[{}] {}'.format(rung, len(individuals)))
    if highest_rung_reached != max(rungs):
        raise RuntimeWarning("Highest rung not reached.")
    if not timer:
        log.info('ASHA terminated because maximum time has elapsed.'
                 '{} individuals have been evaluated.'.format(sum(map(len, individuals_by_rung.values()))))

    return list(map(lambda p: p[1], individuals_by_rung[highest_rung_reached]))


def evaluate_on_rung(individual, rung, *args, **kwargs):
    individual = evaluate_individual(individual, *args, **kwargs)
    return individual.fitness.values[0], individual, rung
