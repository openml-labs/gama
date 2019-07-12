import logging
import math
from typing import List

import stopit

from gama.logging.machine_logging import TOKENS, log_parseable_event
from gama.utilities.generic.async_executor import AsyncExecutor
from gama.genetic_programming.compilers.scikitlearn import evaluate_individual
from gama.genetic_programming.components.individual import Individual

"""
TODO:
 - instead of list, use a min-heap by rung.
 - promoted pipelines as set and set-intersection to determine promotability?
"""

log = logging.getLogger(__name__)


def asha(operations, output: List[Individual], start_candidates=None,  # General Search Hyperparameters
         reduction_factor=3, minimum_resource=100, maximum_resource=1700, minimum_early_stopping_rate=1):  # Algorithm Specific
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
    try:
        with AsyncExecutor() as async_:
            def start_new_job():
                individual, rung = get_job()
                futures.add(async_.submit(operations.evaluate, individual, rung, subsample=resource_for_rung[rung]))

            for _ in range(8):
                start_new_job()

            while sum(map(len, individuals_by_rung.values())) < 3000:
                done, futures = operations.wait_first_complete(futures)
                for individual, loss, rung in [future.result() for future in done]:
                    individuals_by_rung[rung].append((loss, individual))
                    if rung == max(rungs):
                        log_parseable_event(log, TOKENS.EVALUATION_RESULT, individual.fitness.start_time,
                                            individual.fitness.wallclock_time, individual.fitness.process_time,
                                            individual.fitness.values, individual._id, individual.pipeline_str())
                    start_new_job()

            highest_rung_reached = max(rungs)
    except stopit.TimeoutException:
        log.info('ASHA ended due to timeout.')
        highest_rung_reached = max(rung for rung, individuals in individuals_by_rung.items() if individuals != [])
        if highest_rung_reached != max(rungs):
            raise RuntimeWarning("Highest rung not reached.")
    finally:
        for rung, individuals in individuals_by_rung.items():
            log.info('[{}] {}'.format(rung, len(individuals)))

    return list(map(lambda p: p[1], individuals_by_rung[highest_rung_reached]))


def evaluate_on_rung(individual, rung, *args, **kwargs):
    individual = evaluate_individual(individual, *args, **kwargs)
    return individual, individual.fitness.values[0], rung
