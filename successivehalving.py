import math
from functools import partial

from gama.utilities.generic.async_executor import AsyncExecutor, wait_first_complete
from gama.genetic_programming.compilers.scikitlearn import evaluate_individual
"""
TODO:
 - instead of list, use a min-heap by rung.
 - promoted pipelines as set and set-intersection to determine promotability?
"""


def asha(operations, start_candidates=None, timeout=300,  # General Search Hyperparameters
         reduction_factor=3, minimum_resource=100, maximum_resource=10, minimum_early_stopping_rate=1):  # Algorithm Specific
    rungs = range(math.floor(math.log(maximum_resource/minimum_resource, reduction_factor))
                  - minimum_early_stopping_rate)
    pipelines_by_rung = {rung: [] for rung in rungs}
    promoted_pipelines = {rung: [] for rung in rungs}

    def get_job():
        for rung, pipelines in list(reversed(list(pipelines_by_rung.items())))[1:]:
            candidates = sorted(pipelines)[:math.floor(len(pipelines) / reduction_factor)]
            promotable = [candidate for candidate in candidates if candidate not in promoted_pipelines[rung]]
            if len(promotable) > 0:
                return promotable[0], rung + 1

        if start_candidates is not None and len(start_candidates) > 0:
            return start_candidates.pop(), 0
        else:
            return operations.individual(), 0

    futures = set()
    with AsyncExecutor() as async_:
        for _ in range(8):
            pipeline, rung = get_job()
            n = minimum_resource * (reduction_factor ** (minimum_early_stopping_rate + rung))
            futures.add(async_.submit(operations.evaluate, pipeline, subsample=n))

        for _ in range(100):
            done, futures = wait_first_complete(futures)
            for loss, pipeline, rung in [future.result() for future in done]:
                pipelines_by_rung[rung] = (loss, pipeline)
            pipeline, rung = get_job()
            futures.add(async_.submit(operations.evaluate, (pipeline, rung)))

    highest_rung_reached = max(rung for rung, pipelines in pipelines_by_rung.items() if pipelines != [])
    if highest_rung_reached != rungs:
        raise RuntimeWarning("Highest rung not reached.")

    return sorted(pipelines_by_rung[highest_rung_reached])[0]


if __name__ == '__main__':
    from gama import GamaClassifier
    from sklearn.datasets import load_digits
    g = GamaClassifier()
    X, y = load_digits(return_X_y=True)

    g._operator_set.evaluate = partial
    pipeline = asha(g._operator_set, start_candidates=[g._operator_set.individual() for _ in range(10)])
