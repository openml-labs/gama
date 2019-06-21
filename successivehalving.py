import math
from functools import partial

from gama.utilities.generic.async_executor import AsyncExecutor, wait_first_complete
from gama.genetic_programming.compilers.scikitlearn import evaluate_individual
from gama.genetic_programming.algorithms.metrics import Metric
from gama.utilities.preprocessing import define_preprocessing_steps
"""
TODO:
 - instead of list, use a min-heap by rung.
 - promoted pipelines as set and set-intersection to determine promotability?
"""


def asha(operations, start_candidates=None, timeout=300,  # General Search Hyperparameters
         reduction_factor=3, minimum_resource=100, maximum_resource=1700, minimum_early_stopping_rate=1):  # Algorithm Specific
    rungs = range(math.floor(math.log(maximum_resource/minimum_resource, reduction_factor))
                  - minimum_early_stopping_rate
                  + 1)
    individuals_by_rung = {rung: [] for rung in rungs}
    promoted_individuals = {rung: [] for rung in rungs}

    def get_job():
        for rung, individuals in list(reversed(list(individuals_by_rung.items())))[1:]:
            candidates = sorted(individuals)[:math.floor(len(individuals) / reduction_factor)]
            promotable = [candidate for candidate in candidates if candidate not in promoted_individuals[rung]]
            if len(promotable) > 0:
                return promotable[0], rung + 1

        if start_candidates is not None and len(start_candidates) > 0:
            return start_candidates.pop(), 0
        else:
            return operations.individual(), 0

    futures = set()
    with AsyncExecutor() as async_:
        for _ in range(8):
            individual, rung = get_job()
            n = minimum_resource * (reduction_factor ** (minimum_early_stopping_rate + rung))
            futures.add(async_.submit(operations.evaluate, individual, rung, subsample=n))

        for _ in range(100):
            done, futures = wait_first_complete(futures)
            for loss, individual, rung in [future.result() for future in done]:
                individuals_by_rung[rung] = (loss, individual)
            individual, rung = get_job()
            n = minimum_resource * (reduction_factor ** (minimum_early_stopping_rate + rung))
            futures.add(async_.submit(operations.evaluate, individual, rung, subsample=n))

    highest_rung_reached = max(rung for rung, individuals in individuals_by_rung.items() if individuals != [])
    if highest_rung_reached != rungs:
        raise RuntimeWarning("Highest rung not reached.")

    return sorted(individuals_by_rung[highest_rung_reached])[0]


def evaluate_on_rung(individual, rung, *args, **kwargs):
    individual = evaluate_individual(individual, *args, **kwargs)
    return individual.fitness.values[0], individual, rung


if __name__ == '__main__':
    from gama import GamaClassifier
    from sklearn.datasets import load_digits
    g = GamaClassifier()
    X, y = load_digits(return_X_y=True)

    g.fit(X, y)
    #steps = define_preprocessing_steps(X, max_extra_features_created=None, max_categories_for_one_hot=10)
    #g._operator_set._safe_compile = partial(g._operator_set._compile, preprocessing_steps=steps)
    #g._operator_set.evaluate = partial(evaluate_on_rung, evaluate_pipeline_length=False, X=X, y_train=y, y_score=y, timeout=300, metrics=[Metric.from_string('accuracy')])
    #start_pop = [g._operator_set.individual() for _ in range(10)]
    #pipeline = asha(g._operator_set, start_candidates=start_pop)
    #print(pipeline)
