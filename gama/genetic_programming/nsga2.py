""" Implementation of NSGA-II and its subroutines as defined in Deb et al. (2002)

Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002).
A fast and elitist multiobjective genetic algorithm: NSGA-II.
IEEE transactions on evolutionary computation, 6(2), 182-197.
"""
import itertools
from functools import cmp_to_key
from typing import List, Any, Callable


class NSGAMeta:

    def __init__(self, obj, metrics):
        self.obj = obj
        self.values = tuple((m(obj) for m in metrics))
        self.rank = None
        self.distance = 0
        self.dominating = []
        self.domination_counter = 0

    def dominates(self, other):
        for self_val, other_val in zip(self.values, other.values):
            if self_val <= other_val:  # or maybe <?
                return False
        return True

    def crowd_compare(self, other):
        """ Favor higher rank, if equal, favor less crowded. """
        self_better = (self.rank < other.rank or
                       (self.rank == other.rank and self.distance > other.distance))
        return -1 if self_better else 1


def NSGA2(population: List[Any], n: int, metrics: List[Callable[[Any], float]]) -> List[Any]:
    """ Selects n individuals from the population to create offspring with according to NSGA-II.

    Parameters
    ----------
    population: List[T]
        A list of objects.
    n: int
        Number of objects to pick out of population. Must be greater than 0 and smaller than len(population).
    metrics: List[Callable[[T], float]]
        List of functions which obtain the values for each dimension on which to compare elements of population.

    Returns
    -------
    selection: List[T]
        A list of size n containing a subset of population.
    """
    if n == 0 or n > len(population):
        raise ValueError(f"{n} is not a valid value for `n`, must be 0 < n < len(population) ({len(population)}).")
    population = [NSGAMeta(p, metrics) for p in population]
    selection = []
    fronts = fast_non_dominated_sort(population)
    i = 0

    while len(selection) < n:
        crowding_distance_assignment(fronts[i])
        if len(selection) + len(fronts[i]) < n:
            # Enough space for the entire Pareto front, include all
            selection += fronts[i]
        else:
            # Only the least crowded remainder is selected
            s = sorted(fronts[i], key=cmp_to_key(lambda x, y: x.crowd_compare(y)))
            selection += s[: (n - len(selection))]  # Fill up to n
        i += 1

    return [s.obj for s in selection]


def fast_non_dominated_sort(P):
    """ Sorts P into Pareto fronts. """
    fronts = [[]]
    for p, q in itertools.combinations(P, 2):
        if p.dominates(q):
            p.dominating.append(q)
            q.domination_counter += 1
        elif q.dominates(p):
            q.dominating.append(p)
            p.domination_counter += 1

    for p in P:
        if p.domination_counter == 0:
            p.rank = 1
            fronts[0].append(p)

    i = 0
    while fronts[i] != []:
        fronts.append([])
        for p in fronts[i]:
            for q in p.dominating:
                q.domination_counter -= 1
                if q.domination_counter == 0:
                    q.rank = i + 1
                    fronts[i + 1].append(q)
        i += 1
    return fronts


def crowding_distance_assignment(I):
    for m in range(len(I[0].values)):
        I = sorted(I, key=lambda x: x.values[m])
        I[0].distance = I[-1].distance = float('inf')
        for i_prev, i, i_next in zip(I, I[1:], I[2:]):
            i.distance += (i_next.values[m] - i_prev.values[m]) / (I[-1].values[m] - I[0].values[m])


# def correctness():
#     P = [(5, 6), (6, 5), (3, 5), (4, 4), (5, 3), (2, 3), (3, 2), (2.5, 2.5), (2.49, 2.51)]
#     metrics = [lambda x:x[0], lambda x:x[1]]
#     for i in range(1,10):
#         print(NSGA2(P, i, metrics))
#
#
# def performance():
#     import time
#     import numpy as np
#     start = time.time()
#
#     for i in range(10):
#         loop_time = time.time()
#         P = [p for p in np.random.random((100, 3))]
#         metrics = [lambda x: x[0], lambda x: x[1], lambda x: x[2]]
#         for N in [5, 10, 25, 100]:
#             print(f'starting i={i}, n={N}')
#             selected = NSGA2(P, n=N, metrics=metrics)
#         print(f'Four selects took {time.time() - loop_time}')
#     print(f'total time: {time.time() - start}')
#
#
# if __name__ == '__main__':
#     performance()
