""" Implementation of NSGA-II and its subroutines as defined in Deb et al. (2002)

Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002).
A fast and elitist multiobjective genetic algorithm: NSGA-II.
IEEE transactions on evolutionary computation, 6(2), 182-197.
"""
import itertools
import random
from functools import cmp_to_key
from typing import List, Any, Callable
import numpy as np


class NSGAMeta:
    def __init__(self, obj, metrics):
        self.obj = obj
        self.values = tuple((m(obj) for m in metrics))
        self.rank = None
        self.distance = 0
        self.dominating = []
        self.domination_counter = 0

    def dominates(self, other: "NSGAMeta"):
        for self_val, other_val in zip(self.values, other.values):
            if self_val <= other_val:  # or maybe <?
                return False
        return True

    def crowd_compare(self, other: "NSGAMeta"):
        """ Favor higher rank, if equal, favor less crowded. """
        self_better = self.rank < other.rank or (
            self.rank == other.rank and self.distance > other.distance
        )
        return -1 if self_better else 1


def nsga2_select(
    population: List[Any], n: int, metrics: List[Callable[[Any], float]]
) -> List[Any]:
    """ Select n pairs from the population.

     Selection is done through binary tournament selection based on crowding distance.
     Parent pairs may be repeated, but each parent pair consists of two unique parents.
     The population must be at least size 3 (otherwise it is trivial or impossible).
    """
    if len(population) < 3:
        raise ValueError("population must be at least size 3 for a pair to be selected")

    # Entire population is returned, but with rank and distance information.
    candidates = nsga2(population, n=len(population), metrics=metrics, return_meta=True)

    def select_one(exclude=None):
        selected = random.sample(candidates, k=3)
        ind1, ind2 = [s for s in selected if s != exclude][:2]
        return ind1 if ind1.crowd_compare(ind2) < 0 else ind2

    selected = []
    for _ in range(n):
        first = select_one()
        second = select_one(exclude=first)
        selected.append((first.obj, second.obj))
    return selected


def nsga2(
    population: List[Any],
    n: int,
    metrics: List[Callable[[Any], float]],
    return_meta: bool = False,
) -> List[Any]:
    """ Selects n individuals from the population for offspring according to NSGA-II.

    Parameters
    ----------
    population: List[T]
        A list of objects.
    n: int
        Number of objects to pick out of population.
        Must be greater than 0 and smaller than len(population).
    metrics: List[Callable[[T], float]]
        List of functions which obtain the values for each dimension
        on which to compare elements of population.
    return_meta: bool (default=False)
        If True, return the selected individuals wrapped in a NSGAMeta class,
        with information such as rank and distance.
        If False, return the selected individuals as they were passed to this function.

    Returns
    -------
    selection: List[T]
        A list of size n containing a subset of population.
    """
    if n == 0 or n > len(population):
        raise ValueError(f"n is {n} must be 0 < n < len(population)={len(population)}")
    population = [NSGAMeta(p, metrics) for p in population]
    selection: List[NSGAMeta] = []
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

    return selection if return_meta else [s.obj for s in selection]


def fast_non_dominated_sort(P: List[NSGAMeta]) -> List[List[NSGAMeta]]:
    """ Sorts P into Pareto fronts. """
    fronts: List[List[NSGAMeta]] = [[]]
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


def crowding_distance_assignment(I: List[NSGAMeta]) -> None:
    for m in range(len(I[0].values)):
        I = sorted(I, key=lambda x: x.values[m])  # noqa: E741 'I' is name in paper
        I[0].distance = I[-1].distance = float("inf")
        if (
            I[-1].values[m] == I[0].values[m]
            or np.isinf(I[0].values[m])
            or np.isinf(I[-1].values[m])
        ):
            # Would raise divisionbyzero later, or give other numerical warnings.
            # This typically happens only for the worst pareto front(s),
            # so the inaccuracy in crowding distance for the remainder is not a concern.
            # Might consider immediately removing failing individuals.
            continue

        for i_prev, i, i_next in zip(I, I[1:], I[2:]):
            i.distance += (i_next.values[m] - i_prev.values[m]) / (
                I[-1].values[m] - I[0].values[m]
            )
