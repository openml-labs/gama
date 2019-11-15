""" Implementation of NSGA-II and its subroutines as defined in Deb et al. (2002)

Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002).
A fast and elitist multiobjective genetic algorithm: NSGA-II.
IEEE transactions on evolutionary computation, 6(2), 182-197.
"""
from functools import partial, cmp_to_key
import time




def NSGAII(population, n, metrics):
    if n == 0 or n > len(population):
        raise ValueError(f"{n} is not a valid value for `n`, must be 0 < n < len(population) ({len(population)}).")
    selection = []
    distances = {}
    dominate = partial(dominates, metrics=metrics)
    fronts, ranks = fast_non_dominated_sort(population, dominates=dominate)
    i = 0

    while len(selection) < n:
        distances.update(crowding_distance_assignment(fronts[i], metrics))
        if len(selection) + len(fronts[i]) < n:
            # Enough space for the entire Pareto front, include all
            selection += fronts[i]
        else:
            # Only the least crowded remainder is selected
            crowd_comparison = partial(crowded_comparison, ranks=ranks, distances=distances)
            s = sorted(fronts[i], key=cmp_to_key(crowd_comparison))
            selection += s[: (n - len(selection))]  # Fill up to n
        i += 1

    return selection


def fast_non_dominated_sort(P, dominates):
    """ Sorts P into Pareto fronts. """
    fronts = [[]]
    S = {id(p): [] for p in P}  # key_dominates_value
    domination_counter = {id(p): 0 for p in P}
    ranks = {id(p): None for p in P}
    for p in P:
        for q in P:
            if dominates(p, q):
                S[id(p)].append(q)
            elif dominates(q, p):
                domination_counter[id(p)] += 1
        if domination_counter[id(p)] == 0:
            ranks[id(p)] = 1
            fronts[0].append(p)
    i = 0
    while fronts[i] != []:
        fronts.append([])
        for p in fronts[i]:
            for q in S[id(p)]:
                domination_counter[id(q)] -= 1
                if domination_counter[id(q)] == 0:
                    ranks[id(q)] = i + 1
                    fronts[i + 1].append(q)
        i += 1
    return fronts, ranks


def crowding_distance_assignment(I, metrics):
    distance = {id(i): 0 for i in I}
    for m in metrics:
        I = sorted(I, key=m)
        distance[id(I[0])] = distance[id(I[-1])] = float('inf')
        for i_prev, i, i_next in zip(I, I[1:], I[2:]):
            distance[id(i)] += (m(i_next) - m(i_prev)) / (m(I[-1]) - m(I[0]))
    return distance


def crowded_comparison(p, q, ranks, distances):
    """ -1 if p <_n q, 1 otherwise """
    p_better_q = (ranks[id(p)] < ranks[id(q)] or
                  (ranks[id(p)] == ranks[id(q)] and distances[id(p)] > distances[id(q)]))
    return -1 if p_better_q else 1


def dominates(p, q, metrics):
    """ P dominates Q."""
    for m in metrics:
        if m(p) <= m(q):
            return False
    return True


if __name__ == '__main__':
    import numpy as np
    start = time.time()

    for i in range(10):
        loop_time = time.time()
        P = [p for p in np.random.random((5000, 3))]
        metrics = [lambda x: x[0], lambda x: x[1], lambda x: x[2]]
        for N in [5, 10, 25, 100]:
            print(f'starting i={i}, n={N}')
            selected = NSGAII(P, n=N, metrics=metrics)
        print(f'Four selects took {time.time() - loop_time}')
    print(f'total time: {time.time() - start}')
