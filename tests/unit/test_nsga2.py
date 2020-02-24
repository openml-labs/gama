from typing import List, Tuple
from gama.genetic_programming.nsga2 import (
    NSGAMeta,
    fast_non_dominated_sort,
    crowding_distance_assignment,
)


def _tuples_to_NSGAMeta(tuples: List[Tuple]) -> List[NSGAMeta]:
    """ Converts a list of tuples to NSGAMeta objects. """
    # Can't declare it directly in a loop as it does not create a new scope.
    def fetch_value(i):
        return lambda x: x[i]

    metrics = [fetch_value(i) for i in range(len(tuples[0]))]
    return [NSGAMeta(t, metrics) for t in tuples]


def test_nsgameta_value_assignment():
    pareto = _tuples_to_NSGAMeta([(3, 5), (5, 3), (4, 4)])
    three_five, five_three, four_four = pareto

    assert three_five.values == (3, 5)
    assert five_three.values == (5, 3)
    assert four_four.values == (4, 4)


def test_dominates():
    pareto = _tuples_to_NSGAMeta([(3, 5), (5, 3), (2, 4)])
    three_five, five_three, two_four = pareto

    assert not three_five.dominates(five_three)
    assert not five_three.dominates(three_five)

    assert three_five.dominates(two_four)
    assert not two_four.dominates(three_five)

    assert not five_three.dominates(two_four)
    assert not two_four.dominates(five_three)


def test_crowding_distance_assignment():
    pareto = _tuples_to_NSGAMeta([(3, 5), (5, 3), (4, 4)])
    three_five, five_three, four_four = pareto
    crowding_distance_assignment(pareto)

    assert three_five.distance == float("inf")
    assert five_three.distance == float("inf")
    assert four_four.distance == 2


def test_crowding_distance_assignment_inf():
    pareto = _tuples_to_NSGAMeta([(3, float("inf")), (5, 3), (4, 4)])
    three_inf, five_three, four_four = pareto
    crowding_distance_assignment(pareto)

    assert three_inf.distance == float("inf")
    assert five_three.distance == float("inf")
    #  In our implementation, we ignore 'axis' that contain inf values.
    assert four_four.distance == 1


def test_crowd_compare():
    pareto = _tuples_to_NSGAMeta([(3, 5), (5, 3), (4, 4), (4.01, 3.99), (4.5, 3.5)])
    three_five, five_three, four_four, approx_four_four, half_half = pareto
    fast_non_dominated_sort(pareto)  # assigns rank
    crowding_distance_assignment(pareto)  # assigns distance

    assert all([three_five.crowd_compare(other) == -1 for other in pareto[2:]])
    assert all([five_three.crowd_compare(other) == -1 for other in pareto[2:]])
