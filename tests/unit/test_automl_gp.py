from gama.genetic_programming.components import Fitness
from gama.genetic_programming.selection import eliminate_from_pareto


def test_individual_length(GNB, ForestPipeline, LinearSVC):
    assert 1 == len(list(GNB.primitives))
    assert 2 == len(list(ForestPipeline.primitives))
    assert 1 == len(list(LinearSVC.primitives))
