from gama.genetic_programming.components import Fitness
from gama.genetic_programming.selection import eliminate_from_pareto


def test_individual_length(GNB, ForestPipeline, LinearSVC):
    assert 1 == len(list(GNB.primitives))
    assert 2 == len(list(ForestPipeline.primitives))
    assert 1 == len(list(LinearSVC.primitives))


def test_eliminate_NSGA(GNB, ForestPipeline, LinearSVC):
    GNB.fitness = Fitness((3, -2), 0, 0, 0)
    ForestPipeline.fitness = Fitness((4, -2), 0, 0, 0)
    LinearSVC.fitness = Fitness((3, -1), 0, 0, 0)

    eliminated = eliminate_from_pareto(pop=[GNB, ForestPipeline, LinearSVC], n=1)
    assert eliminated == [
        GNB
    ], "The element (3, -2) is dominated by both others and should be eliminated."

    # Check order independence
    eliminated = eliminate_from_pareto(pop=[ForestPipeline, GNB, LinearSVC], n=1)
    assert eliminated == [GNB], "Individual should be dominated regardless of order."
