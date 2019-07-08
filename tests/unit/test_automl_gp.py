from gama.genetic_programming.components import Fitness
from gama.genetic_programming.selection import eliminate_from_pareto
from .unit_fixtures import pset, GaussianNB, RandomForestPipeline, LinearSVC


def test_individual_length(GaussianNB, RandomForestPipeline, LinearSVC):
    assert 1 == len(list(GaussianNB.primitives))
    assert 2 == len(list(RandomForestPipeline.primitives))
    assert 1 == len(list(LinearSVC.primitives))


def test_eliminate_NSGA(GaussianNB, RandomForestPipeline, LinearSVC):
    GaussianNB.fitness = Fitness((3, -2), 0, 0, 0)
    RandomForestPipeline.fitness = Fitness((4, -2), 0, 0, 0)
    LinearSVC.fitness = Fitness((3, -1), 0, 0, 0)

    eliminated = eliminate_from_pareto(pop=[GaussianNB, RandomForestPipeline, LinearSVC], n=1)
    assert eliminated == [GaussianNB], "The element (3, -2) is dominated by both others and should be eliminated."

    # Check order independence
    eliminated = eliminate_from_pareto(pop=[RandomForestPipeline, GaussianNB, LinearSVC], n=1)
    assert eliminated == [GaussianNB], "Individual should be dominated regardless of order."
