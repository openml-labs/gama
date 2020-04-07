import pytest

from gama.genetic_programming.components import Fitness
from gama.genetic_programming.selection import (
    create_from_population,
    eliminate_from_pareto,
)


def test_eliminate_more_than_1_from_pareto():
    with pytest.raises(NotImplementedError):
        eliminate_from_pareto([], 2)


def test_eliminate_from_pareto(GNB, ForestPipeline, LinearSVC):
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


def test_create_from_population(opset, GNB, ForestPipeline, LinearSVC):
    GNB.fitness = Fitness((3, -2), 0, 0, 0)
    ForestPipeline.fitness = Fitness((4, -2), 0, 0, 0)
    LinearSVC.fitness = Fitness((3, -1), 0, 0, 0)
    parents = [GNB, ForestPipeline, LinearSVC]

    new = create_from_population(opset, pop=parents, n=1, cxpb=0.5, mutpb=0.5)
    assert 1 == len(new)
    assert new[0]._id not in map(lambda i: i._id, parents)
    assert new[0].pipeline_str() not in map(lambda i: i.pipeline_str(), parents)

    # Not sure how to test NSGA2 selection is applied correctly
    # Can do it many times and see if the best individuals are parent more
    # With these fixtures, crossover can't be tested either.
