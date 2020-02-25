import pytest

from gama.genetic_programming.crossover import (
    random_crossover,
    crossover_primitives,
    _shared_terminals,
    crossover_terminals,
)


def test_shared_terminals(SS_BNB, RS_MNB, GNB):
    """ Test shared terminals are found, if they exist. """
    assert 0 == len(list(_shared_terminals(SS_BNB, SS_BNB, value_match="different")))
    assert 2 == len(list(_shared_terminals(SS_BNB, SS_BNB, value_match="equal")))
    assert 2 == len(list(_shared_terminals(SS_BNB, SS_BNB, value_match="all")))

    assert 1 == len(list(_shared_terminals(SS_BNB, RS_MNB, value_match="different")))
    assert 1 == len(list(_shared_terminals(SS_BNB, RS_MNB, value_match="equal")))
    assert 2 == len(list(_shared_terminals(SS_BNB, RS_MNB, value_match="all")))

    assert 0 == len(list(_shared_terminals(SS_BNB, GNB, value_match="all")))


def test_crossover_primitives(SS_BNB, RS_MNB):
    """ Two individuals of at least length 2 produce two new ones with crossover. """
    ind1_copy, ind2_copy = SS_BNB.copy_as_new(), RS_MNB.copy_as_new()

    # Cross-over is in-place
    crossover_primitives(SS_BNB, RS_MNB)
    # Both parents and children should be unique
    all_individuals = [SS_BNB, RS_MNB, ind1_copy, ind2_copy]

    assert 4 == len({ind.pipeline_str() for ind in all_individuals})
    assert ind1_copy.pipeline_str() != SS_BNB.pipeline_str()


def test_crossover_terminal(SS_BNB, RS_MNB):
    """ Two individuals with shared Terminals produce two new ones with crossover. """
    ind1_copy, ind2_copy = SS_BNB.copy_as_new(), RS_MNB.copy_as_new()
    # Cross-over is in-place
    crossover_terminals(SS_BNB, RS_MNB)
    # Both parents and children should be unique
    all_individuals = [SS_BNB, RS_MNB, ind1_copy, ind2_copy]

    assert 4 == len({ind.pipeline_str() for ind in all_individuals})
    assert ind1_copy.pipeline_str() != SS_BNB.pipeline_str()


def test_crossover(SS_BNB, RS_MNB):
    """ Two eligible individuals should produce two new individuals with crossover. """
    ind1_copy, ind2_copy = SS_BNB.copy_as_new(), RS_MNB.copy_as_new()
    # Cross-over is in-place
    random_crossover(SS_BNB, RS_MNB)
    # Both parents and children should be unique
    all_individuals = [SS_BNB, RS_MNB, ind1_copy, ind2_copy]
    assert 4 == len({ind.pipeline_str() for ind in all_individuals})
    assert ind1_copy.pipeline_str() != SS_BNB.pipeline_str()


def test_crossover_max_length_exceeded(SS_RBS_SS_BNB, RS_MNB):
    """ Raise ValueError if either provided individual exceeds `max_length`. """
    with pytest.raises(ValueError) as _:
        random_crossover(SS_RBS_SS_BNB, RS_MNB, max_length=2)

    with pytest.raises(ValueError) as _:
        random_crossover(RS_MNB, SS_RBS_SS_BNB, max_length=2)


def test_crossover_max_length(SS_RBS_SS_BNB):
    """ Setting `max_length` affects only maximum produced length. """
    primitives_in_parent = len(SS_RBS_SS_BNB.primitives)
    produced_lengths = []
    for _ in range(60):  # guarantees all length pipelines are produced with prob >0.999
        ind1, ind2 = random_crossover(
            SS_RBS_SS_BNB.copy_as_new(),
            SS_RBS_SS_BNB.copy_as_new(),
            max_length=primitives_in_parent,
        )
        # Only the first child is guaranteed to contain at most `max_length` primitives.
        produced_lengths.append(len(ind1.primitives))
    assert {2, 3, 4} == set(produced_lengths)
