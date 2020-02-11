import pytest

from gama.genetic_programming.crossover import random_crossover, crossover_primitives, _shared_terminals, crossover_terminals
from .unit_fixtures import pset, BernoulliNBStandardScaler, MultinomialNBRobustScaler, GaussianNB, BernoulliNBThreeScalers


def test_shared_terminals(BernoulliNBStandardScaler, MultinomialNBRobustScaler, GaussianNB):
    """ Test shared terminals are found, if they exist. """
    def cmp_len(n, generator):
        return n == len(list(generator))

    assert cmp_len(0, _shared_terminals(BernoulliNBStandardScaler, BernoulliNBStandardScaler, value_match='different'))
    assert cmp_len(2, _shared_terminals(BernoulliNBStandardScaler, BernoulliNBStandardScaler, value_match='equal'))
    assert cmp_len(2, _shared_terminals(BernoulliNBStandardScaler, BernoulliNBStandardScaler, value_match='all'))

    assert cmp_len(1, _shared_terminals(BernoulliNBStandardScaler, MultinomialNBRobustScaler, value_match='different'))
    assert cmp_len(1, _shared_terminals(BernoulliNBStandardScaler, MultinomialNBRobustScaler, value_match='equal'))
    assert cmp_len(2, _shared_terminals(BernoulliNBStandardScaler, MultinomialNBRobustScaler, value_match='all'))

    assert cmp_len(0, _shared_terminals(BernoulliNBStandardScaler, GaussianNB, value_match='all'))


def test_crossover_primitives(BernoulliNBStandardScaler, MultinomialNBRobustScaler):
    """ Two individuals of at least length 2 should produce two new individuals with crossover. """
    ind1_copy, ind2_copy = BernoulliNBStandardScaler.copy_as_new(), MultinomialNBRobustScaler.copy_as_new()
    # Cross-over is in-place
    crossover_primitives(BernoulliNBStandardScaler, MultinomialNBRobustScaler)
    # Both parents and children should be unique
    all_individuals = [BernoulliNBStandardScaler, MultinomialNBRobustScaler, ind1_copy, ind2_copy]
    assert 4 == len({ind.pipeline_str() for ind in all_individuals})
    assert ind1_copy.pipeline_str() != BernoulliNBStandardScaler.pipeline_str()


def test_crossover_terminal(BernoulliNBStandardScaler, MultinomialNBRobustScaler):
    """ Two individuals with shared Terminals should produce two new individuals with crossover. """
    ind1_copy, ind2_copy = BernoulliNBStandardScaler.copy_as_new(), MultinomialNBRobustScaler.copy_as_new()
    # Cross-over is in-place
    crossover_terminals(BernoulliNBStandardScaler, MultinomialNBRobustScaler)
    # Both parents and children should be unique
    assert len({ind.pipeline_str() for ind in [
        BernoulliNBStandardScaler, MultinomialNBRobustScaler, ind1_copy, ind2_copy]}) == 4
    assert ind1_copy.pipeline_str() != BernoulliNBStandardScaler.pipeline_str()


def test_crossover(BernoulliNBStandardScaler, MultinomialNBRobustScaler):
    """ Two eligible individuals should produce two new individuals with crossover. """
    ind1_copy, ind2_copy = BernoulliNBStandardScaler.copy_as_new(), MultinomialNBRobustScaler.copy_as_new()
    # Cross-over is in-place
    random_crossover(BernoulliNBStandardScaler, MultinomialNBRobustScaler)
    # Both parents and children should be unique
    assert len({ind.pipeline_str() for ind in [
        BernoulliNBStandardScaler, MultinomialNBRobustScaler, ind1_copy, ind2_copy]}) == 4
    assert ind1_copy.pipeline_str() != BernoulliNBStandardScaler.pipeline_str()


def test_crossover_max_length_exceeded(BernoulliNBThreeScalers, MultinomialNBRobustScaler):
    """ Raise ValueError if either provided individual exceeds `max_length`. """
    with pytest.raises(ValueError) as error:
        random_crossover(BernoulliNBThreeScalers, MultinomialNBRobustScaler, max_length=2)

    with pytest.raises(ValueError) as error:
        random_crossover(MultinomialNBRobustScaler, BernoulliNBThreeScalers, max_length=2)


def test_crossover_max_length(BernoulliNBThreeScalers):
    """ Setting `max_length` affects maximum produced length, and maximum length only. """
    primitives_in_parent = len(BernoulliNBThreeScalers.primitives)
    produced_lengths = []
    for _ in range(60):  # guarantees all allowed length pipelines are produced with probability >0.999
        ind1, ind2 = random_crossover(BernoulliNBThreeScalers.copy_as_new(),
                                      BernoulliNBThreeScalers.copy_as_new(),
                                      max_length=primitives_in_parent)
        # Only the first child is guaranteed to contain at most `max_length` primitives.
        produced_lengths.append(len(ind1.primitives))
    assert set(produced_lengths) == {2, 3, 4}
