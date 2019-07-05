import pytest

from gama.genetic_programming.components import Individual
from gama.genetic_programming.mutation import crossover, crossover_primitives, crossover_terminals, shared_terminals
from gama import GamaClassifier


@pytest.fixture
def pset():
    return GamaClassifier()._pset


@pytest.fixture
def GaussianNB(pset):
    return Individual.from_string("GaussianNB(data)", pset, None)


@pytest.fixture
def BernoulliNBStandardScaler(pset):
    return Individual.from_string("BernoulliNB(StandardScaler(data), alpha=0.1, fit_prior=True)", pset, None)


@pytest.fixture
def MultinomialNBRobustScaler(pset):
    return Individual.from_string("MultinomialNB(RobustScaler(data), alpha=1.0, fit_prior=True)", pset, None)


def test_shared_terminals(BernoulliNBStandardScaler, MultinomialNBRobustScaler, GaussianNB):
    """ Test shared terminals are found, if they exist. """
    def cmp_len(n, generator):
        return n == len(list(generator))

    assert cmp_len(0, shared_terminals(BernoulliNBStandardScaler, BernoulliNBStandardScaler, value_match='different'))
    assert cmp_len(2, shared_terminals(BernoulliNBStandardScaler, BernoulliNBStandardScaler, value_match='equal'))
    assert cmp_len(2, shared_terminals(BernoulliNBStandardScaler, BernoulliNBStandardScaler, value_match='all'))

    assert cmp_len(1, shared_terminals(BernoulliNBStandardScaler, MultinomialNBRobustScaler, value_match='different'))
    assert cmp_len(1, shared_terminals(BernoulliNBStandardScaler, MultinomialNBRobustScaler, value_match='equal'))
    assert cmp_len(2, shared_terminals(BernoulliNBStandardScaler, MultinomialNBRobustScaler, value_match='all'))

    assert cmp_len(0, shared_terminals(BernoulliNBStandardScaler, GaussianNB, value_match='all'))


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
    crossover(BernoulliNBStandardScaler, MultinomialNBRobustScaler)
    # Both parents and children should be unique
    assert len({ind.pipeline_str() for ind in [
        BernoulliNBStandardScaler, MultinomialNBRobustScaler, ind1_copy, ind2_copy]}) == 4
    assert ind1_copy.pipeline_str() != BernoulliNBStandardScaler.pipeline_str()
