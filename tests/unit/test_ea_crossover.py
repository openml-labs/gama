import pytest

from gama.genetic_programming.components import Individual
from gama.genetic_programming.mutation import crossover, crossover_primitives, crossover_terminals
from gama import GamaClassifier


@pytest.fixture
def pset():
    return GamaClassifier()._pset


@pytest.fixture
def BernoulliNBStandardScaler(pset):
    return Individual.from_string(
        "BernoulliNB(StandardScaler(data), alpha=0.1, fit_prior=True)",
        pset,
        None
    )


@pytest.fixture
def MultinomialNBRobustScaler(pset):
    return Individual.from_string(
        "MultinomialNB(RobustScaler(data), alpha=1.0, fit_prior=True)",
        pset,
        None
    )


def test_crossover_primitives(BernoulliNBStandardScaler, MultinomialNBRobustScaler):
    ind1_copy, ind2_copy = BernoulliNBStandardScaler.copy_as_new(), MultinomialNBRobustScaler.copy_as_new()
    # Cross-over is in-place
    crossover_primitives(BernoulliNBStandardScaler, MultinomialNBRobustScaler)
    # Both parents and children should be unique
    assert len({ind.pipeline_str() for ind in [
        BernoulliNBStandardScaler, MultinomialNBRobustScaler, ind1_copy, ind2_copy]}) == 4
    assert ind1_copy.pipeline_str() != BernoulliNBStandardScaler.pipeline_str()


def test_crossover_terminal(BernoulliNBStandardScaler, MultinomialNBRobustScaler):
    ind1_copy, ind2_copy = BernoulliNBStandardScaler.copy_as_new(), MultinomialNBRobustScaler.copy_as_new()
    # Cross-over is in-place
    crossover_terminals(BernoulliNBStandardScaler, MultinomialNBRobustScaler)
    # Both parents and children should be unique
    assert len({ind.pipeline_str() for ind in [
        BernoulliNBStandardScaler, MultinomialNBRobustScaler, ind1_copy, ind2_copy]}) == 4
    assert ind1_copy.pipeline_str() != BernoulliNBStandardScaler.pipeline_str()
