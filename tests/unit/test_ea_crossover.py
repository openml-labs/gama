import pytest

from gama.genetic_programming.components import Individual
from gama.genetic_programming.mutation import crossover, crossover_primitives, crossover_terminals
from gama import GamaClassifier

@pytest.fixture
def pset():
    return GamaClassifier()._pset


def test_crossover_primitives(pset):
    ind1 = Individual.from_string(
        "GaussianNB(StandardScaler(data))",
        pset,
        None
    )
    ind2 = Individual.from_string(
        "MultinomialNB(RobustScaler(data), alpha=1.0, fit_prior=True)",
        pset,
        None
    )
    ind1_copy, ind2_copy = ind1.copy_as_new(), ind2.copy_as_new()
    # Cross-over is in-place
    crossover_primitives(ind1, ind2)
