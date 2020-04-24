""" Selection operators. """
import random
from typing import List

from gama.genetic_programming.operator_set import OperatorSet
from gama.genetic_programming.components import Individual
from gama.genetic_programming.nsga2 import nsga2_select
from gama.utilities.generic.paretofront import ParetoFront
from gama.genetic_programming.crossover import _valid_crossover_functions


def create_from_population(
    operator_shell: OperatorSet,
    pop: List[Individual],
    n: int,
    cxpb: float,
    mutpb: float,
) -> List[Individual]:
    """ Creates n new individuals based on the population. """
    offspring = []
    metrics = [lambda ind: ind.fitness.values[0], lambda ind: ind.fitness.values[1]]
    parent_pairs = nsga2_select(pop, n, metrics)
    for (ind1, ind2) in parent_pairs:
        if random.random() < cxpb and len(_valid_crossover_functions(ind1, ind2)) > 0:
            ind1 = operator_shell.mate(ind1, ind2)
        else:
            ind1 = operator_shell.mutate(ind1)
        offspring.append(ind1)
    return offspring


def eliminate_from_pareto(pop: List[Individual], n: int) -> List[Individual]:
    # For now we only eliminate one at a time so this will do.
    if n != 1:
        raise NotImplementedError("Currently only n=1 is supported.")

    def inverse_fitness(ind):
        return [-value for value in ind.fitness.values]

    pareto_worst = ParetoFront(pop, inverse_fitness)
    return [random.choice(pareto_worst)]
