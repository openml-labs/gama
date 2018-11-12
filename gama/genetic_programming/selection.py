import random

from deap import tools


def create_from_population2(operator_shell, pop, n, cxpb, mutpb):
    """ Creates n new individuals based on the population. Can apply both crossover and mutation. """
    offspring = []
    for _ in range(n):
        ind1, ind2 = random.sample(pop, k=2)
        ind1, ind2 = ind1.copy_as_new(), ind2.copy_as_new()
        if random.random() < cxpb and len(list(ind1.primitives))>1 and len(list(ind2.primitives))>1:
            ind1 = operator_shell.mate(ind1, ind2)
        else:
            ind1 = operator_shell.mutate(ind1)
        offspring.append(ind1)
    return offspring


def eliminate_NSGA(pop, n):
    return tools.selNSGA2(pop, k=len(pop))[-n:]
