import functools
import logging
import uuid

from deap import tools
import numpy as np

from ..utilities.logging_utilities import TOKENS, log_parseable_event
from .mutation import random_valid_mutation
from .modified_deap import cxOnePoint

log = logging.getLogger(__name__)
created_individuals = {}


def is_new(item):
    """ Check whether this individual (genotype) has been seen before. If not, store it as seen. """
    _is_new = str(item) not in created_individuals
    if _is_new:
        created_individuals[str(item)] = item
    return _is_new


def try_until_new(func):
    def fn_new(*args, **kwargs):
        max_tries = 50
        ind_is_new = is_new
        for _ in range(max_tries):
            new_ind, log_args = func(*args, **kwargs)
            if ind_is_new(new_ind):
                return new_ind, log_args
        log.warning("Could not create a new individual from 50 iterations of {}".format(func.__name__))
        return new_ind, log_args
    return fn_new


@try_until_new
def generate_new_with_id(container, generator):
    individual = tools.initIterate(container, generator)
    individual.id = uuid.uuid4()
    return individual, []


def generate_new(*args, **kwargs):
    return generate_new_with_id(*args, **kwargs)[0]


@try_until_new
def random_valid_mutation_new(ind, toolbox, pset):
    new_ind = toolbox.clone(ind)
    (new_ind,), mut_fn = random_valid_mutation(new_ind, pset, return_function=True)
    new_ind.id = uuid.uuid4()
    log_args = [TOKENS.MUTATION, new_ind.id, ind.id, mut_fn.__name__]
    return new_ind, log_args


@try_until_new
def mate_new(ind1, ind2):
    parent1_id, parent2_id = ind1.id, ind2.id
    new_ind, _ = cxOnePoint(ind1, ind2)
    new_ind.id = uuid.uuid4()
    log_args = [TOKENS.CROSSOVER, new_ind.id, parent1_id, parent2_id]
    return new_ind, log_args


def create_from_population(pop, n, cxpb, mutpb, toolbox):
    """ Creates n new individuals based on the population. Can apply both crossover and mutation. """
    offspring = []
    for _ in range(n):
        ind1, ind2 = np.random.choice(range(len(pop)), size=2, replace=False)
        ind1, ind2 = toolbox.clone(pop[ind1]), toolbox.clone(pop[ind2])
        if np.random.random() < cxpb:
            new_ind, log_args = toolbox.mate(ind1, ind2)
            log_parseable_event(log, *log_args)
        else:
            new_ind, log_args = toolbox.mutate(ind1, toolbox)
            log_parseable_event(log, *log_args)
        offspring.append(new_ind)
    return offspring
