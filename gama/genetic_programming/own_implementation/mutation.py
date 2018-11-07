"""
Contains mutation functions for genetic programming.
Each mutation function takes an individual and either returns a different individual, or None.
"""
import random

from deap import gp, creator
import numpy as np

from .components import Individual, DATA_TERMINAL, random_terminals_for_primitive, PrimitiveNode


def mut_replace_terminal(individual: Individual, primitive_set: dict):
    terminals = list(enumerate(individual.terminals))
    if len(terminals) == 0:
        raise ValueError("Individual has no terminals.")

    terminal_index, old_terminal = random.sample(terminals, k=1)[0]
    acceptable_new_terminals = [t for t in primitive_set[old_terminal._identifier] if t.value != old_terminal.value]
    new_terminal = random.sample(acceptable_new_terminals, k=1)[0]
    individual.replace_terminal(terminal_index, new_terminal)


def mut_replace_primitive(individual: Individual, primitive_set: dict):
    replaceable_primitives = [(i, p) for i, p in enumerate(individual.primitives)
                              if len(primitive_set[p._primitive.output]) > 1]
    if replaceable_primitives == 0:
        raise ValueError("Individual has no primitives which can be replaced with a different primitive.")

    primitive_index, old_primitive_node = random.sample(replaceable_primitives, k=1)[0]
    new_primitive = random.sample(primitive_set[old_primitive_node._primitive.output], k=1)[0]
    primitive_node = PrimitiveNode(new_primitive,
                                   data_node=None,  # To be replaced by `replace_primitive`.
                                   terminals=random_terminals_for_primitive(primitive_set, new_primitive))
    individual.replace_primitive(primitive_index, primitive_node)


def random_valid_mutation(ind, pset, return_function=False):
    """ Picks a mutation uniform at random from options which are possible.

    :params ind: an individual to be mutated *in place*
    :params pset: a DEAP primitive set
    :params return_function: bool, optional.
        If True, also return the function object which was applied.

    :returns: the mutated individual and optionally the mutation function

    The choices are `mut_random_primitive`, `mut_random_terminal`,
    `mutShrink` and `mutInsert`.
    A pipeline can not shrink a primitive if it only has one.
    A terminal can not be replaced if it there is none.
    """
    available_mutations = [mut_replace_primitive, gp.mutInsert]
    if len([el for el in ind if issubclass(type(el), gp.Primitive)]) > 1:
        available_mutations.append(gp.mutShrink)
    # The data-input terminal is always guaranteed, but can not be mutated.
    if len(find_replaceable_terminals(ind, pset)) > 0:
        available_mutations.append(mut_replace_terminal)

    mut_fn = np.random.choice(available_mutations)
    if gp.mutShrink == mut_fn:
        # only mutShrink function does not need pset.
        new_ind, = mut_fn(ind)
    else:
        new_ind, = mut_fn(ind, pset)

    if return_function:
        return (new_ind, ), mut_fn
    else:
        return new_ind,
