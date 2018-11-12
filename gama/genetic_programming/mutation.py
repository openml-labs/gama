"""
Contains mutation functions for genetic programming.
Each mutation function takes an individual and either returns a different individual, or None.
"""
import random
from typing import Callable

from .components import Individual, DATA_TERMINAL, random_primitive_node


def mut_replace_terminal(individual: Individual, primitive_set: dict) -> None:
    terminals = list([(i, t) for i, t in enumerate(individual.terminals) if len(primitive_set[t._identifier]) > 1])
    if len(terminals) == 0:
        raise ValueError("Individual has no terminals or no terminals suitable for mutation.")

    terminal_index, old_terminal = random.choice(terminals)
    acceptable_new_terminals = [t for t in primitive_set[old_terminal._identifier] if t.value != old_terminal.value]
    new_terminal = random.choice(acceptable_new_terminals)
    individual.replace_terminal(terminal_index, new_terminal)


def mut_replace_primitive(individual: Individual, primitive_set: dict) -> None:
    replaceable_primitives = [(i, p) for i, p in enumerate(individual.primitives)
                              if len(primitive_set[p._primitive.output]) > 1]
    if replaceable_primitives == 0:
        raise ValueError("Individual has no primitives which can be replaced with a different primitive.")

    primitive_index, old_primitive_node = random.choice(replaceable_primitives)
    primitive_node = random_primitive_node(output_type=old_primitive_node._primitive.output,
                                           primitive_set=primitive_set)
    individual.replace_primitive(primitive_index, primitive_node)


def mut_shrink(individual: Individual, primitive_set=None) -> None:
    """ Shrinks the individual by removing any number of nodes, starting with the last primitive node.

    primitive_set parameter is not used and only to create a matching function signature with other mutations.
    """
    n_primitives = len(list(individual.primitives))
    if n_primitives == 1:
        raise ValueError("Can not shrink an individual with only one primitive.")

    n_primitives_to_cut = random.randint(1, n_primitives-1)
    current_primitive_node = individual.main_node
    primitives_left = n_primitives - 1
    while primitives_left > n_primitives_to_cut:
        current_primitive_node = current_primitive_node._data_node
        primitives_left -= 1
    current_primitive_node._data_node = DATA_TERMINAL


def mut_insert(individual: Individual, primitive_set: dict) -> None:
    """ Adds a PrimitiveNode at a random location, except as root node. """
    if len(list(individual.primitives)) == 1:
        parent_node = individual.main_node
    else:
        parent_node = random.choice(list(individual.primitives)[:-1])
    new_primitive_node = random_primitive_node(output_type=DATA_TERMINAL, primitive_set=primitive_set)
    new_primitive_node._data_node = parent_node._data_node
    parent_node._data_node = new_primitive_node


def random_valid_mutation_in_place(individual: Individual, primitive_set: dict) -> Callable:
    """ Apply a random valid mutation in place.

    :params individual: Individual.
      An individual to be mutated *in place*
    :params primitive_set: dict.
      A dictionary defining the set of primitives and terminals.

    :returns: the mutation function used

    The choices are `mut_random_primitive`, `mut_random_terminal`,
    `mutShrink` and `mutInsert`.
    A pipeline can not shrink a primitive if it only has one.
    A terminal can not be replaced if it there is none.
    """
    available_mutations = [mut_replace_primitive, mut_insert]
    if len(list(individual.primitives)) > 1:
        available_mutations.append(mut_shrink)
    if len([t for t in individual.terminals if len(primitive_set[t.output]) > 1]):
        available_mutations.append(mut_replace_terminal)

    mut_fn = random.choice(available_mutations)
    mut_fn(individual, primitive_set)

    return mut_fn


def crossover(individual1: Individual, individual2: Individual) -> None:
    parent_node_1 = random.choice(list(individual1.primitives)[:-1])
    parent_node_2 = random.choice(list(individual2.primitives)[:-1])
    parent_node_1._data_node, parent_node_2._data_node = parent_node_2._data_node, parent_node_1._data_node
