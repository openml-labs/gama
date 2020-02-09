"""
Contains mutation functions for genetic programming.
Each mutation function takes an individual and modifies it in-place.
"""
import random
from typing import Callable

from .components import Individual, DATA_TERMINAL
from .operations import random_primitive_node


def mut_replace_terminal(individual: Individual, primitive_set: dict) -> None:
    """ Mutates an Individual in-place by replacing one of its Terminals.

    Parameters
    ----------
    individual: Individual
        Individual to mutate in-place.
    primitive_set: dict
    """
    terminals = list([(i, t) for i, t in enumerate(individual.terminals) if len(primitive_set[t.identifier]) > 1])
    if len(terminals) == 0:
        raise ValueError("Individual has no terminals or no terminals suitable for mutation.")

    terminal_index, old_terminal = random.choice(terminals)
    acceptable_new_terminals = [t for t in primitive_set[old_terminal.identifier] if t.value != old_terminal.value]
    new_terminal = random.choice(acceptable_new_terminals)
    individual.replace_terminal(terminal_index, new_terminal)


def mut_replace_primitive(individual: Individual, primitive_set: dict) -> None:
    """ Mutates an Individual in-place by replacing one of its Primitives.

    Parameters
    ----------
    individual: Individual
        Individual to mutate in-place.
    primitive_set: dict
    """
    replaceable_primitives = [(i, p) for i, p in enumerate(individual.primitives)
                              if len(primitive_set[p._primitive.output]) > 1]
    if replaceable_primitives == 0:
        raise ValueError("Individual has no primitives which can be replaced with a different primitive.")

    primitive_index, old_primitive_node = random.choice(replaceable_primitives)
    primitive_node = random_primitive_node(output_type=old_primitive_node._primitive.output,
                                           primitive_set=primitive_set,
                                           exclude=old_primitive_node._primitive)
    individual.replace_primitive(primitive_index, primitive_node)


def mut_shrink(individual: Individual, primitive_set: dict = None) -> None:
    """ Mutates an Individual in-place by removing any number of primitive nodes, starting with the last.

    Parameters
    ----------
    individual: Individual
        Individual to mutate in-place.
    primitive_set: dict, optional
        Not used. Present to create a matching function signature with other mutations.
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
    """ Mutate an Individual in-place by inserting a PrimitiveNode at a random location, except as root node.

    Parameters
    ----------
    individual: Individual
        Individual to mutate in-place.
    primitive_set: dict
    """
    parent_node = random.choice(list(individual.primitives))
    new_primitive_node = random_primitive_node(output_type=DATA_TERMINAL, primitive_set=primitive_set)
    new_primitive_node._data_node = parent_node._data_node
    parent_node._data_node = new_primitive_node


def random_valid_mutation_in_place(individual: Individual, primitive_set: dict) -> Callable:
    """ Apply a random valid mutation in place.

    The random mutation can be one of:

     - mut_random_primitive
     - mut_random_terminal, if the individual has at least one
     - mutShrink, if individual has at least two primitives
     - mutInsert

    Parameters
    ----------
    individual: Individual
      An individual to be mutated *in-place*.
    primitive_set: dict
      A dictionary defining the set of primitives and terminals.

    Returns
    -------
    Callable
        The mutation function used.
    """
    available_mutations = [mut_replace_primitive, mut_insert]
    if len(list(individual.primitives)) > 1:
        available_mutations.append(mut_shrink)
    if len([t for t in individual.terminals if len(primitive_set[t.identifier]) > 1]):
        available_mutations.append(mut_replace_terminal)

    mut_fn = random.choice(available_mutations)
    mut_fn(individual, primitive_set)

    return mut_fn
