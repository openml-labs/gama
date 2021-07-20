"""
Contains mutation functions for genetic programming.
Each mutation function takes an individual and modifies it in-place.
"""
import random
from functools import partial
from typing import Callable, Optional, cast, List, Dict

from gama.genetic_programming.components import PrimitiveNode
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

    def terminal_replaceable(index_terminal):
        _, terminal = index_terminal
        return len(primitive_set[terminal.identifier]) > 1

    terminals = list(filter(terminal_replaceable, enumerate(individual.terminals)))
    if len(terminals) == 0:
        raise ValueError("Individual has no terminals suitable for mutation.")

    terminal_index, old = random.choice(terminals)
    candidates = filter(lambda t: t.value != old.value, primitive_set[old.identifier])

    new_terminal = random.choice(list(candidates))
    individual.replace_terminal(terminal_index, new_terminal)


def mut_replace_primitive(individual: Individual, primitive_set: dict) -> None:
    """ Mutates an Individual in-place by replacing one of its Primitives.

    Parameters
    ----------
    individual: Individual
        Individual to mutate in-place.
    primitive_set: dict
    """

    def primitive_replaceable(index_primitive):
        _, primitive = index_primitive
        return len(primitive_set[primitive._primitive.output]) > 1

    primitives = list(filter(primitive_replaceable, enumerate(individual.primitives)))
    if len(primitives) == 0:
        raise ValueError("Individual has no primitives suitable for replacement.")

    primitive_index, old_primitive_node = random.choice(primitives)
    primitive_node = random_primitive_node(
        output_type=old_primitive_node._primitive.output,
        primitive_set=primitive_set,
        exclude=old_primitive_node._primitive,
    )
    individual.replace_primitive(primitive_index, primitive_node)


def mut_shrink(
    individual: Individual, primitive_set: dict = None, shrink_by: Optional[int] = None
) -> None:
    """ Mutates an Individual in-place by removing any number of primitive nodes.

    Primitive nodes are removed from the preprocessing end.

    Parameters
    ----------
    individual: Individual
        Individual to mutate in-place.
    primitive_set: dict, optional
        Not used. Present to create a matching function signature with other mutations.
    shrink_by: int, optional (default=None)
        Number of primitives to remove.
        Must be at least one greater than the number of primitives in `individual`.
        If None, a random number of primitives is removed.
    """
    n_primitives = len(list(individual.primitives))
    if shrink_by is not None and n_primitives <= shrink_by:
        raise ValueError(f"Can't shrink size {n_primitives} individual by {shrink_by}.")
    if shrink_by is None:
        shrink_by = random.randint(1, n_primitives - 1)

    current_primitive_node = individual.main_node
    primitives_left = n_primitives - 1
    while primitives_left > shrink_by:
        current_primitive_node = cast(PrimitiveNode, current_primitive_node._data_node)
        primitives_left -= 1
    current_primitive_node._data_node = DATA_TERMINAL


def mut_insert(individual: Individual, primitive_set: dict) -> None:
    """ Mutate an Individual in-place by inserting a PrimitiveNode at a random location.

    The new PrimitiveNode will not be inserted as root node.

    Parameters
    ----------
    individual: Individual
        Individual to mutate in-place.
    primitive_set: dict
    """
    parent_node = random.choice(list(individual.primitives))
    new_primitive_node = random_primitive_node(
        output_type=DATA_TERMINAL, primitive_set=primitive_set
    )
    new_primitive_node._data_node = parent_node._data_node
    parent_node._data_node = new_primitive_node


def random_valid_mutation_in_place(
    individual: Individual, primitive_set: dict, max_length: Optional[int] = None
) -> Callable:
    """ Apply a random valid mutation in place.

    The random mutation can be one of:

     - mut_random_primitive
     - mut_random_terminal, if the individual has at least one
     - mutShrink, if individual has at least two primitives
     - mutInsert, if it would not exceed `new_max_length` when specified.

    Parameters
    ----------
    individual: Individual
      An individual to be mutated *in-place*.
    primitive_set: dict
      A dictionary defining the set of primitives and terminals.
    max_length: int, optional (default=None)
     If specified, impose a maximum length on the new individual.


    Returns
    -------
    Callable
        The mutation function used.
    """
    n_primitives = len(list(individual.primitives))
    available_mutations: List[Callable[[Individual, Dict], None]] = []
    if max_length is not None and n_primitives > max_length:
        available_mutations.append(
            partial(mut_shrink, shrink_by=n_primitives - max_length)
        )
    else:
        replaceable_primitives = filter(
            lambda p: len(primitive_set[p._primitive.output]) > 1, individual.primitives
        )
        if len(list(replaceable_primitives)) > 1:
            available_mutations.append(mut_replace_primitive)

        if max_length is None or n_primitives < max_length:
            available_mutations.append(mut_insert)
        if n_primitives > 1:
            available_mutations.append(mut_shrink)

        replaceable_terminals = filter(
            lambda t: len(primitive_set[t.identifier]) > 1, individual.terminals
        )
        if len(list(replaceable_terminals)) > 1:
            available_mutations.append(mut_replace_terminal)

    mut_fn = random.choice(available_mutations)
    mut_fn(individual, primitive_set)

    return mut_fn
