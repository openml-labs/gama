"""
Contains mutation functions for genetic programming.
Each mutation function takes an individual and either returns a different individual, or None.
"""
import random
from typing import Callable

from .components import Individual, DATA_TERMINAL
from .operations import random_primitive_node


def mut_replace_terminal(individual: Individual, primitive_set: dict) -> None:
    terminals = list([(i, t) for i, t in enumerate(individual.terminals) if len(primitive_set[t.identifier]) > 1])
    if len(terminals) == 0:
        raise ValueError("Individual has no terminals or no terminals suitable for mutation.")

    terminal_index, old_terminal = random.choice(terminals)
    acceptable_new_terminals = [t for t in primitive_set[old_terminal.identifier] if t.value != old_terminal.value]
    new_terminal = random.choice(acceptable_new_terminals)
    individual.replace_terminal(terminal_index, new_terminal)


def mut_replace_primitive(individual: Individual, primitive_set: dict) -> None:
    replaceable_primitives = [(i, p) for i, p in enumerate(individual.primitives)
                              if len(primitive_set[p._primitive.output]) > 1]
    if replaceable_primitives == 0:
        raise ValueError("Individual has no primitives which can be replaced with a different primitive.")

    primitive_index, old_primitive_node = random.choice(replaceable_primitives)
    primitive_node = random_primitive_node(output_type=old_primitive_node._primitive.output,
                                           primitive_set=primitive_set,
                                           exclude=old_primitive_node._primitive)
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
    if len([t for t in individual.terminals if len(primitive_set[t.identifier]) > 1]):
        available_mutations.append(mut_replace_terminal)

    mut_fn = random.choice(available_mutations)
    mut_fn(individual, primitive_set)

    return mut_fn


def crossover(individual1: Individual, individual2: Individual) -> None:
    other_primitives = list(map(lambda primitive_node: primitive_node._primitive, individual2.primitives))
    shared_primitives = [p for p in individual1.primitives if p._primitive in other_primitives]
    both_at_least_length_2 = len(other_primitives) >= 2 and len(list(individual1.primitives)) >= 2

    crossover_choices = []
    if shared_primitives:
        crossover_choices.append(crossover_terminals)
    if both_at_least_length_2:
        crossover_choices.append(crossover_primitives)

    crossover_fn = random.choice(crossover_choices)
    #print(crossover_fn.__name__)
    crossover_fn(individual1, individual2)


def crossover_primitives(individual1: Individual, individual2: Individual) -> None:
    parent_node_1 = random.choice(list(individual1.primitives)[:-1])
    parent_node_2 = random.choice(list(individual2.primitives)[:-1])
    parent_node_1._data_node, parent_node_2._data_node = parent_node_2._data_node, parent_node_1._data_node


def crossover_terminals(individual1: Individual, individual2: Individual) -> None:
    shared_primitives = []
    for primitive_node in individual1.primitives:
        for primitive_node_2 in individual2.primitives:
            if primitive_node._primitive == primitive_node_2._primitive:
                shared_primitives.append((primitive_node, primitive_node_2))

    ind1_primitive, ind2_primitive = random.choice(shared_primitives)
    ind1_primitive.terminals = [(t1, t2)[int(random.random()*2)]
                                for (t1, t2) in zip(ind1_primitive._terminals, ind2_primitive._terminals)]
