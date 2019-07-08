"""
Contains mutation functions for genetic programming.
Each mutation function takes an individual and either returns a different individual, or None.
This file duplicates the functions in mutation.py, but each function checks before a given operation will
obey a grammar for individuals.
"""
import random
from typing import Callable

from .components import Individual, DATA_TERMINAL
from .operations import random_primitive_node

GRAMMAR_PATIENCE = 10


def _matches_rule(individual, rule):
    return rule.matches(individual.primitives)


def mut_replace_terminal(individual: Individual, primitive_set: dict, rule) -> None:
    terminals = list([(i, t) for i, t in enumerate(individual.terminals) if len(primitive_set[t.identifier]) > 1])
    if len(terminals) == 0:
        raise ValueError("Individual has no terminals or no terminals suitable for mutation.")

    for _ in range(GRAMMAR_PATIENCE):
        terminal_index, old_terminal = random.choice(terminals)
        acceptable_new_terminals = [t for t in primitive_set[old_terminal.identifier] if t.value != old_terminal.value]
        new_terminal = random.choice(acceptable_new_terminals)
        copy = individual.copy_as_new()
        copy.replace_terminal(terminal_index, new_terminal)
        if _matches_rule(copy, rule):
            individual.replace_terminal(terminal_index, new_terminal)
            break


def mut_replace_primitive(individual: Individual, primitive_set: dict, rule) -> None:
    replaceable_primitives = [(i, p) for i, p in enumerate(individual.primitives)
                              if len(primitive_set[p._primitive.output]) > 1]
    if replaceable_primitives == 0:
        raise ValueError("Individual has no primitives which can be replaced with a different primitive.")

    for _ in range(GRAMMAR_PATIENCE):
        primitive_index, old_primitive_node = random.choice(replaceable_primitives)
        primitive_node = random_primitive_node(output_type=old_primitive_node._primitive.output,
                                               primitive_set=primitive_set,
                                               exclude=old_primitive_node._primitive)
        copy = individual.copy_as_new()
        copy.replace_primitive(primitive_index, primitive_node)
        if _matches_rule(copy, rule):
            individual.replace_primitive(primitive_index, primitive_node)
            break


def mut_shrink(individual: Individual, primitive_set, rule) -> None:
    """ Shrinks the individual by removing any number of nodes, starting with the last primitive node.

    primitive_set parameter is not used and only to create a matching function signature with other mutations.
    """
    n_primitives = len(list(individual.primitives))
    if n_primitives == 1:
        raise ValueError("Can not shrink an individual with only one primitive.")

    def cut_primitives(individual, n_primitives_to_cut):
        current_primitive_node = individual.main_node
        primitives_left = n_primitives - 1
        while primitives_left > n_primitives_to_cut:
            current_primitive_node = current_primitive_node._data_node
            primitives_left -= 1
        current_primitive_node._data_node = DATA_TERMINAL

    tried_n = set()

    for _ in range(GRAMMAR_PATIENCE):
        n_primitives_to_cut = random.randint(1, n_primitives - 1)
        if n_primitives_to_cut in tried_n:
            continue
        tried_n.add(n_primitives_to_cut)
        copy = individual.copy_as_new()
        cut_primitives(copy, n_primitives_to_cut)
        if _matches_rule(copy, rule):
            cut_primitives(individual, n_primitives_to_cut)
            break


def mut_insert(individual: Individual, primitive_set: dict, rule) -> None:
    """ Adds a PrimitiveNode at a random location, except as root node. """

    for _ in range(GRAMMAR_PATIENCE):
        new_primitive_node = random_primitive_node(output_type=DATA_TERMINAL, primitive_set=primitive_set)
        copy = individual.copy_as_new()
        if len(list(individual.primitives)) == 1:
            parent_node = copy.main_node
        else:
            insert_index = random.randint(0, len(list(individual.primitives)[:-1]))
            parent_node = list(copy.primitives)[insert_index]
        new_primitive_node._data_node = parent_node._data_node
        parent_node._data_node = new_primitive_node
        if _matches_rule(copy, rule):
            if len(list(individual.primitives)) == 1:
                parent_node = individual.main_node
            else:
                parent_node = list(individual.primitives)[insert_index]
            new_primitive_node._data_node = parent_node._data_node
            parent_node._data_node = new_primitive_node
            break


def random_valid_mutation_in_place(individual: Individual, primitive_set: dict, rule) -> Callable:
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
    mut_fn(individual, primitive_set, rule)

    return mut_fn


def crossover(individual1: Individual, individual2: Individual, rule) -> None:
    other_primitives = list(map(lambda primitive_node: primitive_node._primitive, individual2.primitives))
    shared_primitives = [p for p in individual1.primitives if p._primitive in other_primitives]
    both_at_least_length_2 = len(other_primitives) >= 2 and len(list(individual1.primitives)) >= 2

    crossover_choices = []
    if shared_primitives:
        crossover_choices.append(crossover_terminals)
    if both_at_least_length_2:
        crossover_choices.append(crossover_primitives)

    random.choice(crossover_choices)(individual1, individual2, rule)


def crossover_primitives(individual1: Individual, individual2: Individual, rule) -> None:

    def do_crossover(individual1, individual2, node_index_1=None, node_index_2=None):
        if node_index_1 is None:
            node_index_1 = random.randint(0, len(individual1.primitives) - 1)
        if node_index_2 is None:
            node_index_2 = random.randint(0, len(individual2.primitives) - 1)
        parent_node_1 = list(individual1.primitives)[node_index_1]
        parent_node_2 = list(individual2.primitives)[node_index_2]
        parent_node_1._data_node, parent_node_2._data_node = parent_node_2._data_node, parent_node_1._data_node
        return node_index_1, node_index_2

    for _ in range(GRAMMAR_PATIENCE):
        copy1 = individual1.copy_as_new()
        copy2 = individual2.copy_as_new()
        ni1, ni2 = do_crossover(copy1, copy2)
        if _matches_rule(copy1, rule) and _matches_rule(copy2, rule):
            do_crossover(individual1, individual2, ni1, ni2)


def crossover_terminals(individual1: Individual, individual2: Individual, rule) -> None:

    def do_crossover(individual1, individual2, spec=None):
        if spec is None:
            shared_primitives = []
            for index1, primitive_node in enumerate(individual1.primitives):
                for index2, primitive_node_2 in enumerate(individual2.primitives):
                    if primitive_node._primitive == primitive_node_2._primitive:
                        shared_primitives.append((index1, index2))
            index1, index2 = random.choice(shared_primitives)
        else:
            index1 = spec[0]
            index2 = spec[1]

        ind1_primitive = individual1.primitives[index1]
        ind2_primitive = individual2.primitives[index2]

        if spec is None:
            which_to_use = [int(random.random()*2) for _ in ind1_primitive._terminals]
        else:
            which_to_use = spec[2]

        ind1_primitive._terminals = [ item[ item[2] ]
                                      for item in zip(ind1_primitive._terminals, ind2_primitive._terminals, which_to_use) ]

        return index1, index2, which_to_use

    for _ in range(GRAMMAR_PATIENCE):
         copy1 = individual1.copy_as_new()
         copy2 = individual2.copy_as_new()
         spec = do_crossover(copy1, copy2)
         if _matches_rule(copy1, rule) and _matches_rule(copy2, rule):
             do_crossover(individual1, individual2, spec)
             break



