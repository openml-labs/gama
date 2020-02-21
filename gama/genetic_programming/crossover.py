""" Functions which take two Individuals, and produce at least one new Individual from them. """
import random
from typing import List, Callable, Iterable, Optional, Tuple

from gama.genetic_programming.components import Individual


def random_crossover(individual1: Individual, individual2: Individual, max_length: Optional[int] = None
                     ) -> Tuple[Individual, Individual]:
    """ Perform a random valid crossover between two individuals in-place, if there is one.

    Parameters
    ----------
    individual1: Individual
        The individual to crossover with individual2.
    individual2: Individual
        The individual to crossover with individual1.
    max_length: int, optional(default=None)
        If set, the first individual in the returned tuple will have at most `max_length` primitives.
        Requires both provided individuals to contain at most `max_length` primitives.

    Raises
    ------
    ValueError
        - If there is no valid crossover function for the two individuals.
        - If `max_length` is set and either `individual1` or `individual2` contain more primitives than `max_length`.
    """
    if max_length is not None and len(individual1.primitives) > max_length:
        raise ValueError(f"`individual1` ({individual1}) exceeds `max_length` ({max_length}).")
    if max_length is not None and len(individual2.primitives) > max_length:
        raise ValueError(f"`individual2` ({individual1}) exceeds `max_length` ({max_length}).")

    crossover_choices = _valid_crossover_functions(individual1, individual2)
    if len(crossover_choices) == 0:
        raise ValueError(f"{individual1.pipeline_str()} and {individual2.pipeline_str()} can't mate.")
    ind1, ind2 = random.choice(crossover_choices)(individual1, individual2)

    if max_length is not None and len(ind1.primitives) > max_length:
        return ind2, ind1
    return ind1, ind2


def crossover_primitives(individual1: Individual, individual2: Individual) -> Tuple[Individual, Individual]:
    """ Crossover two individuals by splitting both at a random PrimitiveNode and switching one part out.

    Parameters
    ----------
    individual1: Individual
        The individual to crossover with individual2.
    individual2: Individual
        The individual to crossover with individual1.
    """
    parent_node_1 = random.choice(list(individual1.primitives)[:-1])
    parent_node_2 = random.choice(list(individual2.primitives)[:-1])
    parent_node_1._data_node, parent_node_2._data_node = parent_node_2._data_node, parent_node_1._data_node
    return individual1, individual2


def crossover_terminals(individual1: Individual, individual2: Individual) -> Tuple[Individual, Individual]:
    """ Crossover two individuals in-place by exchanging two Terminals with shared output type but different values.

    Parameters
    ----------
    individual1: Individual
        The individual to crossover with individual2.
    individual2: Individual
        The individual to crossover with individual1.
    """
    candidates = list(_shared_terminals(individual1, individual2, with_indices=True, value_match='different'))
    i, ind1_term, j, ind2_term = random.choice(candidates)
    individual1.replace_terminal(i, ind2_term)
    individual2.replace_terminal(j, ind1_term)
    return individual1, individual2


def _shared_terminals(individual1: Individual, individual2: Individual,
                      with_indices: bool = True, value_match: str = 'different') -> Iterable:
    """ Finds all shared Terminals between two Individuals.

    Parameters
    ----------
    individual1: Individual
    individual2: Individual
    with_indices: bool (default=True)
        If True, also return the indices of the Terminals w.r.t. the Individual.
    value_match: str (default='different')
        Indicates with matches to return, based on terminal values.
        Accepted values are:

         - 'different': only return shared terminals which have different values from each other
         - 'equal': only return shared terminals which have equal values from each other
         - 'all': return all shared terminals regardless of value

    Returns
    -------
    Sequence of Tuples with both Terminals, with the Terminal from Individual1 first.
        Tuple[Terminal, Terminal] if `with_indices` is False
        Tuple[int, Terminal, int, Terminal] if `with_indices` is True,
        each int specifies the index of the Terminal directly after.
    """
    if value_match not in ['different', 'equal', 'all']:
        raise ValueError(f"`value_match` must be one of 'all', 'equal' or 'different' but is '{value_match}'.")

    for i, ind1_term in enumerate(individual1.terminals):
        for j, ind2_term in enumerate(individual2.terminals):
            if ind1_term.identifier == ind2_term.identifier:
                if ((value_match == 'different' and ind1_term.value == ind2_term.value)
                        or (value_match == 'equal' and ind1_term.value != ind2_term.value)):
                    continue
                if with_indices:
                    yield (i, ind1_term, j, ind2_term)
                else:
                    yield (ind1_term, ind2_term)


def _valid_crossover_functions(individual1: Individual, individual2: Individual) -> List[Callable]:
    """ Return all crossover functions which can produce a new individual from the given two individuals.

    Parameters
    ----------
    individual1: Individual
        The individual to crossover with individual2.
    individual2: Individual
        The individual to crossover with individual1.

    Returns
    -------
    List[Callable]
        List of valid crossover functions given the content of individual1 and individual2.
    """
    crossover_choices = []
    if list(_shared_terminals(individual1, individual2)) != []:
        crossover_choices.append(crossover_terminals)
    if len(list(individual1.primitives)) >= 2 and len(list(individual2.primitives)) >= 2:
        crossover_choices.append(crossover_primitives)
    return crossover_choices
