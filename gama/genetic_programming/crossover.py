""" Functions which take two Individuals and produce at least one new Individual. """
import random
from typing import List, Callable, Iterable, Optional, Tuple

from gama.genetic_programming.components import Individual


def random_crossover(
    ind1: Individual, ind2: Individual, max_length: Optional[int] = None
) -> Tuple[Individual, Individual]:
    """ Random valid crossover between two individuals in-place, if it can be done.

    Parameters
    ----------
    ind1: Individual
        The individual to crossover with ind2.
    ind2: Individual
        The individual to crossover with ind1.
    max_length: int, optional(default=None)
        The first individual in the returned tuple has at most `max_length` primitives.
        Requires both provided individuals to contain at most `max_length` primitives.

    Raises
    ------
    ValueError
        - If there is no valid crossover function for the two individuals.
        - If `max_length` is set and either `ind1` or `ind2` contain
          more primitives than `max_length`.
    """
    if max_length is not None and len(ind1.primitives) > max_length:
        raise ValueError(f"`individual1` ({ind1}) exceeds `max_length` ({max_length}).")
    if max_length is not None and len(ind2.primitives) > max_length:
        raise ValueError(f"`individual2` ({ind2}) exceeds `max_length` ({max_length}).")

    crossover_choices = _valid_crossover_functions(ind1, ind2)
    if len(crossover_choices) == 0:
        raise ValueError(f"{ind1.pipeline_str()} and {ind2.pipeline_str()} can't mate.")
    ind1, ind2 = random.choice(crossover_choices)(ind1, ind2)

    if max_length is not None and len(ind1.primitives) > max_length:
        return ind2, ind1
    return ind1, ind2


def crossover_primitives(
    ind1: Individual, ind2: Individual
) -> Tuple[Individual, Individual]:
    """ Crossover two individuals by exchanging any number of preprocessing steps.

    Parameters
    ----------
    ind1: Individual
        The individual to crossover with individual2.
    ind2: Individual
        The individual to crossover with individual1.
    """
    p1_node = random.choice(list(ind1.primitives)[:-1])
    p2_node = random.choice(list(ind2.primitives)[:-1])
    p1_node._data_node, p2_node._data_node = p2_node._data_node, p1_node._data_node
    return ind1, ind2


def crossover_terminals(
    ind1: Individual, ind2: Individual
) -> Tuple[Individual, Individual]:
    """ Crossover two individuals in-place by exchanging two Terminals.

    Terminals must share output type but have different values.

    Parameters
    ----------
    ind1: Individual
        The individual to crossover with individual2.
    ind2: Individual
        The individual to crossover with individual1.
    """
    options = _shared_terminals(ind1, ind2, with_indices=True, value_match="different")
    i, ind1_term, j, ind2_term = random.choice(list(options))
    ind1.replace_terminal(i, ind2_term)
    ind2.replace_terminal(j, ind1_term)
    return ind1, ind2


def _shared_terminals(
    ind1: Individual,
    ind2: Individual,
    with_indices: bool = True,
    value_match: str = "different",
) -> Iterable:
    """ Finds all shared Terminals between two Individuals.

    Parameters
    ----------
    ind1: Individual
    ind2: Individual
    with_indices: bool (default=True)
        If True, also return the indices of the Terminals w.r.t. the Individual.
    value_match: str (default='different')
        Indicates with matches to return, based on terminal values.
        Accepted values are:

         - 'different': only return terminals with different values from each other
         - 'equal': only return terminals with equal values
         - 'all': return all shared terminals regardless of value

    Returns
    -------
    Sequence of Tuples with both Terminals, with the Terminal from ind1 first.
        Tuple[Terminal, Terminal] if `with_indices` is False
        Tuple[int, Terminal, int, Terminal] if `with_indices` is True,
        each int specifies the index of the Terminal directly after.
    """
    if value_match not in ["different", "equal", "all"]:
        raise ValueError(f"`value_match` ('{value_match}') is not valid.")

    for i, ind1_term in enumerate(ind1.terminals):
        for j, ind2_term in enumerate(ind2.terminals):
            if ind1_term.identifier == ind2_term.identifier:
                if value_match == "different" and ind1_term.value == ind2_term.value:
                    continue
                if value_match == "equal" and ind1_term.value != ind2_term.value:
                    continue
                if with_indices:
                    yield (i, ind1_term, j, ind2_term)
                else:
                    yield (ind1_term, ind2_term)


def _valid_crossover_functions(ind1: Individual, ind2: Individual) -> List[Callable]:
    """ Find all crossover functions that can produce new individuals from this input.

    Parameters
    ----------
    ind1: Individual
        The individual to crossover with individual2.
    ind2: Individual
        The individual to crossover with individual1.

    Returns
    -------
    List[Callable]
        List of valid crossover functions given the content of ind1 and ind2.
    """
    crossover_choices = []
    if list(_shared_terminals(ind1, ind2)):
        crossover_choices.append(crossover_terminals)
    if len(list(ind1.primitives)) >= 2 and len(list(ind2.primitives)) >= 2:
        crossover_choices.append(crossover_primitives)
    return crossover_choices
