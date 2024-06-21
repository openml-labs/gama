"""
Contains mutation functions for genetic programming.
Each mutation function takes an individual and modifies it in-place.
"""
import logging
import random
from functools import partial
from typing import Callable, Optional, cast, List, Dict

import ConfigSpace as cs
import numpy as np

from gama.genetic_programming.components import PrimitiveNode, Terminal
from .components import Individual, DATA_TERMINAL
from .operations import random_primitive_node
from ..utilities.config_space import get_internal_output_types

# Avoid stopit from logging warnings every time a pipeline evaluation times out
logging.getLogger("stopit").setLevel(logging.ERROR)
log = logging.getLogger(__name__)


def mut_replace_terminal(
    individual: Individual, config_space: cs.ConfigurationSpace
) -> None:
    """Mutates an Individual in-place by replacing one of its Terminals.

    Parameters
    ----------
    individual: Individual
        Individual to mutate in-place.
    config_space: ConfigurationSpace
        The ConfigSpace object which defines the search space. Refer to the
        configuration/(classification||regression).py file for further details.
    """

    def terminal_replaceable(index_terminal):
        _, terminal = index_terminal
        return has_multiple_options(
            config_space.get_hyperparameter(terminal.config_space_name)
        )

    terminals = list(filter(terminal_replaceable, enumerate(individual.terminals)))
    if not terminals:
        raise ValueError("Individual has no terminals suitable for mutation.")

    terminal_index, old = random.choice(terminals)

    while True:
        new_terminal_value = config_space.get_hyperparameter(
            old.config_space_name
        ).sample(np.random.RandomState(42))
        if new_terminal_value != old.value:
            break

    new_terminal = Terminal(
        identifier=old.identifier,
        value=new_terminal_value,
        output=old.output,
        config_space_name=old.config_space_name,
    )
    individual.replace_terminal(terminal_index, new_terminal)


def mut_replace_primitive(
    individual: Individual, config_space: cs.ConfigurationSpace
) -> None:
    """Mutates an Individual in-place by replacing one of its Primitives.

    Parameters
    ----------
    individual: Individual
        Individual to mutate in-place.
    config_space: cs.ConfigurationSpace
        The ConfigSpace object which defines the search space. Refer to the
        configuration/(classification||regression).py file for further details.
    """

    def primitive_replaceable(index_primitive):
        _, primitive = index_primitive
        return has_multiple_options(
            config_space.get_hyperparameter(
                config_space.meta[primitive._primitive.output]
                if primitive._primitive.output in get_internal_output_types()
                else primitive._primitive.output
            )
        )

    primitives = list(filter(primitive_replaceable, enumerate(individual.primitives)))
    if not primitives:
        raise ValueError("Individual has no primitives suitable for replacement.")

    primitive_index, old_primitive_node = random.choice(primitives)
    primitive_node = random_primitive_node(
        output_type=old_primitive_node._primitive.output,
        config_space=config_space,
        exclude=old_primitive_node._primitive,
    )
    individual.replace_primitive(primitive_index, primitive_node)


def mut_shrink(
    individual: Individual,
    _config_space: Optional[cs.ConfigurationSpace] = None,
    shrink_by: Optional[int] = None,
) -> None:
    """Mutates an Individual in-place by removing any number of primitive nodes.

    Primitive nodes are removed from the preprocessing end.

    Parameters
    ----------
    individual: Individual
        Individual to mutate in-place.
    _config_space: dict, optional
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


def mut_insert(individual: Individual, config_space: cs.ConfigurationSpace) -> None:
    """Mutate an Individual in-place by inserting a PrimitiveNode at a random location.

    The new PrimitiveNode will not be inserted as root node.

    Parameters
    ----------
    individual: Individual
        Individual to mutate in-place.
    config_space: cs.ConfigurationSpace
        The ConfigSpace object which defines the search space. Refer to the
        configuration/(classification||regression).py file for further details.
    """
    parent_node = random.choice(list(individual.primitives))
    new_primitive_node = random_primitive_node(
        output_type=DATA_TERMINAL, config_space=config_space
    )
    new_primitive_node._data_node = parent_node._data_node
    parent_node._data_node = new_primitive_node


def has_multiple_options(hyperparameter: cs.hyperparameters.hyperparameter) -> bool:
    """Check if a ConfigSpace hyperparameter has more than one option.

    Only Constant, Float, Integer, and Categorical hyperparameters are currently
    supported. A TypeError is thrown if the hyperparameter is not of one of these
    types. Additionally, readers are directed to our Github Issues page to request
    support for other types.

    Parameters
    ----------
    hyperparameter: cs.hyperparameters.hyperparameter
        The hyperparameter to check.

    Returns
    -------
    bool
        True if the hyperparameter has more than one option, False otherwise.
    """
    if isinstance(
        hyperparameter,
        (
            cs.hyperparameters.FloatHyperparameter,
            cs.hyperparameters.IntegerHyperparameter,
        ),
    ):
        # For Float and Integer, check if the upper and lower bounds are different
        return hyperparameter.upper > hyperparameter.lower
    elif isinstance(hyperparameter, cs.CategoricalHyperparameter):
        # For Categorical, check if there are more than one unique items
        return len(set(hyperparameter.choices)) > 1
    elif isinstance(hyperparameter, cs.hyperparameters.Constant):
        # Constant has only one option
        return False
    else:
        # Default case, assuming no options or not a recognised type
        raise TypeError(f"Hyperparameter type {type(hyperparameter)} not supported")


def random_valid_mutation_in_place(
    individual: Individual,
    config_space: cs.ConfigurationSpace,
    max_length: Optional[int] = None,
) -> Callable:
    """Apply a random valid mutation in place.

    The random mutation can be one of:

     - mut_random_primitive
     - mut_random_terminal, if the individual has at least one
     - mutShrink, if individual has at least two primitives
     - mutInsert, if it would not exceed `new_max_length` when specified.

    Parameters
    ----------
    individual: Individual
        An individual to be mutated *in-place*.
    config_space: cs.ConfigurationSpace
        The ConfigSpace object which defines the search space. Refer to the
        configuration/(classification||regression).py file for further details.
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
            lambda p: has_multiple_options(
                config_space.get_hyperparameter(
                    config_space.meta[p._primitive.output]
                    if p._primitive.output in get_internal_output_types()
                    else p._primitive.output
                )
            ),
            individual.primitives,
        )

        if len(list(replaceable_primitives)) > 1:
            available_mutations.append(mut_replace_primitive)

        if max_length is None or n_primitives < max_length:
            available_mutations.append(mut_insert)
        if n_primitives > 1:
            available_mutations.append(mut_shrink)

        replaceable_terminals = filter(
            lambda t: has_multiple_options(
                config_space.get_hyperparameter(t.config_space_name)
            ),
            individual.terminals,
        )
        if len(list(replaceable_terminals)) > 1:
            available_mutations.append(mut_replace_terminal)

    if not available_mutations:
        log.warning(
            f"Individual {individual} has no valid mutations. "
            f"Returning original individual."
        )
        return lambda ind, config: ind

    mut_fn = random.choice(available_mutations)
    mut_fn(individual, config_space)
    return mut_fn
