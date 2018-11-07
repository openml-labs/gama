"""
Contains mutation functions for genetic programming.
Each mutation function takes an individual and either returns a different individual, or None.
"""
import random

from deap import gp, creator
import numpy as np

from .components import Individual


def mut_replace_terminal(individual: Individual, primitive_set: dict):
    terminals = list(individual.terminals)
    if len(terminals) == 0:
        raise ValueError("Individual has no terminals.")

    terminal_index, old_terminal = random.sample(terminals, k=1)[0]
    acceptable_new_terminals = [t for t in primitive_set[old_terminal._identifier] if t.value != old_terminal.value]
    new_terminal = random.sample(acceptable_new_terminals, k=1)[0]
    individual.replace_terminal(terminal_index, new_terminal)


def mut_replace_primitive(individual: Individual, primitive_set: dict):
    replaceable_primitives = [p for p in list(individual.primitives) if len(primitive_set[p._primitive.output]) > 1]
    if replaceable_primitives == 0:
        raise ValueError("Individual has no primitives which can be replaced with a different primitive.")


def mut_replace_primitive(ind, pset):
    """ Mutation function which replaces a primitive (and corresponding terminals). """
    # DEAP.gp's mutNodeReplacement does not work since it will only replace primitives
    # if they have the same input arguments (which is not true in this context)

    eligible = [i for i, el in enumerate(ind) if
                (issubclass(type(el), gp.Primitive) and len(pset.primitives[el.ret]) > 1)]
    if eligible == []:
        raise ValueError('Individual could not be mutated because no valid primitive was available: {}'.format(ind))

    to_change = np.random.choice(eligible)
    number_of_removed_terminals = len(ind[to_change].args) - 1

    # Determine new primitive and terminals that need to be added.
    alternatives = [prim for prim in pset.primitives[ind[to_change].ret] if prim.name != ind[to_change].name]
    new_primitive = np.random.choice(alternatives)
    new_terminals = [np.random.choice(pset.terminals[ret_type]) for ret_type in new_primitive.args[1:]]

    # Determine which terminals should also be removed.
    # We want to find the first unmatched terminal, but can ignore the data
    # input terminal, as that is a subtree we do not wish to replace.
    terminal_index = find_unmatched_terminal(ind[to_change + 1:])
    if terminal_index is False:
        if number_of_removed_terminals == 0:
            # No terminals need to be removed and everything after the primitive is a perfect (data) subtree.
            new_expr = ind[:] + new_terminals
            new_expr[to_change] = new_primitive
            return creator.Individual(new_expr),
        else:
            raise Exception(
                "Found no unmatched terminals after removing a primitive which had terminals: {}".format(str(ind)))
    else:
        # Adjust for the fact the searched individual had part removed.
        # (Since the unmatched terminal was created through removing primitives
        # before it, this means the adjustment is always necessary)
        terminal_index += (to_change + 1)
        # In the case the unmatched terminal was the Data terminal, we actually
        # would like to start adding terminals only after this position.
        # This way there is no need to make a distinction later on whether a
        # primitive's data-terminal is a leaf or a subtree.
        if ind[terminal_index].value in pset.arguments:
            terminal_index += 1

        # 3. Construct the new individual
        # Replacing terminals can not be done in-place, as the number of terminals can vary.
        new_expr = ind[:terminal_index] + new_terminals + ind[terminal_index + number_of_removed_terminals:]
        # Replacing the primitive can be done in-place.
        new_expr[to_change] = new_primitive
        # expr = ind[:to_change] + [new_primitive] + ind[to_change+1:terminal_index] + new_terminals + ind[terminal_index+number_of_removed_terminals:]
        ind = creator.Individual(new_expr)

        return ind,


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
