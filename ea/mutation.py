from deap import gp, creator
import numpy as np


def find_unmatched_terminal(individual):
    """ Finds the location of the first terminal that can not be matched with a primitive.

    Raises a `ValueError` if no terminals are found.
    """
    unmatched_args = []
    for i, el in enumerate(individual):
        if len(unmatched_args) > 0 and el.ret == unmatched_args[0]:
            unmatched_args.pop(0)
        elif issubclass(type(el), gp.Terminal):
            return i
        if issubclass(type(el), gp.Primitive):
            # Replace with list-inserts if performance is bad.
            unmatched_args = el.args + unmatched_args

    return False


def mut_replace_terminal(ind, pset):
    """ Mutation function which replaces a terminal."""

    eligible = [i for i, el in enumerate(ind) if
                (issubclass(type(el), gp.Terminal) and len(pset.terminals[el.ret]) > 1)]
    # els = [el for i,el in enumerate(ind) if (issubclass(type(el), gp.Terminal) and len(pset.terminals[el.ret])>1)]
    if eligible == []:
        # print('No way to mutate '+str(ind)+' was found.')
        return ind,

    to_change = np.random.choice(eligible)
    alternatives = [t for t in pset.terminals[ind[to_change].ret] if t != ind[to_change]]
    ind[to_change] = np.random.choice(alternatives)
    return ind,


def mut_replace_primitive(ind, pset):
    """ Mutation function which replaces a primitive (and corresponding terminals). """
    # DEAP.gp's mutNodeReplacement does not work since it will only replace primitives
    # if they have the same input arguments (which is not true in this context)

    eligible = [i for i, el in enumerate(ind) if
                (issubclass(type(el), gp.Primitive) and len(pset.primitives[el.ret]) > 1)]
    if eligible == []:
        return ind,

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
