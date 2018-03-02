""" This file contains function(s) that are in DEAP, but need slight
modifications in order to work for automatic pipeline construction.
"""

import inspect
import sys

import numpy as np

def gen_grow_safe(pset, min_, max_, type_=None):
    """Generate an expression where each leaf might have a different depth between min_ and max_.
    Condition specifies if desired return type is not Data or Predictions, that a terminal be used.
    """

    def condition(height, depth, type_):
        """Stop when the depth is equal to height or when a node should be a terminal."""
        #return (type_ not in [Data, Predictions]) or (depth == height)
        return (len(pset.primitives[type_]) == 0) or (depth == height)

    return generate(pset, min_, max_, condition, type_)

def generate(pset, min_, max_, condition, type_=None):
    """Generate a Tree as a list of lists.
    The tree is build from the root to the leaves, and it stop growing when
    the condition is fulfilled.
    Parameters
    ----------
    pset: PrimitiveSetTyped
        Primitive set from which primitives are selected.
    min_: int
        Minimum height of the produced trees.
    max_: int
        Maximum Height of the produced trees.
    condition: function
        The condition is a function that takes two arguments,
        the height of the tree to build and the current
        depth in the tree.
    type_: class
        The type that should return the tree when called, when
        :obj:None (default) no return type is enforced.
    Returns
    -------
    individual: list
        A grown tree with leaves at possibly different depths
        dependending on the condition function.
    """
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = np.random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()

        # We've added a type_ parameter to the condition function
        if condition(height, depth, type_):
            try:
                term = np.random.choice(pset.terminals[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError(
                    'The gp.generate function tried to add '
                    'a terminal of type {}, but there is'
                    'none available. {}'.format(type_, traceback)
                )
            if inspect.isclass(term):
                term = term()
            expr.append(term)
        else:
            try:
                prim = np.random.choice(pset.primitives[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError(
                    'The gp.generate function tried to add '
                    'a primitive of type {}, but there is'
                    'none available. {}'.format(type_, traceback)
                )
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr

from collections import defaultdict
import random
__type__ = object
def cxOnePoint(ind1, ind2):
    """Randomly select in each individual and exchange each subtree with the
    point as root between each individual.
    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :returns: A tuple of two trees.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2

    # List all available primitive types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)
    if ind1.root.ret == __type__:
        # Not STGP optimization
        types1[__type__] = range(1, len(ind1))
        types2[__type__] = range(1, len(ind2))
        common_types = [__type__]
    else:
        for idx, node in enumerate(ind1[1:], 1):
            types1[node.ret].append(idx)
        for idx, node in enumerate(ind2[1:], 1):
            types2[node.ret].append(idx)
        common_types = set(types1.keys()).intersection(set(types2.keys()))

    common_types = sorted(common_types, key= lambda x:x.__name__)
    if len(common_types) > 0:
        type_ = random.choice(list(common_types))

        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])

        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2