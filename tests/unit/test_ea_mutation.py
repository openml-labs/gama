from collections import defaultdict
import pytest

import numpy as np

from gama.genetic_programming.components import Individual
from gama.genetic_programming.mutation import mut_replace_terminal, mut_replace_primitive, random_valid_mutation_in_place
from gama.genetic_programming.compilers.scikitlearn import compile_individual
from .unit_fixtures import pset, GaussianNB, RandomForestPipeline, LinearSVC

def test_mut_replace_terminal(RandomForestPipeline, pset):
    """ Tests if mut_replace_terminal replaces exactly one terminal. """
    _test_mutation(RandomForestPipeline, mut_replace_terminal, _mut_replace_terminal_is_applied, pset)


def test_mut_replace_terminal_none_available(GaussianNB, pset):
    """ Tests if mut_replace_terminal raises an exception if no valid mutation is possible. """
    with pytest.raises(ValueError) as error:
        mut_replace_terminal(GaussianNB, pset)

    assert "Individual has no terminals or no terminals suitable for mutation." in str(error.value)


def test_mut_replace_primitive_len_1(LinearSVC, pset):
    """ Tests if mut_replace_primitive replaces exactly one primitive. """
    _test_mutation(LinearSVC, mut_replace_primitive, _mut_replace_primitive_is_applied, pset)


def test_mut_replace_primitive_len_2(RandomForestPipeline, pset):
    """ Tests if mut_replace_primitive replaces exactly one primitive. """
    _test_mutation(RandomForestPipeline, mut_replace_primitive, _mut_replace_primitive_is_applied, pset)


def test_random_valid_mutation_with_all(RandomForestPipeline, pset):
    """ Test if a valid mutation is applied at random.

    I am honestly not sure of the best way to test this.
    Because of the random nature, we repeat this enough times to ensure each mutation is tried with
    probability >0.99999.
    """

    applied_mutation = defaultdict(int)
    N = _min_trials(n_mutations=4)

    for i in range(N):
        ind_clone = RandomForestPipeline.copy_as_new()
        random_valid_mutation_in_place(ind_clone, pset)
        if _mutShrink_is_applied(RandomForestPipeline, ind_clone)[0]:
            applied_mutation['shrink'] += 1
        elif _mutInsert_is_applied(RandomForestPipeline, ind_clone)[0]:
            applied_mutation['insert'] += 1
        elif _mut_replace_terminal_is_applied(RandomForestPipeline, ind_clone)[0]:
            applied_mutation['terminal'] += 1
        elif _mut_replace_primitive_is_applied(RandomForestPipeline, ind_clone)[0]:
            applied_mutation['primitive'] += 1
        else:
            assert False, "No mutation (or one that is unaccounted for) is applied."

    assert all([n > 0 for (mut, n) in applied_mutation.items()])


def test_random_valid_mutation_without_shrink(LinearSVC, pset):
    """ Test if a valid mutation is applied at random.

    I am honestly not sure of the best way to test this.
    Because of the random nature, we repeat this enough times to ensure each mutation is tried with
    probability >0.99999.
    """

    applied_mutation = defaultdict(int)
    N = _min_trials(n_mutations=3)

    for i in range(N):
        ind_clone = LinearSVC.copy_as_new()
        random_valid_mutation_in_place(ind_clone, pset)
        if _mutInsert_is_applied(LinearSVC, ind_clone)[0]:
            applied_mutation['insert'] += 1
        elif _mut_replace_terminal_is_applied(LinearSVC, ind_clone)[0]:
            applied_mutation['terminal'] += 1
        elif _mut_replace_primitive_is_applied(LinearSVC, ind_clone)[0]:
            applied_mutation['primitive'] += 1
        else:
            assert False, "No mutation (or one that is unaccounted for) is applied."

    assert all([n > 0 for (mut, n) in applied_mutation.items()])


def test_random_valid_mutation_without_terminal(GaussianNB, pset):
    """ Test if a valid mutation is applied at random.

    I am honestly not sure of the best way to test this.
    Because of the random nature, we repeat this enough times to ensure each mutation is tried with
    probability >0.99999.
    """
    # The tested individual contains no terminals and one primitive,
    # and thus is not eligible for replace_terminal and mutShrink.
    applied_mutation = defaultdict(int)
    N = _min_trials(n_mutations=2)

    for i in range(N):
        ind_clone = GaussianNB.copy_as_new()
        random_valid_mutation_in_place(ind_clone, pset)
        if _mutInsert_is_applied(GaussianNB, ind_clone)[0]:
            applied_mutation['insert'] += 1
        elif _mut_replace_primitive_is_applied(GaussianNB, ind_clone)[0]:
            applied_mutation['primitive'] += 1
        else:
            assert False, "No mutation (or one that is unaccounted for) is applied."

    assert all([n > 0 for (mut, n) in applied_mutation.items()])


def _min_trials(n_mutations, max_error_rate=0.0001):
    return int(np.ceil(np.log(max_error_rate) / np.log((n_mutations - 1) / n_mutations)))


def _mutShrink_is_applied(original, mutated):
    """ Checks if mutation was caused by `mut_shrink`.

    :param original: the pre-mutation individual
    :param mutated:  the post-mutation individual
    :return: (bool, str). If mutation was caused by function, True. False otherwise.
        str is a message explaining why mutation is not caused by function.
    """
    if len(list(original.primitives)) <= len(list(mutated.primitives)):
        return (False, "Number of primitives should be strictly less, was {} is {}."
                .format(len(list(original.primitives)), len(list(mutated.primitives))))
    return (True, None)


def _mutInsert_is_applied(original, mutated):
    """ Checks if mutation was caused by `mut_insert`.

    :param original: the pre-mutation individual
    :param mutated:  the post-mutation individual
    :return: (bool, str). If mutation was caused by function, True. False otherwise.
        str is a message explaining why mutation is not caused by function.
    """
    if len(list(original.primitives)) >= len(list(mutated.primitives)):
        return (False, "Number of primitives should be strictly greater, was {} is {}."
                .format(len(list(original.primitives)), len(list(mutated.primitives))))
    return (True, None)


def _mut_replace_terminal_is_applied(original, mutated):
    """ Checks if mutation was caused by `gama.ea.mutation.mut_replace_terminal`.

    :param original: the pre-mutation individual
    :param mutated:  the post-mutation individual
    :return: (bool, str). If mutation was caused by function, True. False otherwise.
        str is a message explaining why mutation is not caused by function.
    """
    if len(list(original.terminals)) != len(list(mutated.terminals)):
        return (False, "Number of terminals should be unchanged, was {} is {}."
                .format(len(list(original.terminals)), len(list(mutated.terminals))))

    replaced_terminals = [t1 for t1, t2 in zip(original.terminals, mutated.terminals) if str(t1) != str(t2)]
    if len(replaced_terminals) != 1:
        return (False, "Expected 1 replaced Terminal, found {}.".format(len(replaced_terminals)))
    return (True, None)


def _mut_replace_primitive_is_applied(original, mutated):
    """ Checks if mutation was caused by `gama.ea.mutation.mut_replace_primitive`.

    :param original: the pre-mutation individual
    :param mutated:  the post-mutation individual
    :return: (bool, str). If mutation was caused by function, True. False otherwise.
        str is a message explaining why mutation is not caused by function.
    """
    if len(list(original.primitives)) != len(list(mutated.primitives)):
        return (False, "Number of primitives should be unchanged, was {} is {}."
                .format(len(list(original.primitives)), len(list(mutated.primitives))))

    replaced_primitives = [p1 for p1, p2 in zip(original.primitives, mutated.primitives)
                           if str(p1._primitive) != str(p2._primitive)]
    if len(replaced_primitives) != 1:
        return (False, "Expected 1 replaced Primitive, found {}.".format(len(replaced_primitives)))
    return (True, None)


def _test_mutation(individual: Individual, mutation, mutation_check, pset):
    """ Test if an individual mutated by `mutation` passes `mutation_check` and compiles.

    :param individual: The individual to be mutated.
    :param mutation: function: ind -> (ind,). Should mutate the individual
    :param mutation_check: function: (ind1, ind2)->(bool, str).
       A function to check if ind2 could have been created by `mutation(ind1)`, see above functions.
    """
    ind_clone = individual.copy_as_new()
    mutation(ind_clone, pset)

    applied, message = mutation_check(individual, ind_clone)
    assert applied, message

    # Should be able to compile the individual, will raise an Exception if not.
    compile_individual(ind_clone, pset)

