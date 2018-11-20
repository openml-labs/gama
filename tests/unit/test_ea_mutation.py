from collections import defaultdict
import unittest

import numpy as np

from gama.genetic_programming.components import Individual
from gama.genetic_programming.mutation import mut_replace_terminal, mut_replace_primitive, random_valid_mutation_in_place
from gama.genetic_programming.compilers.scikitlearn import compile_individual
from gama.configuration.testconfiguration import clf_config
from gama import GamaClassifier


def mutation_test_suite():
    test_cases = [MutationTestCase]
    return unittest.TestSuite(map(unittest.TestLoader().loadTestsFromTestCase, test_cases))


class MutationTestCase(unittest.TestCase):
    """ Unit Tests for ea/mutation.py """

    def setUp(self):
        self.gama = GamaClassifier(random_state=0,
                                   config=clf_config,
                                   objectives=('accuracy', 'size'),
                                   keep_analysis_log=False)

        self.ind_strings = [
            "GaussianNB(data)",
            """RandomForestClassifier(
            FeatureAgglomeration(
                    data,
                    FeatureAgglomeration.affinity='l2',
                    FeatureAgglomeration.linkage='complete'
                    ),   
            RandomForestClassifier.bootstrap=True,
            RandomForestClassifier.criterion='gini',
            RandomForestClassifier.max_features=0.6,
            RandomForestClassifier.min_samples_leaf=7,  
            RandomForestClassifier.min_samples_split=6, 
            RandomForestClassifier.n_estimators=100)""",
            """LinearSVC(data,
            LinearSVC.C=0.001,
            LinearSVC.dual=True,
            LinearSVC.loss='squared_hinge',
            LinearSVC.penalty='l2',
            LinearSVC.tol=1e-05)"""
        ]

        self.individuals = {
            ind_str: Individual.from_string(''.join(ind_str.split()).replace(',', ', '), self.gama._pset)
            for ind_str in self.ind_strings
        }

    def tearDown(self):
        self.gama.delete_cache()

    def test_mut_replace_terminal(self):
        """ Tests if mut_replace_terminal replaces exactly one terminal. """
        ind1 = self.individuals[self.ind_strings[1]]
        self._test_mutation(ind1, mut_replace_terminal, self._mut_replace_terminal_is_applied)

    def test_mut_replace_terminal_none_available(self):
        """ Tests if mut_replace_terminal raises an exception if no valid mutation is possible. """
        ind = self.individuals[self.ind_strings[0]]
        ind_clone = ind.copy_as_new()

        with self.assertRaises(ValueError) as error:
            mut_replace_terminal(ind_clone, self.gama._pset)

        self.assertTrue("Individual has no terminals or no terminals suitable for mutation." in str(error.exception))

    def test_mut_replace_primitive_len_1(self):
        """ Tests if mut_replace_primitive replaces exactly one primitive. """
        ind1 = self.individuals[self.ind_strings[2]]
        self._test_mutation(ind1, mut_replace_primitive, self._mut_replace_primitive_is_applied)

    def test_mut_replace_primitive_len_2(self):
        """ Tests if mut_replace_primitive replaces exactly one primitive. """
        ind1 = self.individuals[self.ind_strings[1]]
        self._test_mutation(ind1, mut_replace_primitive, self._mut_replace_primitive_is_applied)

    def test_random_valid_mutation_with_all(self):
        """ Test if a valid mutation is applied at random.

        I am honestly not sure of the best way to test this.
        Because of the random nature, we repeat this enough times to ensure each mutation is tried with
        probability >0.99999.
        """

        applied_mutation = defaultdict(int)
        N = self._min_trials(n_mutations=4)

        for i in range(N):
            ind = self.individuals[self.ind_strings[1]]
            ind_clone = ind.copy_as_new()
            random_valid_mutation_in_place(ind_clone, self.gama._pset)
            if self._mutShrink_is_applied(ind, ind_clone)[0]:
                applied_mutation['shrink'] += 1
            elif self._mutInsert_is_applied(ind, ind_clone)[0]:
                applied_mutation['insert'] += 1
            elif self._mut_replace_terminal_is_applied(ind, ind_clone)[0]:
                applied_mutation['terminal'] += 1
            elif self._mut_replace_primitive_is_applied(ind, ind_clone)[0]:
                applied_mutation['primitive'] += 1
            else:
                self.fail("No mutation (or one that is unaccounted for) is applied.")

        self.assertTrue(all([n > 0 for (mut, n) in applied_mutation.items()]))

    def test_random_valid_mutation_without_shrink(self):
        """ Test if a valid mutation is applied at random.

        I am honestly not sure of the best way to test this.
        Because of the random nature, we repeat this enough times to ensure each mutation is tried with
        probability >0.99999.
        """

        applied_mutation = defaultdict(int)
        N = self._min_trials(n_mutations=3)

        for i in range(N):
            ind = self.individuals[self.ind_strings[2]]
            ind_clone = ind.copy_as_new()
            random_valid_mutation_in_place(ind_clone, self.gama._pset)
            if self._mutInsert_is_applied(ind, ind_clone)[0]:
                applied_mutation['insert'] += 1
            elif self._mut_replace_terminal_is_applied(ind, ind_clone)[0]:
                applied_mutation['terminal'] += 1
            elif self._mut_replace_primitive_is_applied(ind, ind_clone)[0]:
                applied_mutation['primitive'] += 1
            else:
                self.fail("No mutation (or one that is unaccounted for) is applied.")

        self.assertTrue(all([n > 0 for (mut, n) in applied_mutation.items()]))

    def test_random_valid_mutation_without_terminal(self):
        """ Test if a valid mutation is applied at random.

        I am honestly not sure of the best way to test this.
        Because of the random nature, we repeat this enough times to ensure each mutation is tried with
        probability >0.99999.
        """
        # The tested individual contains no terminals and one primitive,
        # and thus is not eligible for replace_terminal and mutShrink.
        applied_mutation = defaultdict(int)
        N = self._min_trials(n_mutations=2)

        for i in range(N):
            ind = self.individuals[self.ind_strings[0]]
            ind_clone = ind.copy_as_new()
            random_valid_mutation_in_place(ind_clone, self.gama._pset)
            if self._mutInsert_is_applied(ind, ind_clone)[0]:
                applied_mutation['insert'] += 1
            elif self._mut_replace_primitive_is_applied(ind, ind_clone)[0]:
                applied_mutation['primitive'] += 1
            else:
                self.fail("No mutation (or one that is unaccounted for) is applied.")

        self.assertTrue(all([n > 0 for (mut, n) in applied_mutation.items()]))

    def _min_trials(self, n_mutations, max_error_rate=0.0001):
        return int(np.ceil(np.log(max_error_rate) / np.log((n_mutations - 1) / n_mutations)))

    def _mutShrink_is_applied(self, original, mutated):
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

    def _mutInsert_is_applied(self, original, mutated):
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

    def _mut_replace_terminal_is_applied(self, original, mutated):
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

    def _mut_replace_primitive_is_applied(self, original, mutated):
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

    def _test_mutation(self, individual: Individual, mutation, mutation_check):
        """ Test if an individual mutated by `mutation` passes `mutation_check` and compiles.

        :param individual: The individual to be mutated.
        :param mutation: function: ind -> (ind,). Should mutate the individual
        :param mutation_check: function: (ind1, ind2)->(bool, str).
           A function to check if ind2 could have been created by `mutation(ind1)`, see above functions.
        """
        ind_clone = individual.copy_as_new()
        mutation(ind_clone, self.gama._pset)

        applied, message = mutation_check(individual, ind_clone)
        if not applied:
            self.fail(message)

        # Should be able to compile the individual, will raise an Exception if not.
        compile_individual(ind_clone, self.gama._pset)

