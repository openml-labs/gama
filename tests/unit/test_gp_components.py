import unittest

import gama

from gama.genetic_programming.components import Individual


def gp_components_test_suite():
    test_cases = [IndividualUnitTestCase]
    return unittest.TestSuite(map(unittest.TestLoader().loadTestsFromTestCase, test_cases))


class IndividualUnitTestCase(unittest.TestCase):
    """ Contains unit tests for Gama, that test small components. """

    def setUp(self):
        g = gama.GamaClassifier(random_state=1, keep_analysis_log=False)
        self.pset = g._pset

        self.individuals = [
            ("GaussianNB(data)", 1, 0),
            ("""RandomForestClassifier(
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
            RandomForestClassifier.n_estimators=100)""", 2, 8),
            ("""LinearSVC(data,
            LinearSVC.C=0.001,
            LinearSVC.dual=True,
            LinearSVC.loss='squared_hinge',
            LinearSVC.penalty='l2',
            LinearSVC.tol=1e-05)""", 1, 5)
        ]

    def tearDown(self):
        pass

    def test_individual_from_string(self):
        """ Individual can be instantiated from a string. """
        return
        for ind_str, n_primitives, n_terminals in self.individuals:
            individual = Individual.from_string(ind_str, self.pset)
            self.assertEqual(n_primitives, len(individual.primitives))
            self.assertEqual(n_terminals, len(individual.terminals))

    def test_individual_copy_is_deep(self):
        return
        from gama.genetic_programming.mutation import mut_insert
        original = Individual.from_string("GaussianNB(data)", self.pset)
        length_before = len(original.primitives)
        copy = original.copy_as_new()
        mut_insert(copy, self.pset)

        self.assertGreater(len(copy.primitives), len(original.primitives))
        self.assertEqual(length_before, len(original.primitives))

