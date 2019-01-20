import unittest

from gama.genetic_programming.components import Individual, Fitness
from gama.genetic_programming.selection import eliminate_from_pareto
from gama.configuration.testconfiguration import clf_config
from gama import GamaClassifier


def automl_gp_test_suite():
    test_cases = [AutomlGpTestCase]
    return unittest.TestSuite(map(unittest.TestLoader().loadTestsFromTestCase, test_cases))


class AutomlGpTestCase(unittest.TestCase):
    """ Unit Tests for some functions of automl_gp.py.
    
    Functions excluded (for now?):
    [ ] pset_from_config
    [ ] compile_individual 
        If this breaks it's probably not subtle, hard to test edge cases, leave it to system test?
    [ ] expression_to_component
    [ ] generate_valid
        If this breaks it's probably not subtle, hard to test edge cases, leave it to system test?

    [ ] eliminate_worst
    [ ] offspring_mate_and_mutate
    """
    
    def setUp(self):
        self.gama = GamaClassifier(config=clf_config, scoring='accuracy', keep_analysis_log=False)

        self.ind_strings = [
            "GaussianNB(data)",
            """RandomForestClassifier(
            FeatureAgglomeration(
                    data,
                    FeatureAgglomeration.affinity='l2',
                    FeatureAgglomeration.linkage='complete'),
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

        self.individual_list = [Individual.from_string(''.join(ind_str.split()).replace(',', ', '), self.gama._pset)
                                for ind_str in self.ind_strings]

    def tearDown(self):
        self.gama.delete_cache()

    def test_individual_length(self):
        # GaussianNB
        self.assertEqual(len(list(self.individual_list[0].primitives)), 1)
        # RandomForest(FeatureAgglomeration)
        self.assertEqual(len(list(self.individual_list[1].primitives)), 2)
        # LinearSVC
        self.assertEqual(len(list(self.individual_list[2].primitives)), 1)

    def test_eliminate_NSGA(self):
        self.individual_list[0].fitness = Fitness((3, -2), 0, 0)
        self.individual_list[1].fitness = Fitness((4, -2), 0, 0)
        self.individual_list[2].fitness = Fitness((3, -1), 0, 0)

        eliminated = eliminate_from_pareto(pop=self.individual_list, n=1)
        self.assertListEqual(eliminated, [self.individual_list[0]],
                             "The element (3, -2) is dominated by both (3, -1)  and (4, -2) so should be eliminated.")

        # Check order independence
        eliminated = eliminate_from_pareto(pop=list(reversed(self.individual_list)), n=1)
        self.assertListEqual(eliminated, [self.individual_list[0]],
                             "Individual should be dominated regardless of order.")
