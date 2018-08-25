import unittest

from deap import gp, creator

from gama.configuration.testconfiguration import clf_config
from gama.ea.automl_gp import compile_individual, individual_length, eliminate_NSGA
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
        self.gama = GamaClassifier(config=clf_config, objectives=('accuracy', 'size'))

        self.ind_strings = [
            "GaussianNB(data)",
            """RandomForestClassifier(
            FeatureAgglomeration(
                    data,
                    FeatureAgglomeration.linkage='complete',
                    FeatureAgglomeration.affinity='l2'), 
            RandomForestClassifier.n_estimators=100, 
            RandomForestClassifier.criterion='gini', 
            RandomForestClassifier.max_features=0.6000000000000001, 
            RandomForestClassifier.min_samples_split=6, 
            RandomForestClassifier.min_samples_leaf=7, 
            RandomForestClassifier.bootstrap=True)""",
            """LinearSVC(data,
            LinearSVC.penalty='l2',
            LinearSVC.loss='squared_hinge',
            LinearSVC.dual=True,
            LinearSVC.tol=1e-05,
            LinearSVC.C=0.001)"""
        ]

        self.individual_list = [creator.Individual(gp.PrimitiveTree.from_string(ind_str, self.gama._pset))
                                for ind_str in self.ind_strings]

    def tearDown(self):
        self.gama.delete_cache()

    def test_individual_length(self):
        # GaussianNB
        self.assertEqual(individual_length(self.individual_list[0]), 1)
        # RandomForest(FeatureAgglomeration)
        self.assertEqual(individual_length(self.individual_list[1]), 2)
        # LinearSVC
        self.assertEqual(individual_length(self.individual_list[2]), 1)

    def test_eliminate_NSGA(self):
        self.individual_list[0].fitness.wvalues = (2, 1)
        self.individual_list[1].fitness.wvalues = (1, 2)
        self.individual_list[2].fitness.wvalues = (3, 1)

        eliminated = eliminate_NSGA(pop=self.individual_list, n=1)
        self.assertListEqual(eliminated, [self.individual_list[0]])

        # Check order independence
        eliminated = eliminate_NSGA(pop=list(reversed(self.individual_list)), n=1)
        self.assertListEqual(eliminated, [self.individual_list[0]],
                             "Individual should be dominated regardless of order.")
