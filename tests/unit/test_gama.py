import importlib
import unittest

import gama


def gama_test_suite():
    test_cases = [GamaUnitTestCase]
    return unittest.TestSuite(map(unittest.TestLoader().loadTestsFromTestCase, test_cases))


class GamaUnitTestCase(unittest.TestCase):
    """ Contains unit tests for Gama, that test small components. """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_reproducible_initialization(self):
        g1 = gama.GamaClassifier(random_state=1)
        pop1 = [g1._operator_set.individual() for _ in range(10)]

        g2 = gama.GamaClassifier(random_state=1)
        pop2 = [g2._operator_set.individual() for _ in range(10)]
        for ind1, ind2 in zip(pop1, pop2):
            self.assertEqual(ind1.pipeline_str(), ind2.pipeline_str(), "The initial population should be reproducible.")
