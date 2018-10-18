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
        pop1 = g1._toolbox.population(n=10)

        #  For now, operations contains a variable which caches created individuals.
        importlib.reload(gama.ea.operations)

        g2 = gama.GamaClassifier(random_state=1)
        pop2 = g2._toolbox.population(n=10)
        for ind1, ind2 in zip(pop1, pop2):
            self.assertEqual(str(ind1), str(ind2), "The initial population should be reproducible.")
