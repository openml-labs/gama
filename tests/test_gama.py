""" Contains all unit tests for """
import unittest

from gama import Gama

def gama_suite():
    test_cases = [GamaUnitTestCase, GamaSystemTestCase]
    return unittest.TestSuite(map(unittest.TestLoader().loadTestsFromTestCase, test_cases))

class GamaUnitTestCase(unittest.TestCase):
    """ Contains unit tests for Gama, that test small components. """
    
    def setUp(self):
        self.gama = Gama(random_state=0)
    
    def tearDown(self):
        pass
    
    def test_same_initial_population(self):   
        g1 = Gama(random_state=1)
        pop1 = g1._toolbox.population(n=10)
        g2 = Gama(random_state=1)
        pop2 = g2._toolbox.population(n=10)
        for ind1, ind2 in zip(pop1, pop2):
            self.assertEqual(str(ind1), str(ind2), "The initial population should be reproducible.")
        
class GamaSystemTestCase(unittest.TestCase):
    """ Contain complete system tests for Gama. """
    
    def setUp(self):
        self.gama = Gama(random_state=0)
    
    def tearDown(self):
        pass
        
    def test_full_system(self):
        self.assertTrue(False, "Implement test.")
        