""" Contains all unit tests for """
import unittest

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from gama import GamaClassifier


def gama_test_suite():
    test_cases = [GamaUnitTestCase, GamaSystemTestCase]
    return unittest.TestSuite(map(unittest.TestLoader().loadTestsFromTestCase, test_cases))


class GamaUnitTestCase(unittest.TestCase):
    """ Contains unit tests for Gama, that test small components. """
    
    def setUp(self):
        self.gama = GamaClassifier(random_state=0)
    
    def tearDown(self):
        pass
    
    def test_reproducible_initialization(self):
        g1 = GamaClassifier(random_state=1)
        pop1 = g1._toolbox.population(n=10)
        g2 = GamaClassifier(random_state=1)
        pop2 = g2._toolbox.population(n=10)
        for ind1, ind2 in zip(pop1, pop2):
            self.assertEqual(str(ind1), str(ind2), "The initial population should be reproducible.")


class GamaSystemTestCase(unittest.TestCase):
    """ Contain complete system tests for Gama. """
    
    def setUp(self):
        self.gama = GamaClassifier(random_state=0, max_total_time=120)
    
    def tearDown(self):
        pass
        
    def test_full_system_single_core(self):
        X, y = load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
                
        # Add checks on individuals (reproducibility)
        self.gama.fit(X_train, y_train)
        
        # Add checks
        self.gama.predict(X_test)

    def test_full_system_multi_core(self):
        self.gama._n_jobs = 2
        X, y = load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

        # Add checks on individuals (reproducibility)
        self.gama.fit(X_train, y_train)

        # Add checks
        self.gama.predict(X_test)
