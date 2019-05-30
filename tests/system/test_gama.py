import unittest

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from gama import GamaClassifier


def gama_test_suite():
    test_cases = [GamaSystemTestCase]
    return unittest.TestSuite(map(unittest.TestLoader().loadTestsFromTestCase, test_cases))


class GamaSystemTestCase(unittest.TestCase):
    """ Contain complete system tests for Gama. """
    
    def setUp(self):
        import logging
        self.gama = GamaClassifier(random_state=0, max_total_time=120, verbosity=logging.DEBUG)
    
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
