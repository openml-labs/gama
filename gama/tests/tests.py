import unittest
import warnings

from gama.tests.test_automl_gp import automl_gp_test_suite
from gama.tests.test_gama import gama_test_suite
from gama.tests.test_utilities_generic_paretofront import paretofront_test_suite

# ..\deap\creator.py:141: RuntimeWarning: A class named 'FitnessMax'/'Individual' has already
#  been created and it will be overwritten. Consider deleting previous creation of that class
# or rename it. RuntimeWarning)
warnings.filterwarnings('ignore',
                        message='A class named ',
                        category=RuntimeWarning,
                        module='deap',
                        lineno=141)

tests_succeeded = unittest.TextTestRunner().run(paretofront_test_suite()).wasSuccessful()
tests_succeeded &= unittest.TextTestRunner().run(automl_gp_test_suite()).wasSuccessful()
if tests_succeeded:
    print('continuing tests....')
    #unittest.TextTestRunner().run(gama_test_suite())
