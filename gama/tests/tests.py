import unittest
from gama.tests.test_automl_gp import automl_gp_test_suite
from gama.tests.test_gama import gama_test_suite

result = unittest.TextTestRunner().run(automl_gp_test_suite())
if result.wasSuccessful():
    unittest.TextTestRunner().run(gama_test_suite())
