import unittest

from deap import gp

from gama.ea.automl_gp import compile_individual
from gama.ea.mutation import mut_replace_primitive, mut_replace_terminal, find_unmatched_terminal
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
    [ ] generate_valid
        If this breaks it's probably not subtle, hard to test edge cases, leave it to system test?

    [ ] expression_to_component
    [ ] individual_length
    [ ] eliminate_NSGA
    [ ] eliminate_worst
    [ ] offspring_mate_and_mutate
    """
    
    def setUp(self):
        pass

    def tearDown(self):
        pass
