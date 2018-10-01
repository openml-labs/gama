import os
import sys
import unittest
import warnings

from gama.tests.test_automl_gp import automl_gp_test_suite
from gama.tests.test_gama import gama_test_suite
from gama.tests.test_ea_mutation import mutation_test_suite
from gama.tests.test_utilities_generic_paretofront import paretofront_test_suite
from gama.tests.test_utilities_generic_stopwatch import stopwatch_test_suite
from gama.tests.test_gamaclassifier import gamaclassifier_test_suite
from gama.tests.test_gamaregressor import gamaregressor_test_suite
from gama.tests.test_ea_metrics import metrics_test_suite

# ..\deap\creator.py:141: RuntimeWarning: A class named 'FitnessMax'/'Individual' has already
#  been created and it will be overwritten. Consider deleting previous creation of that class
# or rename it. RuntimeWarning)
warnings.filterwarnings('ignore',
                        message='A class named ',
                        category=RuntimeWarning,
                        module='deap',
                        lineno=141)


def run_suite(suite):
    return unittest.TextTestRunner(verbosity=2).run(suite())


if __name__ == '__main__':
    suite = sys.argv[1] if len(sys.argv) > 1 else None

    if os.environ.get('TEST_SUITE') == 'unit' or suite == 'unit':
        run_suite(paretofront_test_suite)
        run_suite(mutation_test_suite)
        run_suite(automl_gp_test_suite)
        run_suite(metrics_test_suite)
        run_suite(stopwatch_test_suite)
    elif os.environ.get('TEST_SUITE') == 'system' or suite == 'system':
        run_suite(gama_test_suite)
        run_suite(gamaclassifier_test_suite)
        run_suite(gamaregressor_test_suite)
    else:
        print('NO TEST SUITE VARIABLE DETECTED. RUN NO TEST.')
        print('Command line suite specified:', suite)
        quit(-1)
