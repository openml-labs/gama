import time
import unittest

from gama.utilities.generic.stopwatch import Stopwatch


def stopwatch_test_suite():
    test_cases = [StopwatchUnitTestCase]
    return unittest.TestSuite(map(unittest.TestLoader().loadTestsFromTestCase, test_cases))


class StopwatchUnitTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_stopwatch_initialization_zero(self):
        """ Test that elapsed time is 0 if stopwatch is not started yet. """
        sw = Stopwatch()
        self.assertEqual(sw.elapsed_time, 0)

    def test_stopwatch_elapsed_time_while_running(self):
        """ Tests that elapsed_time increments as expected while running. """
        with Stopwatch() as sw:
            self.assertGreaterEqual(sw.elapsed_time, 0)
            self.assertLess(sw.elapsed_time, 1)

            time.sleep(1)
            self.assertGreaterEqual(sw.elapsed_time, 1)
            self.assertLess(sw.elapsed_time, 2)

    def test_stopwatch_elapsed_time_after_running(self):
        """ Tests that time elapsed it stored after exiting the context. """
        with Stopwatch() as sw:
            time.sleep(1)
        time.sleep(1)
        self.assertGreaterEqual(sw.elapsed_time, 1)
        self.assertLess(sw.elapsed_time, 2)
