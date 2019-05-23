import time
import unittest

from gama.utilities.generic.stopwatch import Stopwatch


def stopwatch_test_suite():
    test_cases = [StopwatchUnitTestCase]
    return unittest.TestSuite(map(unittest.TestLoader().loadTestsFromTestCase, test_cases))


class StopwatchUnitTestCase(unittest.TestCase):

    def setUp(self):
        self._round_error = 2

    def tearDown(self):
        pass

    def test_stopwatch_initialization_zero(self):
        """ Test that elapsed time is 0 if stopwatch is not started yet. """
        sw = Stopwatch()
        self.assertEqual(sw.elapsed_time, 0)

    def test_stopwatch_elapsed_time_while_running(self):
        """ Tests that elapsed_time increments as expected while running. """
        with Stopwatch() as sw:
            self.assertAlmostEqual(sw.elapsed_time, 0, places=self._round_error)
            time.sleep(1)
            self.assertAlmostEqual(sw.elapsed_time, 1, places=self._round_error)

    def test_stopwatch_elapsed_time_after_running(self):
        """ Tests that time elapsed is stored after exiting the context. """
        with Stopwatch() as sw:
            time.sleep(1)
        time.sleep(1)
        self.assertAlmostEqual(sw.elapsed_time, 1, places=self._round_error)
