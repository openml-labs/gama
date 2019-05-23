import time
import unittest

from gama.utilities.generic.timekeeper import TimeKeeper


def timekeeper_test_suite():
    test_cases = [TimeKeeperUnitTestCase]
    return unittest.TestSuite(map(unittest.TestLoader().loadTestsFromTestCase, test_cases))


class TimeKeeperUnitTestCase(unittest.TestCase):

    def setUp(self):
        self._round_error = 2

    def tearDown(self):
        pass

    def test_timekeeper_total_time_remaning_error_if_total_time_zero(self):
        """ Ensure `total_time_remaining` is unavailable if `total_time` is not set. """
        timekeeper = TimeKeeper(total_time=0)
        with self.assertRaises(RuntimeError):
            _ = timekeeper.total_time_remaining

    def test_timekeeper_stopwatch_normal_behavior(self):
        """ Ensure normal stopwatch functionality for stopwatch returned by context manager. """
        timekeeper = TimeKeeper()
        with timekeeper.start_activity('test activity') as sw:
            self.assertAlmostEqual(sw.elapsed_time, 0, places=self._round_error)
            self.assertTrue(sw._is_running)
            time.sleep(1)
            self.assertAlmostEqual(sw.elapsed_time, 1, places=self._round_error)
            self.assertTrue(sw._is_running)

        time.sleep(1)
        self.assertFalse(sw._is_running)
        self.assertAlmostEqual(sw.elapsed_time, 1, places=self._round_error)

    def test_timekeeper_total_remaining_time(self):
        """ Ensure total remaining time is correct across activities. """
        total_time = 10
        timekeeper = TimeKeeper(total_time=total_time)
        self.assertEqual(timekeeper.total_time_remaining, total_time)

        activity_length = 1
        with timekeeper.start_activity('part one'):
            time.sleep(activity_length)

        time.sleep(1)
        self.assertAlmostEqual(activity_length, timekeeper.activity_durations['part one'], places=self._round_error)
        self.assertAlmostEqual(total_time - activity_length, timekeeper.total_time_remaining, places=self._round_error)
