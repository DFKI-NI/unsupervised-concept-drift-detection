import unittest

import numpy as np
from scipy import stats

from detectors import BayesianNonparametricDetectionMethod
from test.detectors.helper import get_simple_stream_drifts


class PolyaTreeTestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bndm = BayesianNonparametricDetectionMethod(10)

    @staticmethod
    def normalize(sample_one, sample_two):
        data = np.concatenate((sample_one, sample_two))
        sample_one = (sample_one - np.median(data, axis=0)) / stats.iqr(data, axis=0)
        sample_two = (sample_two - np.median(data, axis=0)) / stats.iqr(data, axis=0)
        return sample_one, sample_two

    def test_simple_detection(self):
        drifts = get_simple_stream_drifts(self.bndm)
        self.assertEqual(100, len(drifts))
        self.assertEqual(1, sum(drifts))
        self.assertEqual(1, sum(drifts[50:70]))

    def test_polya_tree_test_no_overlap(self):
        array_one, array_two = self.normalize(np.zeros(10), np.zeros(10) + 100)
        log_odd_ratios = self.bndm.polya_tree_test(array_one, array_two, 0, "")
        test_statistic = 1 / (1 + np.exp(-log_odd_ratios))
        self.assertAlmostEqual(0, test_statistic, places=4)

    def test_polya_tree_test_full_overlap(self):
        array_one, array_two = self.normalize(np.arange(10), np.arange(10))
        log_odd_ratios = self.bndm.polya_tree_test(array_one, array_two, 0, "")
        test_statistic = 1 / (1 + np.exp(-log_odd_ratios))
        self.assertTrue(test_statistic > 0.5)


class GetIntervalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bndm = BayesianNonparametricDetectionMethod(0)

    def test_initial_partitions(self):
        lower, upper = self.bndm._get_interval("0")
        self.assertEqual(-np.inf, lower)
        self.assertEqual(0, upper)
        lower, upper = self.bndm._get_interval("1")
        self.assertEqual(0, lower)
        self.assertEqual(np.inf, upper)

    def test_second_level_partitions(self):
        lower, upper = self.bndm._get_interval("00")
        self.assertAlmostEqual(-np.inf, lower)
        self.assertAlmostEqual(-0.6745, upper, places=4)
        lower, upper = self.bndm._get_interval("01")
        self.assertAlmostEqual(-0.6745, lower, places=4)
        self.assertAlmostEqual(0, upper)
        lower, upper = self.bndm._get_interval("10")
        self.assertAlmostEqual(0, lower)
        self.assertAlmostEqual(0.6745, upper, places=4)
        lower, upper = self.bndm._get_interval("11")
        self.assertAlmostEqual(0.6745, lower, places=4)
        self.assertAlmostEqual(np.inf, upper)


class GetSamplesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bndm = BayesianNonparametricDetectionMethod(10)

    @staticmethod
    def normalize(sample_one, sample_two):
        data = np.concatenate((sample_one, sample_two))
        sample_one = (sample_one - np.median(data, axis=0)) / stats.iqr(data, axis=0)
        sample_two = (sample_two - np.median(data, axis=0)) / stats.iqr(data, axis=0)
        return sample_one, sample_two

    def test_one_feature(self):
        sample_one_exp = np.arange(10)
        sample_two_exp = np.arange(10) + 10
        sample_one_exp, sample_two_exp = self.normalize(sample_one_exp, sample_two_exp)
        self.bndm.data_window = np.concatenate(
            (sample_one_exp, sample_two_exp)
        ).reshape((20, 1))
        sample_one, sample_two = self.bndm._get_samples(0)
        self.assertAlmostEqual(0, sum(sample_one - sample_one_exp))
        self.assertAlmostEqual(0, sum(sample_two - sample_two_exp))

    def test_two_features(self):
        sample_one_exp = np.zeros((10, 2))
        sample_two_exp = np.zeros((10, 2)) + 10
        for i in range(2):
            sample_one_result, sample_two_result = self.normalize(
                sample_one_exp[:, i], sample_two_exp[:, i]
            )
            self.bndm.data_window = np.concatenate(
                (sample_one_exp, sample_two_exp)
            ).reshape((20, 2))
            sample_one, sample_two = self.bndm._get_samples(i)
            self.assertTrue(np.array_equal(sample_one_result, sample_one))
            self.assertTrue(np.array_equal(sample_two_result, sample_two))


if __name__ == "__main__":
    unittest.main()
