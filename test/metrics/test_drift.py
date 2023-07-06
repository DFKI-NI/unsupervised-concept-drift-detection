"""This module tests the calculation of drift metrics."""
import unittest

import numpy as np

from metrics.drift import calculate_drift_metrics


class CalculateDriftMetricsTest(unittest.TestCase):
    """
    This class contains multiple tests for the calculate_drift_metrics method.
    """

    def test_all_drifts_missed(self):
        """
        Test that missing all drifts returns an mdr of 1.0.
        """
        drifts = [10]
        detections = [1, 2, 3, 4, 5]
        metrics = calculate_drift_metrics(drifts, detections)
        self.assertTrue(np.isnan(metrics["mtr"]))
        self.assertEqual(1.0, metrics["mtfa"])
        self.assertTrue(np.isnan(metrics["mtd"]))
        self.assertEqual(1.0, metrics["mdr"])

    def test_no_false_alerts(self):
        """
        Test that a detection the following timestep results in an mtd of 1.0 and an mdr of 0.
        """
        drifts = [0, 2, 4]
        detections = [1, 3, 5]
        metrics = calculate_drift_metrics(drifts, detections)
        self.assertTrue(np.isnan(metrics["mtr"]))
        self.assertTrue(np.isnan(metrics["mtfa"]))
        self.assertEqual(1.0, metrics["mtd"])
        self.assertEqual(0.0, metrics["mdr"])

    def test_miss_first(self):
        """
        Test that missing the first drift is handled correctly.
        """
        drifts = [0, 1, 3]
        detections = [2, 4]
        metrics = calculate_drift_metrics(drifts, detections)
        self.assertTrue(np.isnan(metrics["mtr"]))
        self.assertTrue(np.isnan(metrics["mtfa"]))
        self.assertEqual(1.0, metrics["mtd"])
        self.assertAlmostEqual(1 / 3, metrics["mdr"], places=7)

    def test_false_alerts_in_the_end(self):
        """
        Test that multiple false alerts after the final drift are handled correctly.
        """
        drifts = [0]
        detections = [1, 2, 4, 7]
        metrics = calculate_drift_metrics(drifts, detections)
        self.assertEqual(2.5 / 1, metrics["mtr"])
        self.assertEqual(2.5, metrics["mtfa"])
        self.assertEqual(1, metrics["mtd"])
        self.assertEqual(0, metrics["mdr"])

    def test_instant_detection(self):
        """
        Test that the instant detection of a drift is handled correctly.
        """
        drifts = [0]
        detections = [0]
        metrics = calculate_drift_metrics(drifts, detections)
        self.assertTrue(np.isnan(metrics["mtr"]))
        self.assertTrue(np.isnan(metrics["mtfa"]))
        self.assertEqual(0, metrics["mtd"])
        self.assertEqual(0, metrics["mdr"])

    def test_three_late_misses(self):
        """
        Test that missing three drifts in the end is handled correctly.
        """
        drifts = [1, 2, 3]
        detections = []
        metrics = calculate_drift_metrics(drifts, detections)
        self.assertTrue(np.isnan(metrics["mtr"]))
        self.assertTrue(np.isnan(metrics["mtfa"]))
        self.assertTrue(np.isnan(metrics["mtd"]))
        self.assertEqual(1.0, metrics["mdr"])

    def test_mtr0(self):
        """
        Test that all metrics are calculated correctly.
        """
        drifts = [1, 11, 12, 14, 20, 25, 26]
        detections = [4, 6, 10, 13, 16, 18, 24, 28]
        metrics = calculate_drift_metrics(drifts, detections)
        self.assertEqual(2.5 * 5 / 7, metrics["mtr"])
        self.assertEqual(6, metrics["mtfa"])
        self.assertEqual(2.4, metrics["mtd"])
        self.assertEqual(2 / 7, metrics["mdr"])

    def test_mtr1(self):
        """
        Test that all metrics are calculated correctly.
        """
        drifts = [1, 11, 12, 14, 20, 25, 26]
        detections = [2, 2.5, 10.5, 13, 15, 21]
        metrics = calculate_drift_metrics(drifts, detections)
        self.assertAlmostEqual(8 * 4 / 7, metrics["mtr"], places=7)
        self.assertEqual(8, metrics["mtfa"])
        self.assertEqual(1, metrics["mtd"])
        self.assertAlmostEqual(3 / 7, metrics["mdr"], places=7)
