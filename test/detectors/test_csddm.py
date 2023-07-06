import unittest
import warnings

from detectors import ClusteredStatisticalTestDriftDetectionMethod
from test.detectors.helper import get_simple_stream_drifts


class CSDDMTest(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = ClusteredStatisticalTestDriftDetectionMethod(
            n_samples=10, n_clusters=1, confidence=0.001
        )

    def test_simple_detection(self):
        warnings.filterwarnings("ignore", category=UserWarning)
        drifts = get_simple_stream_drifts(self.detector)
        self.assertEqual(100, len(drifts))
        self.assertEqual(1, sum(drifts))
        self.assertEqual(1, sum(drifts[50:70]))


if __name__ == "__main__":
    unittest.main()
