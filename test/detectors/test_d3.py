import unittest
import warnings

from detectors import DiscriminativeDriftDetector2019
from test.detectors.helper import get_simple_stream_drifts


class D3Test(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = DiscriminativeDriftDetector2019(
            n_reference_samples=10, recent_samples_proportion=1
        )

    def test_simple_detection(self):
        warnings.filterwarnings("ignore", category=UserWarning)
        drifts = get_simple_stream_drifts(self.detector)
        self.assertEqual(100, len(drifts))
        self.assertEqual(1, sum(drifts))
        self.assertEqual(1, sum(drifts[50:70]))


if __name__ == "__main__":
    unittest.main()
