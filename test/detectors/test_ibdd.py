import unittest

from detectors import ImageBasedDriftDetector
from test.detectors.helper import get_simple_stream_drifts


class IBDDTest(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = ImageBasedDriftDetector(
            n_samples=10, update_interval=5, n_permutations=10, seed=37
        )

    def test_simple_detection(self):
        drifts = get_simple_stream_drifts(self.detector)
        self.assertEqual(100, len(drifts))
        self.assertEqual(3, sum(drifts))
        self.assertEqual(3, sum(drifts[50:70]))


if __name__ == "__main__":
    unittest.main()
