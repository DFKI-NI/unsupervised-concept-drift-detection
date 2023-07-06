import unittest

from detectors import UDetect
from test.detectors.helper import get_simple_random_stream_drifts


class UDetectTest(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = UDetect(n_samples=4, n_windows=5)

    def test_simple_detection(self):
        drifts = get_simple_random_stream_drifts(self.detector, seed=734)
        self.assertEqual(100, len(drifts))
        self.assertEqual(1, sum(drifts))
        self.assertEqual(1, sum(drifts[50:70]))


if __name__ == "__main__":
    unittest.main()
