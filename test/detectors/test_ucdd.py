import unittest

from detectors import UCDD
from test.detectors.helper import get_simple_random_stream_drifts


class UCDDTest(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = UCDD(
            n_reference_samples=10,
            n_recent_samples=10,
            threshold=0.13,
            stability_offset=1,
            seed=523,
        )

    def test_simple_detection(self):
        drifts = get_simple_random_stream_drifts(self.detector, seed=734)
        self.assertEqual(100, len(drifts))
        self.assertEqual(1, sum(drifts))
        self.assertEqual(1, sum(drifts[50:70]))


if __name__ == "__main__":
    unittest.main()
