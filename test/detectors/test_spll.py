import unittest

from detectors import SemiParametricLogLikelihood
from test.detectors.helper import get_simple_random_stream_drifts


class SPLLTest(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = SemiParametricLogLikelihood(
            n_samples=20, n_clusters=2, threshold=0.0005
        )

    def test_simple_detection(self):
        drifts = get_simple_random_stream_drifts(self.detector, seed=33)
        self.assertEqual(100, len(drifts))
        self.assertEqual(1, sum(drifts))
        self.assertEqual(1, sum(drifts[50:70]))


if __name__ == "__main__":
    unittest.main()
