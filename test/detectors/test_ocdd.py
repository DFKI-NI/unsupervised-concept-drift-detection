import unittest

from sklearn.svm import OneClassSVM

from detectors import OneClassDriftDetector
from test.detectors.helper import get_simple_random_stream_drifts


class OCDDTest(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = OneClassDriftDetector(
            n_samples=20,
            threshold=0.9,
            outlier_detector_class=OneClassSVM,
            outlier_detector_kwargs={"nu": 0.5, "kernel": "rbf", "gamma": "auto"},
        )

    def test_simple_detection(self):
        drifts = get_simple_random_stream_drifts(self.detector, seed=22)
        self.assertEqual(100, len(drifts))
        self.assertEqual(1, sum(drifts))
        self.assertEqual(1, sum(drifts[50:70]))


if __name__ == "__main__":
    unittest.main()
