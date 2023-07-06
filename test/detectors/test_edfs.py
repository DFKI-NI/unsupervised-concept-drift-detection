import random
import unittest

from detectors.edfs import EDFS, EDFSMode
from test.detectors.helper import get_simple_random_stream_drifts


class EDFSTest(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = EDFS(n_subspaces=1, feature_percentage=1, window_size=10)

    def test_simple_detection(self):
        drifts = get_simple_random_stream_drifts(self.detector, seed=62334)
        self.assertEqual(100, len(drifts))
        self.assertEqual(1, sum(drifts))
        self.assertEqual(1, sum(drifts[50:70]))


class ResetTest(unittest.TestCase):
    def test_unsupported_mode(self):
        edfs = EDFS(0, 0, "DEFINITELY WRONG MODE")
        with self.assertRaises(ValueError):
            edfs.reset([])

    def test_subspace_selection_mode(self):
        edfs = EDFS(0, 0, EDFSMode.SUBSPACE_SELECTION)
        with self.assertRaises(NotImplementedError):
            edfs.reset([1])

    def test_one_subspace_one_feature(self):
        edfs = EDFS(1, 1, EDFSMode.RANDOM)
        edfs.reset({"a": 1})
        self.assertEqual(len(edfs.subspaces), 1)
        self.assertEqual(len(edfs.subspaces[0]), 1)

    def test_random_subspaces_random_features(self):
        for _ in range(10):
            n_features = random.randint(2, 18)
            n_subspaces = random.randint(2, 22)
            edfs = EDFS(n_subspaces, 0.5)
            features = {f"{i}": i for i in range(n_features * 2)}
            edfs.reset(features)
            self.assertEqual(len(edfs.subspaces), n_subspaces)
            for subspace in edfs.subspaces:
                self.assertEqual(len(subspace), n_features)


class UpdateTest(unittest.TestCase):
    def test_update_no_drift(self):
        """
        Test that no drift is called when no drift happens.
        """
        sample = {f"feature_{i}": {i} for i in range(20)}
        edfs = EDFS(5, 0.4)
        for _ in range(200):
            drift = edfs.update(sample)
            self.assertFalse(drift)

    @unittest.skip("Test is not working as intended, yet.")
    def test_update_with_drift(self):
        edfs = EDFS(5, 0.4)
        drift = False
        for _ in range(100):
            sample = {f"feature_{i}": {random.gauss(0, 1)} for i in range(20)}
            drift = edfs.update(sample)
            self.assertFalse(drift)
        for _ in range(1000):
            sample = {f"feature_{i}": {random.paretovariate(0.5)} for i in range(20)}
            drift = edfs.update(sample)
            if drift:
                break
        self.assertTrue(drift)


if __name__ == "__main__":
    unittest.main()
