"""This module tests the unsupervised NN-DVI concept drift detector."""
import unittest
import numpy as np

from detectors import NNDVI

from test.detectors.helper import get_simple_random_stream_drifts

# pylint: disable=W0212


class NNDVITest(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = NNDVI(n_samples=10, k_neighbors=3, n_permutations=50, seed=128)

    def test_simple_detection(self):
        drifts = get_simple_random_stream_drifts(self.detector, seed=723)
        self.assertEqual(100, len(drifts))
        self.assertEqual(1, sum(drifts))
        self.assertEqual(1, sum(drifts[50:70]))


class GetDistanceTest(unittest.TestCase):
    """
    This class tests the _get_nnps_distance method of the NNDVI class.
    """

    def setUp(self):
        self.nndvi = NNDVI(n_samples=2)

    def test_distance(self):
        """
        Test the distance calculation.
        """
        particle_matrix = np.ndarray(
            (4, 4),
            buffer=np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [1, 1, 1, 1]]),
            dtype=int,
        )
        first_set = np.array([0, 1])
        second_set = np.array([2, 3])
        distance = self.nndvi._get_nnps_distance(particle_matrix, first_set, second_set)
        self.assertAlmostEqual(distance, (1 / 4) * (4832 / 2431))

    def test_paper_example_distance(self):
        """
        Test the distance calculation with the example from the paper.
        """
        particle_matrix = np.ndarray(
            (4, 4),
            buffer=np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]]),
            dtype=int,
        )
        first_set = np.array([0, 1])
        second_set = np.array([2, 3])
        distance = self.nndvi._get_nnps_distance(particle_matrix, first_set, second_set)
        self.assertAlmostEqual(distance, 5 / 7)

    def test_distance_equal_sets(self):
        """
        Test that the distance of equal sets amounts to 0.
        """
        particle_matrix = np.ndarray(
            (4, 4),
            buffer=np.array([[1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 1, 1, 1]]),
            dtype=int,
        )
        indices = np.arange(4)
        distance = self.nndvi._get_nnps_distance(particle_matrix, indices, indices)
        self.assertEqual(distance, 0)


class GetIndicesTest(unittest.TestCase):
    """
    This class tests the _get_indices method of the NNDVI class.
    """

    def setUp(self):
        self.nndvi = NNDVI(n_samples=5)

    def test_first_indices_set(self):
        """
        Test that the correct indices are set for the first indices set.
        """
        first_set, _ = self.nndvi._get_indices(self.nndvi.n_samples)
        np.testing.assert_array_equal(first_set, np.arange(self.nndvi.n_samples))

    def test_second_indices_set0(self):
        """
        Test that the correct indices are set for the second indices set when it overlaps with the
        first indices set.
        """
        _, second_set = self.nndvi._get_indices(self.nndvi.n_samples)
        np.testing.assert_array_equal(second_set, np.arange(self.nndvi.n_samples))

    def test_second_indices_set1(self):
        """
        Test that the correct indices are set for the second indices set when 4 of 5 items overlp
        with the first indices set.
        """
        _, second_set = self.nndvi._get_indices(self.nndvi.n_samples + 1)
        np.testing.assert_array_equal(second_set, np.arange(self.nndvi.n_samples) + 1)

    def test_no_overlap(self):
        """
        Test that there is no overlap in the indices when there is no overlap in the data.
        """
        first_set, second_set = self.nndvi._get_indices(self.nndvi.n_samples * 2)
        self.assertEqual(10, len(set(first_set).union(second_set)))

    def test_full_overlap(self):
        """
        Test that the first and second indices sets are equal when there is a full overlap in the
        data.
        """
        first_set, second_set = self.nndvi._get_indices(self.nndvi.n_samples)
        np.testing.assert_array_equal(first_set, second_set)

    def test_partial_overlap1(self):
        """
        Test that a partial overlap of 1 item yields the correct number of indices.
        """
        first_set, second_set = self.nndvi._get_indices(self.nndvi.n_samples + 4)
        self.assertEqual(9, len(set(first_set).union(second_set)))

    def test_partial_overlap2(self):
        """
        Test that a partial overlap of 4 items yields the correct number of indices.
        """
        first_set, second_set = self.nndvi._get_indices(self.nndvi.n_samples + 1)
        self.assertEqual(6, len(set(first_set).union(second_set)))


class CreateDataSetTest(unittest.TestCase):
    def test_no_shared_samples(self):
        nndvi = NNDVI(n_samples=5)
        nndvi.reference_window = np.arange(10).reshape((5, 2))
        nndvi.sliding_window = np.arange(10).reshape((5, 2)) + 10
        data = nndvi._create_data_set()
        self.assertEqual(10, len(data))
        for i, sample in enumerate(data):
            self.assertEqual(2 * i, sample[0])
            self.assertEqual(2 * i + 1, sample[1])

    def test_shared_samples(self):
        nndvi = NNDVI(n_samples=5)
        nndvi.reference_window = np.arange(10).reshape((5, 2))
        nndvi.sliding_window = np.arange(10).reshape((5, 2))
        data = nndvi._create_data_set()
        self.assertEqual(5, len(data))
        for i, sample in enumerate(data):
            self.assertEqual(2 * i, sample[0])
            self.assertEqual(2 * i + 1, sample[1])
