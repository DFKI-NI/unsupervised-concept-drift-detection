from collections import deque
from typing import Iterable, Optional, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors

from .base import UnsupervisedDriftDetector


class NNDVI(UnsupervisedDriftDetector):
    """
    Nearest Neighbor-based Density Variation Identification (NN-DVI) detects concept drifts by determining the
    dissimilarity of particles across two data samples. Particles are constructed by determining the nearest neighbors
    of each data point in the respective sample and constructing an adjacent matrix. Afterwards, the dissimilarity
    between the particle matrices of the two samples is determined and a custom statistical test is conducted to
    determine if a concept drift occurred. The statistical test revolves around permuting the samples repeatedly to
    establish a baseline the current dissimilarity needs to exceed.

    TODO instance weights are unused right now
    TODO number of neighbors in kNN
    TODO examples show particle matrices with varying numbers of neighbors -- is this actually possible?

    Source: Liu, A.; Lu, J.; Liu, F.; Zhang, G. (2018). Accumulating regional density dissimilarity for concept drift
        detection in data streams. Pattern Recognition.
    """

    def __init__(
        self,
        n_samples: int = 100,
        k_neighbors: int = 30,
        n_permutations: int = 500,
        significance_level: float = 0.01,
        seed: Optional[int] = None,
    ):
        """
        Initialize a new NNDVI concept drift detector.

        :param n_samples: the size of the reference window and the sliding window
        :param k_neighbors: the number of neighbors used to construct particles
        :param n_permutations: the number of permutations used in the statistical test
        :param significance_level: the significance_level that determines the drift threshold
        """
        super().__init__(seed)
        self.n_samples = n_samples
        self.reference_window = deque(maxlen=self.n_samples)
        self.sliding_window = deque(maxlen=self.n_samples)
        self.k_neighbors = k_neighbors
        self.n_permutations = n_permutations
        self.significance_level = significance_level
        self.rng = np.random.default_rng(self.seed)

    def update(self, features) -> bool:
        """
        Update the drift detector with a new data and determine if a concept drift occured.

        :param features: the data
        :returns: True if a concept drift occurred, else False
        """
        features = np.fromiter(features.values(), dtype=float)
        self.sliding_window.append(features)
        if len(self.reference_window) < self.n_samples:
            self.reference_window.append(features)
        elif self._detect_drift():
            self.reference_window = self.sliding_window.copy()
            return True
        return False

    def _detect_drift(self) -> bool:
        """
        Detect whether a concept drift occurred.

        :returns: True if a concept drift occurred, else False
        """
        # data = self._create_data_set()
        data = np.concatenate((np.array(self.reference_window), np.array(self.sliding_window)))
        particle_matrix = self._get_particle_matrix(data)
        reference_indices, sliding_indices = self._get_indices(len(data))
        distance = self._get_nnps_distance(
            particle_matrix, reference_indices, sliding_indices
        )
        distances = []
        for _ in range(self.n_permutations):
            first_set, second_set = self._get_permutation(
                reference_indices, sliding_indices
            )
            distances.append(
                self._get_nnps_distance(particle_matrix, first_set, second_set)
            )
        threshold = norm.ppf(
            1 - self.significance_level, loc=np.mean(distances), scale=np.std(distances)
        )
        return distance > threshold

    def _create_data_set(self):
        data = np.zeros((self.n_samples * 2, len(self.reference_window[0])))
        data[:self.n_samples] = self.reference_window
        added_samples = 0
        for sample in self.sliding_window:
            if sample not in data:
                data[self.n_samples + added_samples] = sample
                added_samples += 1
        data = data[:self.n_samples + added_samples]
        return data

    def _get_indices(self, data_len: int) -> Tuple[np.array, np.array]:
        """
        Get the indices of the data contributed by the reference window and the sliding window.

        :param data_len: the length of the data set
        :returns: a tuple containing two numpy arrays of the respective indices
        """
        first_set = np.arange(self.n_samples)
        second_set = np.arange(self.n_samples) + data_len - self.n_samples
        return first_set, second_set

    def _get_permutation(
        self, first_set: np.array, second_set: np.array
    ) -> Tuple[np.array, np.array]:
        """
        Permute the given index arrays for a Monte Carlo permutation test.

        :param first_set: the index array of the reference window
        :param second_set: the index array of the sliding window
        :returns: a new tuple containing two numpy arrays with shuffled indices
        """
        indices = np.concatenate((first_set, second_set))
        self.rng.shuffle(indices)
        return indices[: self.n_samples], indices[self.n_samples:]

    def _get_particle_matrix(self, data) -> np.ndarray:
        """
        Compute a particle/adjacent matrix from the data set.

        :param data: the data set established from the reference window and the sliding window
        :returns: the particle/adjacent matrix
        """
        neighbors = NearestNeighbors(
            n_neighbors=self.k_neighbors
            + 1,  # add one because the datapoint itself is included in the subsequent fit
            algorithm="kd_tree",
        )
        neighbors.fit(data)
        _, indices = neighbors.kneighbors(data)
        particle_matrix = np.zeros((len(data), len(data)), dtype=int)
        for i, row in enumerate(indices):
            for column in row:
                particle_matrix[i][column] = 1
        # TODO add identity matrix?  --  probably not necessary since we already include the point itself in the knn
        # TODO weights
        return particle_matrix

    @staticmethod
    def _get_nnps_distance(
        particle_matrix: np.ndarray, first_set: Iterable, second_set: Iterable
    ) -> float:
        """
        Compute and get the distance between the reference window's particles and the sliding
        window's particles.

        :param particle_matrix: the particle matrix
        :param first_set: the indices of the reference window
        :param second_set: the indices of the sliding window
        :returns: the distance
        """
        # pylint: disable=R0201
        cardinalities = np.sum(particle_matrix, axis=1)
        lcm = np.lcm.reduce(cardinalities)
        weights = lcm / cardinalities
        particle_sets = (particle_matrix.T * weights).T
        first_particles = np.sum(particle_sets[first_set], axis=0)
        second_particles = np.sum(particle_sets[second_set], axis=0)
        distance = np.sum(
            np.abs(first_particles - second_particles)
            / (first_particles + second_particles)
        ) / (len(particle_matrix))
        return distance
