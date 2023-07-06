from collections import deque
from typing import Optional

import numpy as np
from scipy.spatial.distance import minkowski
from scipy.stats import beta
from sklearn.cluster import KMeans

from .base import UnsupervisedDriftDetector


class UCDD(UnsupervisedDriftDetector):
    """
    Unsupervised Concept Drift Detection (UCDD) detects concept drifts by determining the distribution of artificial
    labels across the reference data window and the recent data window. Assuming that both data windows stem from the
    same distribution, the distribution of the closest data point's artificial class labels should follow a Beta
    distribution. A concept drift is signalled, if the probability yielded by the cumulative density function is lower
    than a pre-determined threshold.

    Source: Shang, D.; Zhang, G.; Lu, J. (2020). Fast concept drift detection using unlabeled data. Developments of
        Artificial Intelligence Technologies in Computation and Robotics. World Scientific.
    """

    def __init__(
        self,
        n_reference_samples: int = 500,
        n_recent_samples: int = 500,
        threshold: float = 0.05,
        stability_offset: float = 1e-9,
        seed: Optional[int] = None,
    ):
        """
        Init a new UCDD instance.

        :param n_reference_samples: the number of samples stored in the reference data window
        :param n_recent_samples: the number of samples stored in the recent data window
        :param threshold: the threshold for concept drift detection
        """
        super().__init__(seed)
        self.window = deque(maxlen=n_recent_samples + n_reference_samples)
        self.n_reference_samples = n_reference_samples
        self.threshold = threshold
        self.stability_offset = stability_offset
        self.kmeans = KMeans(n_clusters=2, random_state=self.seed)

    def update(self, features: dict) -> bool:
        """
        Update the detector with the given features.

        :param features: the features
        :return: True if a drift occurred, else False
        """
        self.window.append(np.fromiter(features.values(), dtype=float))
        if len(self.window) == self.window.maxlen:
            kmeans = self.kmeans.fit(self.window)
            (
                reference_positive,
                reference_negative,
                recent_positive,
                recent_negative,
            ) = self._separate_data(kmeans.labels_)
            beta_positive = self._compute_beta(
                recent_positive, recent_negative, reference_negative
            )
            beta_negative = self._compute_beta(
                recent_negative, recent_positive, reference_positive
            )
            if beta_positive < self.threshold or beta_negative < self.threshold:
                return True
        return False

    def _compute_beta(
        self,
        evaluated_data: np.array,
        recent_data: np.array,
        reference_data: np.array,
    ) -> float:
        """
        Compute the probability for the distribution of the class labels. First, for each data point in the evaluated
        data the closest neighbor in the recent data and reference data respectively is determined. Then, the number
        of unique neighbors for both recent data and reference data is fed into the cumulative distribution function
        of the Beta distribution.

        :param evaluated_data: the evaluated data
        :param recent_data: the recent data which the evaluated data is compared to
        :param reference_data: the reference data which the evaluated data is compared to
        :return: the probability
        """
        windows = (set(), set())
        for d in evaluated_data:
            if len(recent_data) > 0:
                windows[0].add(self._find_closest_neighbor_index(d, recent_data))
            if len(reference_data) > 0:
                windows[1].add(self._find_closest_neighbor_index(d, reference_data))
        return beta.cdf(0.5, len(windows[0]) + self.stability_offset, len(windows[1]) + self.stability_offset)

    @staticmethod
    def _find_closest_neighbor_index(data_point: np.array, neighbor_candidates: np.array) -> int:
        """
        Find the closest neighbors index of the given data point in the provided neighbor candidates.

        :param data_point: the data point
        :param neighbor_candidates: the potential neighbors
        :return: the neighbor's index
        """
        distances = [
            minkowski(data_point, candidate) for candidate in neighbor_candidates
        ]
        return int(np.argmin(distances))

    def _separate_data(self, labels: np.array) -> (np.array, np.array, np.array, np.array):
        """
        Separate the reference data and recent data according to the artificial class labels.

        :param labels: the artificial class labels
        :return: first class reference data, second class reference data, first class recent data, second class recent
            data
        """
        data = np.array(self.window)
        reference_labels = labels[:self.n_reference_samples]
        recent_labels = labels[self.n_reference_samples:]
        reference_positive = data[:self.n_reference_samples][reference_labels == 0]
        reference_negative = data[:self.n_reference_samples][reference_labels == 1]
        recent_positive = data[self.n_reference_samples:][recent_labels == 0]
        recent_negative = data[self.n_reference_samples:][recent_labels == 1]
        return reference_positive, reference_negative, recent_positive, recent_negative
