from collections import deque
from typing import Optional

import numpy as np
from scipy.stats import anderson_ksamp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .base import UnsupervisedDriftDetector


class ClusteredStatisticalTestDriftDetectionMethod(UnsupervisedDriftDetector):
    """
    Clustered Statistical Test Drift Detection Method (CSDDM) detects concept drifts by applying the Anderson-Darling
    k-sample test on clustered data. The reference data is used to create clusters, which the incoming recent data is
    then assigned to. Furthermore, the reference data is also used to create a PCA projection to reduce the
    dimensionality of the feature space.

    Source: Wan, J. S.; Wang, S. (2021). Concept drift detection based on pre-clustering and statistical testing.
        Journal of Internet Technology.
    """

    def __init__(
        self,
        n_samples: int,
        n_clusters: int,
        confidence: float = 0.05,
        feature_proportion: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        Init a new CSDDM drift detector.

        :param n_samples: the number of samples stored by both the reference data window and the recent data window
        :param n_clusters: the number of clusters of the kmeans
        :param confidence: the required confidence to detect a concept drift
        :param feature_proportion: the proportion of features to keep after the PCA
        :param seed: the seed for the pseudo random number generator used in the PCA and the KMeans
        """
        super().__init__(seed)
        self.n_samples = n_samples
        self.reference_data = []
        self.reference_clusters = None
        self.recent_data = deque(maxlen=n_samples)
        self.recent_transformed_data = deque(maxlen=n_samples)
        self.n_clusters = n_clusters
        self.n_components = None
        self.kmeans = None
        self.feature_proportion = feature_proportion
        self.pca = None
        self.confidence = confidence
        self._confidence_index = self._get_confidence_index()

    def update(self, features: dict) -> bool:
        """
        Update the drift detector with the given features and determine if a concept drift occurred.

        :param features: the features
        :return: True if a concept drift occurred, else False
        """
        features = np.fromiter(features.values(), dtype=float)
        if self.kmeans is None:
            self.reference_data.append(features)
            if len(self.reference_data) == self.n_samples:
                self.setup()
        else:
            self.recent_data.append(features)
            padded_features = features.reshape((1, *features.shape))
            self.recent_transformed_data.append(
                self.pca.transform(padded_features).flatten()
            )
            if len(self.recent_data) == self.n_samples:
                if self._detect_drift():
                    self.reset()
                    return True
        return False

    def _detect_drift(self) -> bool:
        """
        Detect if a concept drift detected.

        :return: True if a concept drift occurred, else False
        """
        recent_clusters = self.kmeans.predict(self.recent_transformed_data)
        for i in range(self.n_clusters):
            reference_data_in_cluster = self.reference_data[
                self.reference_clusters == i
            ]
            recent_data_in_cluster = np.array(self.recent_transformed_data)[
                recent_clusters == i
            ]
            for feature in range(self.n_components):
                if (
                    len(reference_data_in_cluster) > 0
                    and len(recent_data_in_cluster) > 0
                    and (
                        len(reference_data_in_cluster) > 1
                        or len(recent_data_in_cluster) > 1
                    )
                ):
                    statistic, critical_values, _ = anderson_ksamp(
                        [
                            reference_data_in_cluster[:, feature],
                            recent_data_in_cluster[:, feature],
                        ]
                    )
                    if statistic >= critical_values[self._confidence_index]:
                        return True
        return False

    def _get_confidence_index(self):
        """
        Match the confidence level given in init to the corresponding index of the confidences returned by the
        anderson_ksamp function.

        :raise: ValueError if the requested confidence level is not supported
        :return: the index
        """
        confidences = [0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001]
        if self.confidence not in confidences:
            raise ValueError(
                f"Confidence level must be one of {confidences} but is {self.confidence}"
            )
        return confidences.index(self.confidence)

    def reset(self):
        """
        Reset the drift detector by deleting the reference data and creating new a PCA projection and KMeans clustering
        with the recent data.
        """
        self.reference_data = self.recent_data
        self.recent_data = deque(maxlen=self.n_samples)
        self.setup()

    def setup(self):
        """
        Create a PCA projection and KMeans clustering based on the reference data.
        """
        self.n_components = int(
            np.ceil(self.feature_proportion * len(self.reference_data[0]))
        )
        self.pca = PCA(n_components=self.n_components, random_state=self.seed)
        self.reference_data = self.pca.fit_transform(self.reference_data)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.seed)
        self.reference_clusters = self.kmeans.fit_predict(self.reference_data)
