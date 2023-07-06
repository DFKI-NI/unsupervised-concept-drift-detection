from enum import Enum, auto
from typing import Dict, List, Optional

import numpy as np

from .base import UnsupervisedDriftDetector
from .ks import KolmogorovSmirnovDriftDetector


class EDFSMode(Enum):
    RANDOM = auto()
    SUBSPACE_SELECTION = auto()


class EDFS(UnsupervisedDriftDetector):
    """
    Ensemble Drift Detection with Feature Subspaces (EDFS) detects concept drifts through an ensemble of univariate
    concept drift detector ensembles. Each univariate ensemble is deployed on a different feature subspace. A concept
    drift in a feature subspace is detected via majority vote. Concept drift is detected on a global scale if a single
    subspace drifts.

    Currently, only random feature subspace selection is supported.

    Source: Korycki, L.; Krawczyk, B. (2019). Unsupervised drift detector ensembles for data stream mining. IEEE
        International Conference on Data Science and Advanced Analytics. IEEE Xplore.
    """

    def __init__(
        self,
        n_subspaces: int = 10,
        feature_percentage: float = 0.1,
        mode: EDFSMode = EDFSMode.RANDOM,
        alpha: float = 0.005,
        window_size: int = 100,
        seed: Optional[int] = None,
    ):
        """
        Init a new EDFS instance.

        :param n_subspaces: the number of subspaces
        :param feature_percentage: the percentage of features used in each of the subspaces
        :param mode: the mode used in subspace creation, defaults to random
        :param alpha: the probability of the test statistic of the Kolmogorov-Smirnov-test
        :param window_size: the size of the sliding window
        """
        super().__init__(seed)
        self.mode = mode
        self.n_subspaces = n_subspaces
        self.feature_percentage = feature_percentage
        self.n_features = 0
        self.n_features_per_space = 0
        self.subspaces: List[Dict[str, KolmogorovSmirnovDriftDetector]] = []
        self.alpha = alpha
        self.window_size = window_size
        self.rng = np.random.default_rng(self.seed)

    def update(self, features: dict) -> bool:
        """
        Update the detector with the given features.

        :param features: the features
        :return: True if a drift occurred, else False
        """
        if len(self.subspaces) == 0:
            self.reset(features)
        drift = self._detect_drift(features)
        if drift:
            self.reset(features)
        return drift

    def _detect_drift(self, features: dict) -> bool:
        """
        Detect if a concept drift occurred.

        :return: True if a drift occurred, else False
        """
        for subspace in self.subspaces:
            drifts = []
            for feature, detector in subspace.items():
                drift = detector.update(features[feature])
                drifts.append(drift)
            if sum(drifts) > len(features) * self.feature_percentage / 2:
                # majority vote
                return True
        return False

    def reset(self, sample):
        """
        Resets the feature subspaces using the method specified at initialization.
        """
        self.n_features = len(sample)
        self.n_features_per_space = int(np.ceil(self.n_features * self.feature_percentage))
        if self.mode is EDFSMode.RANDOM:
            self.__random_reset(sample)
        elif self.mode is EDFSMode.SUBSPACE_SELECTION:
            self.__subspace_selection_reset()
        else:
            raise ValueError(f"Mode {self.mode} is not a mode supported by EDFS.")

    def __random_reset(self, sample):
        """
        Resets by choosing feature subspaces randomly.
        """
        self.subspaces = [
            {
                feature: KolmogorovSmirnovDriftDetector(self.window_size, self.alpha)
                for feature in self.rng.choice(list(sample.keys()), size=self.n_features_per_space, replace=False)
            }
            for _ in range(self.n_subspaces)
        ]

    def __subspace_selection_reset(self):
        raise NotImplementedError(
            "Feature subspace selection was not implemented, yet."
        )
