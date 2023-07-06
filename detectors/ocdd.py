from collections import deque
from typing import Optional

import numpy as np
from sklearn.svm import OneClassSVM

from .base import UnsupervisedDriftDetector


class OneClassDriftDetector(UnsupervisedDriftDetector):
    """
    One-Class Drift Detector (OCDD) detects concept drifts by monitoring an outlier detector. If the rate of recent
    outliers exceeds a pre-determined threshold, a drift is signalled. Furthermore, the outlier detector is re-fitted
    on the most recent data.

    Source: Gözüaçık, Ö.; Can, F. (2020). Concept learning using one-class classifiers for implicit drift detection in
        evolving data streams. Artificial Intelligence Review. Springer Link.
    """

    def __init__(
        self,
        n_samples: int = 100,
        threshold: float = 0.3,
        outlier_detector_class: callable = OneClassSVM,
        outlier_detector_kwargs: dict = None,
        seed: Optional[int] = None,
    ):
        """
        Init a new OCDD instance.

        :param n_samples: the number of data used to monitor the outlier detection rate
        :param threshold: the ratio of outliers among the recent n_samples data considered normal
        :param outlier_detector_class: the init method of an outlier detector
        :param outlier_detector_kwargs: the key word arguments used to initialize the outlier detector
        """
        super().__init__(seed)
        if outlier_detector_kwargs is None:
            outlier_detector_kwargs = {}
        self.n_samples = n_samples
        self.data = deque(maxlen=n_samples)
        self.outliers = deque(maxlen=n_samples)
        self.threshold = threshold
        self.outlier_detector = None
        self.outlier_detector_class = outlier_detector_class
        self.outlier_detector_kwargs = outlier_detector_kwargs

    def update(self, features: dict) -> bool:
        """
        Update the detector with the given features.

        :param features: the features
        :return: True if a drift occurred, else False
        """
        features = np.fromiter(features.values(), dtype=float)
        self.data.append(features)
        if len(self.data) == self.n_samples and self.outlier_detector is None:
            self.setup()
        if self.outlier_detector is not None:
            outlier = self.outlier_detector.predict([features])
            self.outliers.append(outlier)
            if len(self.data) == self.n_samples and self._detect_drift():
                self.reset()
                return True
        return False

    def _detect_drift(self):
        """
        Detect if a concept drift occurred.

        :return: True if a drift occurred, else False
        """
        outlier_rate = np.mean(np.array(self.outliers) == -1)
        return outlier_rate >= self.threshold

    def reset(self):
        """
        Drop the oldest samples and retrain the outlier detector.
        """
        n_dropped = int(self.n_samples * (1 - self.threshold))
        for _ in range(n_dropped):
            self.data.pop()
        self.setup()

    def setup(self):
        """
        Init a new outlier detector and train it on data.
        """
        try:
            self.outlier_detector = self.outlier_detector_class(
                random_state=self.seed,
                **self.outlier_detector_kwargs
            )
        except TypeError:
            self.outlier_detector = self.outlier_detector_class(
                **self.outlier_detector_kwargs
            )
        self.outlier_detector.fit(self.data)
        self.data = deque(maxlen=self.n_samples)
