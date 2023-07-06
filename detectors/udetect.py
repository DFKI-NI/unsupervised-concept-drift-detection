from collections import deque

import numpy as np

from .base import UnsupervisedDriftDetector


class UDetect(UnsupervisedDriftDetector):
    """
    Unsupervised Changed Detection for Activity Recognition (UDetect) detects concept drifts by calculating a statistic
    on the data window, which is the mean distance of each sample in the window to the mean of the window. If this
    statistic exceeds certain thresholds, a concept drift is signalled. The thresholds are determined on multiple
    windows of data through statistical process control.

    Source: Bashir, S.; Petrovski, A.; Doolan, D. (2017). A framework for unsupervised change detection in activity
        recognition. International Journal of Pervasive Computing and Communications.
    """

    def __init__(
        self,
        n_windows: int,
        n_samples: int = 200,
        disjoint_training_windows: bool = True,
        seed: int = None,
    ):
        """
        Init a new UDetect instance
        :param n_windows: the number of windows used to initialize the change thresholds
        :param n_samples: the number of samples per window
        """
        super().__init__(seed)
        self.n_windows = n_windows
        self.n_samples = n_samples
        self.data = deque(maxlen=n_samples)
        self.summaries = []
        self.disjoint_training_windows = disjoint_training_windows
        self.upper_range_limit = None
        self.upper_individual_limit = None
        self.lower_individual_limit = None

    def update(self, features: dict) -> bool:
        """
        Update the detector with the given features and detect if a concept drift occurred.

        :param features: the features
        :return: True if a drift occurred, else False
        """
        self.data.append(np.fromiter(features.values(), dtype=float))
        if len(self.data) == self.data.maxlen:
            if len(self.summaries) < self.n_windows:
                summary = self._calculate_window_summary()
                self.summaries.append(summary)
                if self.disjoint_training_windows:
                    self.data = deque(maxlen=self.n_samples)
            elif self.upper_range_limit is None:
                self._calculate_thresholds()
            else:
                if self._detect_drift():
                    self.reset()
                    return True
        return False

    def _detect_drift(self) -> bool:
        """
        Detect if a concept drift occurred.

        :return: True if a drift occurred, else False
        """
        summary = self._calculate_window_summary()
        if (
                summary < self.lower_individual_limit
                or summary > self.upper_individual_limit
                and summary > self.upper_range_limit
        ):
            return True
        return False

    def _calculate_window_summary(self) -> float:
        """
        Calculate the summary statistics for the current data window.

        :return: the summary
        """
        mean = np.mean(self.data, axis=0)
        data = np.array(self.data)
        distances = np.sum((data - mean) ** 2) ** 1/2  # minkowski_2
        summary = float(np.mean(distances))
        return summary

    def _calculate_thresholds(self):
        """
        Calculate the thresholds that indicate a concept drift occurred.
        """
        summaries = np.array(self.summaries)
        range_ = np.linalg.norm(summaries[1:] - summaries[:-1])
        mean_range = np.mean(range_)
        mean_summary = np.mean(summaries)
        self.upper_range_limit = 3.27 * mean_range
        self.upper_individual_limit = mean_summary + 2.66 * mean_range
        self.lower_individual_limit = mean_summary - 2.66 * mean_range

    def reset(self):
        self.upper_range_limit = None
        self.upper_individual_limit = None
        self.lower_individual_limit = None
        self.data = deque(maxlen=self.n_samples)
