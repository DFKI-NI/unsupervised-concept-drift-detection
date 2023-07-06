import itertools
from collections import deque
from typing import Optional

import numpy as np

from .base import UnsupervisedDriftDetector


class ImageBasedDriftDetector(UnsupervisedDriftDetector):
    """
    Image-Based Drift Detector (IBBD) detects concept drifts by calculating the mean squared deviation of the reference
    data window and the recent data window. If the deviation exceeds thresholds, a drift is signalled. The thresholds
    are determined from the recent deviations and are updated regularly and everytime a drift is detected. Since the
    both deviations and the thresholds are calculated from the initial reference data, the reference data is never
    deleted.

    Source: Souza, V. M. A.; Parmezan, A. R. S.; Chowdhury, F. A.; Mueen, A. (2021). Efficient unsupervised drift
        detector for fast and high-dimensional data streams. Knowledge and Information Systems. Springer Link.
    """

    def __init__(
        self,
        n_samples: int = 300,
        n_consecutive_deviations: int = 1,
        n_permutations: int = 20,
        update_interval: int = 50,
        seed: Optional[int] = None,
    ):
        """
        Init a new IBDD instance.

        :param n_samples: the number of samples stored by both the reference data window and the recent data window
        :param n_consecutive_deviations: the number of consecutive values exceeding thresholds that must be detected
            before a concept drift is signalled
        :param n_permutations: the number of times the reference data is permuted to determine initial thresholds
        :param update_interval: the number of time steps between each update of the thresholds
        """
        super().__init__(seed)
        self.n_samples = n_samples
        self.reference_data = []
        self.recent_data = deque(maxlen=n_samples)
        self.recent_deviations = deque(maxlen=update_interval)
        self.n_consecutive_deviations = n_consecutive_deviations
        self.upper_threshold = None
        self.lower_threshold = None
        self.threshold_diffs = []
        self.update_interval = update_interval
        self.n_permutations = n_permutations
        self.time_step = 0
        self.last_threshold_update = 0
        self.rng = np.random.default_rng(self.seed)

    def update(self, features: dict) -> bool:
        """
        Update the detector with the given features.

        :param features: the features
        :return: True if a drift occurred, else False
        """
        features = np.fromiter(features.values(), dtype=float)
        drift = False
        if self.upper_threshold is None and self.lower_threshold is None:
            self.reference_data.append(features)
            if len(self.reference_data) == self.n_samples:
                self._calculate_initial_thresholds()
        self.recent_data.append(features)
        if (
            len(self.reference_data) == self.n_samples
            and len(self.recent_data) == self.n_samples
        ):
            deviation = self._calculate_mean_squared_deviation(
                np.array(self.recent_data)
            )
            self.recent_deviations.append(deviation)
            if self.time_step - self.last_threshold_update > self.update_interval:
                self._update_thresholds()

            drift = self._detect_drift(deviation)
        self.time_step += 1
        return drift

    def _detect_drift(self, deviation: float):
        """
        Detect if a concept drift occurred and update the upper and lower thresholds accordingly.

        :param deviation: the most recent mean squared deviation
        :return: True if a drift occurred, else False
        """
        evaluation_values = np.fromiter(
            itertools.islice(
                self.recent_deviations,
                len(self.recent_deviations) - (self.n_consecutive_deviations + 1),
                len(self.recent_deviations),
            ),
            dtype=float,
        )
        if np.all(evaluation_values >= self.upper_threshold):
            self.upper_threshold = deviation + np.std(self.recent_deviations)
            self.lower_threshold = deviation - np.mean(self.threshold_diffs)
            self.threshold_diffs.append(self.upper_threshold - self.lower_threshold)
            self.last_threshold_update = self.time_step
            return True
        elif np.all(evaluation_values <= self.lower_threshold):
            self.lower_threshold = deviation - np.std(self.recent_deviations)
            self.upper_threshold = deviation + np.mean(self.threshold_diffs)
            self.threshold_diffs.append(self.upper_threshold - self.lower_threshold)
            self.last_threshold_update = self.time_step
            return True
        return False

    def _calculate_mean_squared_deviation(self, other_data: np.array) -> float:
        """
        Calculate the mean squared deviation of the data stored in the reference window and the given other data.

        :param other_data: the data compared to the reference data
        :return: the mean squared deviation
        """
        reference_data = np.array(self.reference_data)
        if reference_data.shape != other_data.shape:
            raise ValueError(
                f"Shapes of compared data windows do not match: {reference_data.shape} != {other_data.shape}"
            )

        summands = (reference_data - other_data) ** 2
        return float(np.mean(summands))

    def _update_thresholds(self):
        """
        Update the upper and lower thresholds signalling the presence of a concept drift.
        """
        self.lower_threshold = np.mean(self.recent_deviations) - 2 * np.std(
            self.recent_deviations
        )
        self.upper_threshold = np.mean(self.recent_deviations) + 2 * np.std(
            self.recent_deviations
        )
        self.threshold_diffs.append(self.upper_threshold - self.lower_threshold)
        self.last_threshold_update = self.time_step

    def _calculate_initial_thresholds(self):
        """
        Calculate the initial upper and lower concept drift thresholds from expected deviations. The expected deviations
        are determined by evaluating permutations of the reference data.

        :raise: ValueError if either threshold is not None
        """
        if self.upper_threshold is not None or self.lower_threshold is not None:
            raise ValueError(
                "This method is intended for the calculation of initial thresholds only"
            )

        indices = np.arange(self.n_samples)
        for _ in range(self.n_permutations):
            self.rng.shuffle(indices)
            msd = self._calculate_mean_squared_deviation(
                np.array(self.reference_data)[indices]
            )
            self.recent_deviations.append(msd)
        deviations = np.fromiter(self.recent_deviations, dtype=float)
        self.lower_threshold = np.mean(deviations) - 2 * np.std(deviations)
        self.upper_threshold = np.mean(deviations) + 2 * np.std(deviations)
        self.threshold_diffs.append(self.upper_threshold - self.lower_threshold)
