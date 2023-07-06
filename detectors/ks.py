from collections import deque

from scipy.stats import ks_2samp

from .base import UnsupervisedDriftDetector


class KolmogorovSmirnovDriftDetector(UnsupervisedDriftDetector):
    """
    A simple unsupervised univariate drift detector based on the Kolmogorov-Smirnov two sample test.
    """

    def __init__(
        self, window_size: int, threshold: float, reset_after_drift: bool = False
    ):
        """
        Init a new KolmogorovSmirnovDriftDetector.

        :param window_size: the size of the reference data window and the recent data window
        :param threshold: the threshold
        :param reset_after_drift: True if the recent data window shall be purged after a drift, default False
        """
        super().__init__()
        self.window_size = window_size
        self.recent_data = deque(maxlen=window_size)
        self.reference_data = deque(maxlen=window_size)
        self.threshold = threshold
        self.reset_after_drift = reset_after_drift

    def update(self, feature: float) -> bool:
        """
        Update the detector with the most recent feature.

        :param feature: the feature
        """
        if len(self.recent_data) == self.window_size:
            self.reference_data.append(self.recent_data[0])
        self.recent_data.append(feature)
        if len(self.reference_data) == self.window_size:
            statistic, p_value = ks_2samp(self.reference_data, self.recent_data)
            if p_value < self.threshold and statistic > 0.1:
                if self.reset_after_drift:
                    self.reset()
                return True
        return False

    def reset(self):
        """
        Reset the reference data window and recent data window.
        """
        self.reference_data = self.recent_data
        self.recent_data = deque(maxlen=self.window_size)
