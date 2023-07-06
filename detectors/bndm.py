from collections import deque

import numpy as np
from scipy import stats
from scipy.special import betaln

from .base import UnsupervisedDriftDetector


class BayesianNonparametricDetectionMethod(UnsupervisedDriftDetector):
    """
    Bayesian Nonparametric Drift Detection (BNDM) detects concept drifts by performing a Polya tree hypothesis test on
    each feature individually. The test determines the similarity of two samples by recursive comparisons of their
    distributions in binary splits. If the similarity is below a pre-determined threshold for any feature, a concept
    drift is detected.

    Source: Xuan, J.; Lu, J.; Zhang, G. (2020). Bayesian nonparametric unsupervised concept drift detection for data
        stream mining. ACM Transaction on Intelligent Systems and Technology.
    """

    def __init__(
        self,
        n_samples: int,
        const: float = 1.0,
        threshold: float = 0.5,
        max_depth: int = 3,
        seed=None,
    ):
        """
        Initialize a new BayesianNonparametricDetectionMethod.

        :param n_samples: the size of two samples
        :param const: the constant used to determine Polya tree test parameters
        :param threshold: the threshold of the drift detection
        :param max_depth: the max depth of the Polya tree
        """
        super().__init__(seed)
        self.n_samples = n_samples
        self.data_window = deque(maxlen=2 * n_samples)
        self.const = const
        self.threshold = threshold
        self.max_depth = max_depth
        self.distribution = stats.norm(loc=0, scale=1)

    def update(self, features: dict) -> bool:
        """
        Update the detector with the most recent observation and determine if a drift occurred.

        :param features: the features
        :returns: True if a drift was detected else False
        """
        features = np.fromiter(features.values(), dtype=float)
        self.data_window.append(features)
        if len(self.data_window) == self.data_window.maxlen:
            data = np.array(self.data_window)
            for i in range(len(features)):
                sample_one, sample_two = self._get_samples(feature_index=i)
                log_odd_ratios = self.polya_tree_test(sample_one, sample_two, 0)
                test_statistic = 1 / (1 + np.exp(-log_odd_ratios))
                if test_statistic < self.threshold:
                    self.reset()
                    return True
        return False

    def polya_tree_test(
        self,
        sample_one: np.array,
        sample_two: np.array,
        level: int,
        partition: str = "",
    ) -> float:
        """
        Perform a Polya tree two-sample test with the given samples by partitioning the data to estimate data
        distribution. As estimations are gathered over multiple levels in a Polya tree two-sample test, this method
        is called recursively on shrinking partitions until the configured maximum depth is reached.

        :param sample_one: the first sample
        :param sample_two: the second sample
        :param level: the current level in the Polya tree
        :param partition: the binary index of the partition
        :return: the test statistic that the hypothesis H0, sample_one == sample_two, is rejected
        """
        if level > self.max_depth:
            return 0
        partition_left = partition + "0"
        partition_right = partition + "1"
        n_one_left, n_two_left = self._get_interval_count(
            sample_one, sample_two, partition_left
        )
        n_one_right, n_two_right = self._get_interval_count(
            sample_one, sample_two, partition_right
        )
        n_left = n_one_left + n_two_left
        n_right = n_one_right + n_two_right
        if n_one_left + n_one_right == 0 or n_two_left + n_two_right == 0:
            return 0

        alpha = self.const * (level + 1) ** 2
        contribution_num = -betaln(alpha, alpha) + betaln(
            alpha + n_left, alpha + n_right
        )
        contribution_den = (
            -2 * betaln(alpha, alpha)
            + betaln(alpha + n_one_left, alpha + n_one_right)
            + betaln(alpha + n_two_left, alpha + n_two_right)
        )
        contribution = contribution_num - contribution_den
        contribution_left = self.polya_tree_test(
            sample_one, sample_two, level + 1, partition_left
        )
        contribution_right = self.polya_tree_test(
            sample_one, sample_two, level + 1, partition_right
        )
        return contribution + contribution_left + contribution_right

    def _get_interval_count(self, sample_one, sample_two, partition) -> (int, int):
        """
        Count the number of samples in the provided partition.

        :param sample_one: the first sample
        :param sample_two: the second sample
        :param partition: the partition
        :returns: a tuple containing the number of respective items of each sample in the interval
        """
        interval = self._get_interval(partition)
        n_one = np.sum((sample_one > interval[0]) & (sample_one <= interval[1]))
        n_two = np.sum((sample_two > interval[0]) & (sample_two <= interval[1]))
        return n_one, n_two

    def _get_interval(self, partition) -> (float, float):
        """
        Get the interval of the current partition.

        :param partition: the partition
        :returns: a tuple containing start and end of the interval
        """
        partition_index = int(partition, 2)
        level = len(partition)
        quantile_start = partition_index / 2 ** level
        quantile_end = (partition_index + 1) / 2 ** level
        interval = self.distribution.ppf([quantile_start, quantile_end])
        return interval

    def _get_samples(self, feature_index):
        """
        Get the normalized samples of the given index.

        :param feature_index: the index of the feature
        :return: a tuple containing two normalized samples
        """
        data = np.array(self.data_window)
        data_slice = data[:, feature_index]
        normalized_data_slice = self._normalize(data_slice)
        sample_one = normalized_data_slice[: self.n_samples]
        sample_two = normalized_data_slice[self.n_samples:]
        return sample_one, sample_two
    
    @staticmethod
    def _normalize(data):
        """
        Normalize the given data by subtracting the mean and dividing by the interquartile range. If the interquartile
        range is 0, no division occurs.

        :param data: the data
        :return: the normalized data
        """
        normalized = data - np.mean(data, axis=0)
        iqr = stats.iqr(data, axis=0)
        if iqr != 0:
            normalized = normalized / iqr
        return normalized

    def reset(self):
        """
        Reset the drift detector by deleting the reference data and recent data.
        """
        self.data_window = deque(maxlen=2 * self.n_samples)
