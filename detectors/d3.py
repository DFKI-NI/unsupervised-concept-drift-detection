from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .base import UnsupervisedDriftDetector


class DiscriminativeDriftDetector2019(UnsupervisedDriftDetector):
    """
    Discriminative Drift Detector (D3) detects concept drift by attempting to discern reference samples from recent
    samples. If the classifier can successfully discern the two data windows, there must be a difference between them
    implying the presence of concept drift.

    TODO difference to 2021
    TODO why wait before trying detection again?

    Source: Gözüaçık, Ö.; Büyükçakır, A.; Bonab, H.; Can, F. (2019). Unsupervised concept drift detection with a
        discriminative classifier. Proceedings of the 28th ACM International Conference on Information and Knowledge
        Management. ACM.
    """

    def __init__(
        self,
        n_reference_samples: int = 100,
        recent_samples_proportion: float = 0.1,
        threshold: float = 0.7,
        seed: Optional[int] = None,
    ):
        """
        Init new D3 instance.

        :param n_reference_samples: the number of data used to represent the current concept
        :param recent_samples_proportion: the proportion of data used to represent the new concept relative to the
            number of data used to represent current concept
        :param threshold: the threshold above which two concepts can be reliably discerned and a drift is signalled
        """
        super().__init__(seed)
        self.data = []
        self.n_reference_samples = n_reference_samples
        self.recent_samples_proportion = recent_samples_proportion
        self.n_samples = int(n_reference_samples * (1 + recent_samples_proportion))
        self.threshold = threshold
        self.kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=self.seed)

    def update(self, features: dict) -> bool:
        """
        Update the detector with the most recent observation and detect if a drift occurred.

        :param features: the features
        :returns: True if a drift occurred else False
        """
        features = np.fromiter(features.values(), dtype=float)
        if len(self.data) != self.n_samples:
            self.data.append(features)
        else:
            if self._detect_drift():
                self.data = self.data[self.n_reference_samples :]
                return True
            else:
                step = int(np.ceil(self.n_reference_samples * self.recent_samples_proportion))
                self.data = self.data[step:]
        return False

    def _detect_drift(self) -> bool:
        """
        Detect if a drift occurred.

        :return: True if a drift occurred, else False
        """
        labels = self._get_labels()
        discriminator = LogisticRegression(solver="liblinear", random_state=self.seed)
        predictions = self._predict(discriminator, np.array(self.data), labels)
        auc_score = roc_auc_score(labels, predictions)
        return auc_score >= self.threshold

    def _get_labels(self) -> np.array:
        """
        Get labels for the reference data window and the recent data window.

        :return: the labels
        """
        labels = np.zeros(self.n_samples)
        labels[self.n_reference_samples :] = 1
        return labels

    def _predict(self, discriminator, data: np.array, labels: np.array) -> np.array:
        """
        Train and test the discriminator on the given data in a kfold validation scheme.

        :param discriminator: the discriminator
        :param data: the data the discriminator is trained on
        :param labels: the labels of the data
        :return: the predictions
        """
        predictions = np.zeros(self.n_samples)
        # kfold testing is not described in the paper, but used in the source code provided by the authors
        for train_index, test_index in self.kfold.split(data, labels):
            discriminator.fit(data[train_index], labels[train_index])
            predictions[test_index] = discriminator.predict_proba(data[test_index])[
                :, 1
            ]
        return predictions
