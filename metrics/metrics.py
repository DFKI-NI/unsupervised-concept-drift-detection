from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .drift import calculate_drift_metrics
from .lift_per_drift import lift_per_drift


@dataclass
class ExperimentResult:
    """
    This data class stores the following metrics recorded during experiments:
    - accuracy of Hoeffding tree and naive Bayes classifiers with and without the use of a concept drift detector
    - f1 scores of Hoeffding tree and naive Bayes classifiers with and without the use of a concept drift detector
    - lpd of a Hoeffding tree and a naive Bayes classifier
    - mtfa
    - mtr
    - mtd
    - mdr
    """
    accuracies: List[float]
    f1_scores: List[float]
    lpd: (float, float)
    mtfa: float = None
    mtr: float = None
    mtd: float = None
    mdr: float = None

    def to_dict(self, include_drift_metrics: bool) -> dict:
        """
        Convert the stored data to a dictionary.

        :param include_drift_metrics: True if drift metrics (mtr, mtfa, mtd and mdr) shall be included in the dict
        :return: the dict
        """
        results = {
            "lpd (ht)": self.lpd[0],
            "lpd (nb)": self.lpd[1],
            "acc (ht-no dd)": self.accuracies[0],
            "acc (nb-no dd)": self.accuracies[1],
            "acc (ht-dd)": self.accuracies[2],
            "acc (nb-dd)": self.accuracies[3],
            "f1 (ht-no dd)": self.f1_scores[0],
            "f1 (nb-no dd)": self.f1_scores[1],
            "f1 (ht-dd)": self.f1_scores[2],
            "f1 (nb-dd)": self.f1_scores[3],
        }
        if include_drift_metrics:
            results["mtfa"] = self.mtfa
            results["mtr"] = self.mtr
            results["mtd"] = self.mtd
            results["mdr"] = self.mdr
        return results


def get_metrics(stream, predicted_drifts, true_labels, predicted_labels) -> ExperimentResult:
    """
    Calculate performance metrics based on the predicted drifts, the predicted labels and the true labels to calculate
    accuracies, f1 scores and lift-per-drift. If stream contains ground truth concept drift, mtr, mtfa, mtd and mdr are
    calculated as well.

    :param stream: the data stream the experiment was conducted on
    :param predicted_drifts: the positions of detected drifts
    :param true_labels: the true class labels
    :param predicted_labels: the predicted class labels
    :return: an ExperimentResult data class storing the corresponding metrics
    """
    if hasattr(stream, "drifts"):
        drift_metrics = calculate_drift_metrics(stream.drifts, predicted_drifts)
    else:
        drift_metrics = {"mtfa": None, "mdr": None, "mtr": None, "mtd": None}
    f1_scores = []
    accuracies = []
    predicted_labels = np.array(predicted_labels).transpose()
    for predictions in predicted_labels:
        f1_scores.append(f1_score(y_true=true_labels, y_pred=predictions, average="macro"))
        accuracies.append(accuracy_score(y_true=true_labels, y_pred=predictions))
    lpd_hoeffding_tree = lift_per_drift(
        base_accuracy=accuracies[0],
        assisted_accuracy=accuracies[2],
        n_drifts=len(predicted_drifts),
    )
    lpd_gaussian_nb = lift_per_drift(
        base_accuracy=accuracies[1],
        assisted_accuracy=accuracies[3],
        n_drifts=len(predicted_drifts),
    )
    metrics = ExperimentResult(
        accuracies=accuracies,
        f1_scores=f1_scores,
        lpd=(lpd_hoeffding_tree, lpd_gaussian_nb),
        **drift_metrics
    )
    return metrics
