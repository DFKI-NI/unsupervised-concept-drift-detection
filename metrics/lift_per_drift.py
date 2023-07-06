def lift_per_drift(
    base_accuracy: float, assisted_accuracy: float, n_drifts: int, cost_ratio: float = 1
) -> float:
    """

    :param base_accuracy: the accuracy of a detector deployed without concept drift detection
    :param assisted_accuracy: the accuracy of a detector supported by a concept drift detector
    :param n_drifts: the number of drifts detected by the concept drift detector
    :param cost_ratio: the penalization of concept drifts 0 < cost_ratio <= 1 with low values corresponding to a weak
        penalty and high values corresponding to a high penalty. The larger the penalty, the smaller the metric.
        Defaults to 1
    :return: the lift-per-drift
    """
    if n_drifts == 0:
        return 0
    if cost_ratio == 1:
        return (assisted_accuracy - base_accuracy) / n_drifts
    else:
        accuracy_diff = assisted_accuracy - base_accuracy
        penalty = 1 - cost_ratio
        return accuracy_diff * penalty / (1 - cost_ratio**n_drifts)
