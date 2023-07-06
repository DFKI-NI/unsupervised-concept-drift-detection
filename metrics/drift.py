from typing import Dict, List, Union
import numpy as np


def calculate_drift_metrics(
    known_drifts: List[Union[int, float]],
    detections: List[Union[int, float]],
) -> Dict[str, float]:
    """
    Calculate four drift metrics from known drifts and the detected drifts. The four metrics are the
    mean time between false alarms (mtfa), mean time to detection (mtd), missed detection ratio
    (mdr) and the mean time ratio (mtr). mtr combines the three former metrics into a single score.

    :param known_drifts: the known drifts
    :param detections: the detected drifts
    :returns: mtr, mtfa, mtd, mdr
    """
    detection_times = []
    false_detections = []

    was_drift = False
    previous_drift = 0

    unique_timesteps = sorted(set(known_drifts + detections))
    # while len(known_drifts) > 0 and len(detections) > 0:
    for timestep in unique_timesteps:
        if timestep in known_drifts:
            was_drift = True
            previous_drift = timestep
        # check without an else since a timestep might be in both known_drifts and detections
        if timestep in detections:
            if was_drift:
                detection_times.append(timestep - previous_drift)
            else:
                # previous has to be drift or None
                false_detections.append(timestep)
            was_drift = False

    mtfa = (
        np.mean(np.array(false_detections[1:]) - np.array(false_detections[:-1]))
        if len(false_detections) > 1
        else np.nan
    )
    mtd = np.mean(detection_times) if len(detection_times) > 0 else np.nan
    mdr = 1 - len(detection_times) / len(known_drifts)
    mtr = mtfa / mtd * (1 - mdr)

    return {"mtr": mtr, "mtfa": mtfa, "mtd": mtd, "mdr": mdr}
