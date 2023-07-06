import numpy as np


def get_simple_stream_drifts(detector):
    drifts = []
    for i in range(50):
        drifts.append(
            detector.update({"0": 0 + (i % 2) / 10000, "1": 1 + (i % 2) / 10000})
        )
    for i in range(50):
        drifts.append(
            detector.update({"0": 100 + (i % 2) / 10000, "1": 101 + (i % 2) / 10000})
        )
    return drifts


def get_simple_random_stream_drifts(detector, seed=None):
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)
    drifts = []
    for i in range(50):
        drifts.append(detector.update({"0": rng.random(), "1": rng.random()}))
    for i in range(50):
        drifts.append(detector.update({"0": rng.random() + 1, "1": rng.random() + 1}))
    return drifts
