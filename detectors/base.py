import time
from abc import ABC, abstractmethod
from typing import Optional


class SupervisedDriftDetector(ABC):
    """
    This abstract base class provides a consistent interface for all supervised concept drift detectors.
    """

    @abstractmethod
    def update(
        self,
        y_true: int,
        y_pred: int,
    ) -> bool:
        raise NotImplementedError("This abstract base class does not implement update.")


class UnsupervisedDriftDetector(ABC):
    """
    This abstract base class provides a consistent interface for all unsupervised concept drift detectors.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is None:
            seed = int(time.time())
        self.seed = seed

    @abstractmethod
    def update(
        self,
        features: dict,
    ) -> bool:
        raise NotImplementedError("This abstract base class does not implement update.")
