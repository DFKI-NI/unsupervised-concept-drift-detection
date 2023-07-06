"""This module provides the INSECTS drifting data streams."""
from os import path

from river import stream
from river.datasets import base


class Insects(base.FileDataset):
    def __init__(self, **desc):
        super().__init__(**desc)
        self.full_path = ""

    def __iter__(self):
        converters = {f"Att{i}": float for i in range(1, 34)}
        converters["class"] = str
        return stream.iter_csv(
            self.full_path,
            target="class",
            converters=converters,
        )


class InsectsAbruptBalanced(Insects):
    """
    This class provides the abrupt balanced INSECTS datastream.

    References
    ----------
    [^1]: [Challenges in benchmark stream learning algorithms with real-world
    data](https://doi.org/10.1007/s10618-020-00698-5)
    """

    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=52848,
            n_features=33,
            task=base.MULTI_CLF,
            filename="INSECTS-abrupt_balanced_norm.csv",
        )
        self.drifts = [14_352, 19_500, 33_240, 38_682, 39_510]
        self.full_path = path.join(directory_path, self.filename)


class InsectsAbruptImbalanced(Insects):
    """
    This class provides the abrupt imbalanced INSECTS datastream.

    References
    ----------
    [^1]: [Challenges in benchmark stream learning algorithms with real-world
    data](https://doi.org/10.1007/s10618-020-00698-5)
    """

    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=355275,
            n_features=33,
            task=base.MULTI_CLF,
            filename="INSECTS-abrupt_imbalanced_norm.csv",
        )
        self.drifts = [83_859, 128_651, 182_320, 242_883, 268_380]
        self.full_path = path.join(directory_path, self.filename)


class InsectsIncrementalBalanced(Insects):
    """
    This class provides the incremental balanced INSECTS datastream.

    References
    ----------
    [^1]: [Challenges in benchmark stream learning algorithms with real-world
    data](https://doi.org/10.1007/s10618-020-00698-5)
    """

    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=57018,
            n_features=33,
            task=base.MULTI_CLF,
            filename="INSECTS-incremental_balanced_norm.csv",
        )
        self.full_path = path.join(directory_path, self.filename)


class InsectsIncrementalImbalanced(Insects):
    """
    This class provides the incremental imbalanced INSECTS datastream.

    References
    ----------
    [^1]: [Challenges in benchmark stream learning algorithms with real-world
    data](https://doi.org/10.1007/s10618-020-00698-5)
    """

    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=452044,
            n_features=33,
            task=base.MULTI_CLF,
            filename="INSECTS-incremental_imbalanced_norm.csv",
        )
        self.full_path = path.join(directory_path, self.filename)


class InsectsGradualBalanced(Insects):
    """
    This class provides the gradual balanced INSECTS datastream.

    References
    ----------
    [^1]: [Challenges in benchmark stream learning algorithms with real-world
    data](https://doi.org/10.1007/s10618-020-00698-5)
    """

    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=24150,
            n_features=33,
            task=base.MULTI_CLF,
            filename="INSECTS-gradual_balanced_norm.csv",
        )
        self.drifts = [14_028]
        self.full_path = path.join(directory_path, self.filename)


class InsectsGradualImbalanced(Insects):
    """
    This class provides the gradual imbalanced INSECTS datastream.

    References
    ----------
    [^1]: [Challenges in benchmark stream learning algorithms with real-world
    data](https://doi.org/10.1007/s10618-020-00698-5)
    """

    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=143323,
            n_features=33,
            task=base.MULTI_CLF,
            filename="INSECTS-gradual_imbalanced_norm.csv",
        )
        self.drifts = [58_159]
        self.full_path = path.join(directory_path, self.filename)


class InsectsIncrementalAbruptBalanced(Insects):
    """
    This class provides the incremental-abrupt balanced INSECTS datastream.

    References
    ----------
    [^1]: [Challenges in benchmark stream learning algorithms with real-world
    data](https://doi.org/10.1007/s10618-020-00698-5)
    """

    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=79986,
            n_features=33,
            task=base.MULTI_CLF,
            filename="INSECTS-incremental-abrupt_balanced_norm.csv",
        )
        self.drifts = [26_568, 53_364]
        self.full_path = path.join(directory_path, self.filename)


class InsectsIncrementalAbruptImbalanced(Insects):
    """
    This class provides the incremental-abrupt imbalanced INSECTS datastream.

    References
    ----------
    [^1]: [Challenges in benchmark stream learning algorithms with real-world
    data](https://doi.org/10.1007/s10618-020-00698-5)
    """

    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=452044,
            n_features=33,
            task=base.MULTI_CLF,
            filename="INSECTS-incremental-abrupt_imbalanced_norm.csv",
        )
        self.drifts = [150_683, 301_365]
        self.full_path = path.join(directory_path, self.filename)


class InsectsIncrementalReoccurringBalanced(Insects):
    """
    This class provides the incremental-reoccurring balanced INSECTS datastream.

    References
    ----------
    [^1]: [Challenges in benchmark stream learning algorithms with real-world
    data](https://doi.org/10.1007/s10618-020-00698-5)
    """

    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=79986,
            n_features=33,
            task=base.MULTI_CLF,
            filename="INSECTS-incremental-reoccurring_balanced_norm.csv",
        )
        self.drifts = [26_568, 53_364]
        self.full_path = path.join(directory_path, self.filename)


class InsectsIncrementalReoccurringImbalanced(Insects):
    """
    This class provides the incremental-reoccurring imbalanced INSECTS datastream.

    References
    ----------
    [^1]: [Challenges in benchmark stream learning algorithms with real-world
    data](https://doi.org/10.1007/s10618-020-00698-5)
    """

    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=452044,
            n_features=33,
            task=base.MULTI_CLF,
            filename="INSECTS-incremental-reoccurring_imbalanced_norm.csv",
        )
        self.drifts = [150_683, 301_365]
        self.full_path = path.join(directory_path, self.filename)
