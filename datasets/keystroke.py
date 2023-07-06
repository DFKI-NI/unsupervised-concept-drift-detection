from os import path

from river.datasets import base
from river import stream


class Keystroke(base.FileDataset):
    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=1_600,
            n_features=10,
            task=base.MULTI_CLF,
            filename="keystroke.arff",
        )
        self.full_path = path.join(directory_path, self.filename)

    def __iter__(self):
        return stream.iter_arff(
            self.full_path,
            target="class",
        )