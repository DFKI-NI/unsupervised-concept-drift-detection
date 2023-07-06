from os import path

from river.datasets import base
from river import stream


class Powersupply(base.FileDataset):
    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=29_928,
            n_features=2,
            task=base.MULTI_CLF,
            filename="powersupply.csv",
        )
        self.full_path = path.join(directory_path, self.filename)

    def __iter__(self):
        return stream.iter_csv(
            self.full_path,
            target="class",
            converters={
                "attribute0": float,
                "attribute1": float,
                "class": int,
            }
        )
