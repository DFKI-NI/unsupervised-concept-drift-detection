from os import path

from river.datasets import base
from river import stream


class GasSensor(base.FileDataset):
    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=13_910,
            n_features=128,
            task=base.MULTI_CLF,
            filename="gassensor.csv",
        )
        self.full_path = path.join(directory_path, self.filename)

    def __iter__(self):
        converters = {f"V{i}": float for i in range(1, 129)}
        converters["Class"] = int
        return stream.iter_csv(
            self.full_path,
            target="Class",
            converters=converters,
        )
