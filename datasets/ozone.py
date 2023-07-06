from os import path

from river.datasets import base
from river import stream


class Ozone(base.FileDataset):
    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=2_534,
            n_features=72,
            task=base.MULTI_CLF,
            filename="ozone.csv",
        )
        self.full_path = path.join(directory_path, self.filename)

    def __iter__(self):
        converters = {f"V{i}": float for i in range(1, 73)}
        converters["Class"] = int
        return stream.iter_csv(
            self.full_path,
            target="Class",
            converters=converters,
        )
