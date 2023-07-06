from os import path

from river.datasets import base
from river import stream


class OutdoorObjects(base.FileDataset):
    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=4_000,
            n_features=21,
            task=base.MULTI_CLF,
            filename="outdoor.csv",
        )
        self.full_path = path.join(directory_path, self.filename)

    def __iter__(self):
        converters = {f"att{i}": float for i in range(1, 22)}
        converters["class"] = int
        return stream.iter_csv(
            self.full_path,
            converters=converters,
            target="class",
        )
