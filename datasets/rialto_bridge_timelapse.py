from os import path

from river.datasets import base
from river import stream


class RialtoBridgeTimelapse(base.FileDataset):
    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=82_250,
            n_features=27,
            task=base.MULTI_CLF,
            filename="rialto.csv",
        )
        self.full_path = path.join(directory_path, self.filename)

    def __iter__(self):
        converters = {f"att{i}": float for i in range(1, 28)}
        converters["class"] = int
        return stream.iter_csv(
            self.full_path,
            target="class",
            converters=converters,
        )