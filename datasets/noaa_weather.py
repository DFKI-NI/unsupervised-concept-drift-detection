from os import path

from river.datasets import base
from river import stream


class NOAAWeather(base.FileDataset):
    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=18_159,
            n_features=8,
            task=base.MULTI_CLF,
            filename="NOAA.csv",
        )
        self.full_path = path.join(directory_path, self.filename)

    def __iter__(self):
        converters = {f"attribute{i}": float for i in range(1, 9)}
        converters["class"] = int
        return stream.iter_csv(
            self.full_path,
            target="class",
            converters=converters,
        )
