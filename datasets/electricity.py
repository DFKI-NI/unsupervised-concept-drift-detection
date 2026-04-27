from os import path

from river.datasets import base
from river import stream


class Electricity(base.FileDataset):
    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=45_312,
            n_features=8,
            task=base.MULTI_CLF,
            filename="elec.arff",
        )
        self.full_path = path.join(directory_path, self.filename)

    def __iter__(self):
        return stream.iter_arff(
            self.full_path,
            target="class",
        )


class ElectricityNormalized(base.FileDataset):
    def __init__(self, directory_path: str = "datasets/files"):
        super().__init__(
            n_samples=45_312,
            n_features=8,
            task=base.MULTI_CLF,
            filename="electricity-normalized.csv",
        )
        self.full_path = path.join(directory_path, self.filename)

    def __iter__(self):
        return stream.iter_csv(
            self.full_path,
            target="class",
            converters={
                "date": float,
                "day": float,
                "period": float,
                "nswprice": float,
                "nswdemand": float,
                "vicprice": float,
                "vicdemand": float,
                "transfer": float,
                "class": str,
            },
        )
