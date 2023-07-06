from os import path

from river.datasets import base
from river import stream


class SensorStream(base.FileDataset):
    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=2_219_803,
            n_features=5,
            task=base.MULTI_CLF,
            filename="sensorstream.csv",
        )
        self.full_path = path.join(directory_path, self.filename)

    def __iter__(self):
        return stream.iter_csv(
            self.full_path,
            target="class",
            converters={
                "rcdminutes": float,
                "temperature": float,
                "humidity": float,
                "light": float,
                "voltage": float,
                "class": int,
            }
        )
