from os import path

from river.datasets import base
from river import stream


class Luxembourg(base.FileDataset):
    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=1_901,
            n_features=31,
            task=base.MULTI_CLF,
            filename="luxembourg.csv",
        )
        self.full_path = path.join(directory_path, self.filename)

    def __iter__(self):
        return stream.iter_csv(
            self.full_path,
            target="class",
            converters={
                "att1": float,
                "att2": float,
                "att3": float,
                "att4": float,
                "att5": float,
                "att6": int,
                "att7": int,
                "att8": float,
                "att9": float,
                "att10": float,
                "att11": float,
                "att12": float,
                "att13": float,
                "att14": float,
                "att15": float,
                "att16": float,
                "att17": float,
                "att18": int,
                "att19": int,
                "att20": int,
                "att21": int,
                "att22": int,
                "att23": int,
                "att24": int,
                "att25": int,
                "att26": int,
                "att27": int,
                "att28": int,
                "att29": int,
                "att30": int,
                "att31": float,
                "class": int,
            }
        )
