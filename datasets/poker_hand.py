from os import path

from river.datasets import base
from river import stream


class PokerHand(base.FileDataset):
    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=829_201,
            n_features=10,
            task=base.MULTI_CLF,
            filename="poker-lsn.csv",
        )
        self.full_path = path.join(directory_path, self.filename)

    def __iter__(self):
        return stream.iter_csv(
            self.full_path,
            target="class",
            converters={
                "s1": int,
                "r1": float,
                "s2": int,
                "r2": float,
                "s3": int,
                "r3": float,
                "s4": int,
                "r4": float,
                "s5": int,
                "r5": float,
                "class": int,
            }
        )
