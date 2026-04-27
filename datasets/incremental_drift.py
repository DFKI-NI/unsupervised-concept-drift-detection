from os import path

from river.datasets import base
from river import stream


class IncrementalDrift(base.FileDataset): #TODO : Test class
    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=10000,
            n_features=10,
            task=base.MULTI_CLF,
            filename="incremental_drift_concept_id.csv",
        )
        self.full_path = path.join(directory_path, self.filename)

    def __iter__(self): #TODO : call sample -> window
        for x, y in stream.iter_csv(
            self.full_path,
            target="label",
        ):
            # Convert string features to float
            x = {k: float(v) for k, v in x.items()}
            yield x, int(y)
