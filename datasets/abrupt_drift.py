from os import path

from river.datasets import base
from river import stream


class AbruptDrift(base.FileDataset):
    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=10000,  
            n_features=10,
            task=base.MULTI_CLF,
            filename="abrupt_drift_concept_id.csv",
        )
        self.full_path = path.join(directory_path, self.filename)

    def __iter__(self):
        for x, y in stream.iter_csv(
            self.full_path,
            target="label",
        ):
            # Convert string features to float
            x = {k: float(v) for k, v in x.items()}
            yield x, int(y)
