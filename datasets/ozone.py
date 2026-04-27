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


class OzoneLevelDetection(base.FileDataset):
    """Ozone Level Detection dataset with WSR/T column names."""

    _FEATURE_COLS = (
        [f"WSR{i}" for i in range(24)] + ["WSR_PK", "WSR_AV"] +
        [f"T{i}" for i in range(24)] + ["T_PK", "T_AV"] +
        ["T85", "RH85", "U85", "V85", "HT85"] +
        ["T70", "RH70", "U70", "V70", "HT70"] +
        ["T50", "RH50", "U50", "V50", "HT50"] +
        ["KI", "TT", "SLP", "SLP_", "Precp"]
    )

    def __init__(self, directory_path: str = "datasets/files"):
        super().__init__(
            n_samples=5070,
            n_features=72,
            task=base.MULTI_CLF,
            filename="ozone_level_detection.csv",
        )
        self.full_path = path.join(directory_path, self.filename)

    def __iter__(self):
        import csv
        with open(self.full_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                x = {col: float(row[col]) if row[col] != '' else 0.0 for col in self._FEATURE_COLS}
                y = int(row["Class"])
                yield x, y
