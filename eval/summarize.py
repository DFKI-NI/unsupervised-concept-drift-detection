import os
from ast import literal_eval
from pathlib import Path

import pandas as pd

from eval.crawler import ResultsCrawler


class Summarizer:
    def __init__(
        self,
        read_root: str,
        write_root: str,
    ):
        self.read_root = read_root
        self.write_root = write_root
        self.crawler = ResultsCrawler(read_root)

    def summarize(self):
        for file_ in self.crawler.crawl():
            print(
                f"Writing summary of {self.read_root}/{file_} to {self.write_root}/{file_}"
            )
            writer = SummaryWriter(
                read_dir=self.read_root, write_dir=self.write_root, sub_path=file_
            )
            writer.summarize()


class SummaryWriter:
    def __init__(
        self,
        read_dir: str,
        write_dir: str,
        sub_path: str,
    ):
        self.base_metrics = [
            "lpd (ht)",
            "lpd (nb)",
            "acc (ht-no dd)",
            "acc (nb-no dd)",
            "acc (ht-dd)",
            "acc (nb-dd)",
            "f1 (ht-no dd)",
            "f1 (nb-no dd)",
            "f1 (ht-dd)",
            "f1 (nb-dd)",
            "mtr",
            "mtfa",
            "mtd",
            "mdr",
            "drifts",
        ]
        self.read_path = os.path.join(read_dir, sub_path)
        self.write_full_path = os.path.join(write_dir, sub_path)
        self.write_base_path = os.path.dirname(self.write_full_path)

    def summarize(self):
        df, cols = self.load_csv()
        summary = self.group_results(df, cols)
        if len(summary) > 0:
            Path(self.write_base_path).mkdir(parents=True, exist_ok=True)
            summary.to_csv(self.write_full_path)

    def load_csv(self):
        df = pd.read_csv(self.read_path)
        df.drop(columns=["seed"], inplace=True)
        cols = list(df.columns)
        for metric in self.base_metrics:
            if metric in cols:
                cols.remove(metric)
        df.sort_values(by=cols, inplace=True)
        df["drifts"] = df["drifts"].map(lambda drifts: len(literal_eval(drifts)))
        return df, cols

    def group_results(self, df: pd.DataFrame, config_columns):
        means = df.groupby(config_columns).aggregate("mean")
        means.columns = [f"{metric} (mean)" for metric in means.columns]
        stds = df.groupby(config_columns).aggregate("std")
        stds.columns = [f"{metric} (std)" for metric in stds.columns]
        summary = means.join(stds, how="outer")
        count = df.groupby(config_columns).aggregate("count")
        summary["count"] = count["lpd (ht)"]
        return summary
