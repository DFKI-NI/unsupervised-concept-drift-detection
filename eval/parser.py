import os
from abc import ABC
from pathlib import Path

import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

from .crawler import ResultsCrawler


class SummaryParser(ABC):
    def __init__(
            self,
            read_root: str,
            write_path: str,
    ):
        self.read_root = read_root
        self.write_path = write_path
        self.crawler = ResultsCrawler(self.read_root)

    def _read_csv(self, file_, metric):
        df = pd.read_csv(os.path.join(self.read_root, file_))
        drop_metrics = [column for column in df.columns if "(mean)" in column or "(std)" in column]
        drop_metrics.remove(f"{metric} (mean)")
        drop_metrics.remove(f"{metric} (std)")
        df.drop(columns=drop_metrics, inplace=True)
        df.sort_values(by=f"{metric} (mean)", inplace=True, ascending=False)
        return df

    @staticmethod
    def _order_columns(df, metric):
        cols = list(df.columns)
        cols.remove("detector")
        cols.remove(f"{metric} (mean)")
        cols.remove(f"{metric} (std)")
        cols = ["detector", f"{metric} (mean)", f"{metric} (std)"] + cols
        df = df.reindex(columns=cols)
        return df


class SummaryToStreamParser(SummaryParser):
    def get_top_n_configurations(self, n_configs: int = 5, metric: str = "acc (ht-dd)"):
        results = pd.DataFrame()
        detectors = []
        for file_ in self.crawler.crawl():
            df = self._read_csv(file_, metric)
            results = pd.concat((results, df[:n_configs]))
            detector = os.path.splitext(os.path.basename(file_))[0]
            detectors += [detector] * min(n_configs, len(df))
        results["detector"] = detectors
        results.sort_values(by=f"{metric} (mean)", inplace=True, ascending=False)
        results = self._order_columns(results, metric)
        Path(os.path.dirname(self.write_path)).mkdir(parents=True, exist_ok=True)
        results.to_csv(self.write_path, index=False)


class SummaryToDetectorParser(SummaryParser):
    def get_top_n_configurations(self, detector: str, n_configs: int = 5, metric: str = "acc (ht-dd)"):
        results = pd.DataFrame()
        streams = []
        for file_ in self.crawler.crawl():
            if detector not in file_:
                continue
            print(file_)
            df = self._read_csv(file_, metric)
            results = pd.concat((results, df[:n_configs]))
            stream = os.path.dirname(file_)
            streams += [stream] * min(n_configs, len(df))
        results["stream"] = streams
        Path(self.write_path).mkdir(parents=True, exist_ok=True)
        results.to_csv(f"{self.write_path}/{detector}_{metric[:3]}.csv", index=False)
        results.reset_index(drop=True, inplace=True)
        results.drop(columns=["count", "stream"], inplace=True)
        profile = ProfileReport(
            results,
            title=f"{detector}: {metric}",
            correlations={
                "auto": {"calculate": True},
                "pearson": {"calculate": True},
                "spearman": {"calculate": True},
                "kendall": {"calculate": True},
                "phi_k": {"calculate": True},
                "cramers": {"calculate": True},
            },
        )
        profile.to_file(f"{self.write_path}/{detector}_{metric[:3]}.html")


class SummariesToAverageParser(SummaryParser):
    def get_average_rank_per_config(self, detector: str, metric: str):
        print(f"Writing {metric} rank of {self.read_root}/**/{detector} to {self.write_path}")
        results = []
        streams = []
        for file_ in self.crawler.crawl():
            if "Luxembourg" in file_ or "Ozone" in file_:
                continue
            if detector not in file_:
                continue
            df = self._read_csv(file_, metric)
            df = df.drop(columns=[f"{metric} (mean)", f"{metric} (std)", "count"])
            stream = os.path.dirname(file_)
            df[stream] = np.arange(len(df)) + 1
            streams.append(stream)
            results.append(df)
        merged_df = results[0]
        for result in results[1:]:
            merged_df = pd.merge(merged_df, result, how="outer")
        means = merged_df.apply(lambda row: np.mean(row[streams]), axis=1)
        merged_df["mean rank"] = means
        merged_df["mean % rank"] = means / len(merged_df)
        Path(self.write_path).mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(f"{self.write_path}/{detector}_{metric[:3]}_rank.csv", index=False)
        counts = merged_df.apply(lambda x: x.isna().values.sum(), axis=1)
        counts = counts.value_counts()
        print("configurations that yielded no results n times:")
        print(counts)
        print(counts/len(merged_df))
        return dict(counts/len(merged_df))
