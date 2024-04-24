import os
from ast import literal_eval
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eval.crawler import ResultsCrawler


class Cleaner:
    """
    Cleaner parses all experiment data gathered in the provided read directory and splits each experiment's results into
    csv files containing only configurations with periodic drift detections, configurations without any detections and
    the remaining configurations and their respective metrics.
    """

    def __init__(
        self,
        read_root: str,
        write_repeats_root: str,
        write_no_detections_root: str,
        write_clean_root: str,
    ):
        """
        Init a new Cleaner instance.

        :param read_root: path to the directory containing the result data
        :param write_repeats_root: path of the directory into which data with repeat detections only will be written.
            Will be created if it does not exist.
        :param write_no_detections_root: path of the directory into which data without detections will be written.
            Will be created if it does not exist.
        :param write_clean_root: path of the directory into which the remaining data will be written.
            Will be created if it does not exist.
        """
        self.read_root = read_root
        self.write_repeats_root = write_repeats_root
        self.write_no_detections_root = write_no_detections_root
        self.write_clean_root = write_clean_root
        self.crawler = ResultsCrawler(read_root)

    def filter_results(self):
        """
        Filter the results given in read_root.

        :return: None
        """
        for file_ in self.crawler.crawl():
            if not ("Luxembourg" in file_ or "Ozone" in file_):
                print(f"Filtering {self.read_root}/{file_}")
                full_path = os.path.join(self.read_root, file_)
                df, drifts = self._read_df(full_path)
                repeat_detections = self._get_periodic_detection_indices(df, drifts)
                no_detections = self._get_no_detection_indices(df, drifts)
                full_filter_index = repeat_detections.join(no_detections, how="outer")
                self._save_df(
                    df.iloc[repeat_detections], self.write_repeats_root, file_
                )
                self._save_df(
                    df.iloc[no_detections], self.write_no_detections_root, file_
                )
                self._save_df(
                    df.drop(index=full_filter_index), self.write_clean_root, file_
                )

    def _get_periodic_detection_indices(
        self, df: pd.DataFrame, drifts: pd.Series
    ) -> pd.Series:
        """
        Get the indices of all runs with periodic detections.

        :param df: the data frame with the experiment's results
        :param drifts: the drifts
        :return: the indices of all runs with periodic detections
        """
        drift_delta = drifts.map(lambda x: np.array(x[:-1]) - np.array(x[1:]))
        delta_stds = drift_delta.map(self._drift_delta_std)
        return df[delta_stds == 0].index

    @staticmethod
    def _get_no_detection_indices(df: pd.DataFrame, drifts: pd.Series) -> pd.Series:
        """
        Get the indices of all runs without detections.

        :param df: the data frame with the experiment's results
        :param drifts: the drifts
        :return: the indices of all runs without detection
        """
        num_drifts = drifts.map(lambda x: len(x))
        return df[num_drifts == 0].index

    @staticmethod
    def _get_drifts(df: pd.DataFrame) -> pd.Series:
        """
        Get the drifts from the data frame by parsing the strings and converting them to arrays of detections.

        :param df: the data frame
        :return: a series of arrays of detected drifts
        """
        drifts = df["drifts"].map(lambda x: np.array(literal_eval(x)))
        return drifts

    @staticmethod
    def _drift_delta_std(drift_delta: np.array) -> Any:
        """
        Calculate standard deviation of the provided drift detection deltas.

        :param drift_delta: an array of deltas between drift detections
        :return: the standard deviation or nan
        """
        if len(drift_delta) > 1:
            return np.std(drift_delta)
        return np.nan

    @staticmethod
    def _read_df(file_path: str) -> (pd.DataFrame, pd.Series):
        """
        Read the data frame from the given path and parse the detected drifts.

        :param file_path: the path to the data
        :return: the data frame and a series of detected drifts
        """
        df = pd.read_csv(file_path)
        drifts = df["drifts"].map(lambda x: literal_eval(x))
        return df, drifts

    @staticmethod
    def _save_df(
        df: pd.DataFrame,
        write_root: str,
        file_path: str,
        save_empty_df: bool = False,
    ):
        """
        Save the given data frame to the given location consisting of the write_root and file_path. If the directories
        do not exist, they will be created.

        :param df: the data frame to save
        :param write_root: the root directory
        :param file_path: the sub path in the root directory
        :return: None
        """
        if save_empty_df or len(df) > 0:
            full_path = os.path.join(write_root, file_path)
            base_path = os.path.dirname(full_path)
            Path(base_path).mkdir(parents=True, exist_ok=True)
            df.to_csv(full_path, index=False)
