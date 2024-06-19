import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import linregress

from .crawler import ResultsCrawler


class SummaryPlotter:
    def __init__(
        self,
        read_root: str,
        file: str,
        write_root: str = None,
    ):
        self.read_path = os.path.join(read_root, file)
        self.read_root = read_root
        self.file = file
        self.write_root = write_root
        self.crawler = ResultsCrawler(self.read_path)
        self.colors = {
            "bndm": "#332288",
            "csddm": "#117733",
            "d3": "#44AA99",
            "ibdd": "#DDCC77",
            "ocdd": "#CC6677",
            "spll": "#AA4499",
            "udetect": "#882255",
        }
        self.markers = {
            "bndm": "o",
            "csddm": "s",
            "d3": "*",
            "ibdd": "P",
            "ocdd": "X",
            "spll": "v",
            "udetect": "^",
        }
        # matplotlib.use("QtAgg")

    def plot_boxes_for_samples(self, metric="lpd (ht) (mean)"):
        data = []
        labels = []
        for file_ in self.crawler.crawl():
            df = pd.read_csv(os.path.join(self.read_path, file_))
            if "d3" not in file_:
                samples = sorted(df["n_samples"].unique())
                for sample in samples:
                    data.append(df[df["n_samples"] == sample][metric].values)
                    labels.append(f"{file_}/{sample}")
            else:
                samples = df["n_reference_samples"].unique()
                for sample in samples:
                    data.append(df[df["n_reference_samples"] == sample][metric].values)
                    labels.append(f"{file_}/{sample}")
        plt.boxplot(x=data, labels=labels)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{self.read_path}")
        plt.ylabel(metric)
        plt.xlabel("Algorithm/n_samples")
        plt.tight_layout()
        plt.show()

    def plot_top_metric_boxes(
        self, metric="lpd (ht) (mean)", top_n: int = 10000, show: bool = False
    ):
        data = []
        labels = []
        plt.rcParams.update({"font.size": 12})
        for file_ in self.crawler.crawl():
            if "Luxembourg" in file_ or "Ozone" in file_:
                continue
            df = pd.read_csv(os.path.join(self.read_path, file_))
            df = df.filter(items=[metric])
            df.sort_values(by=f"{metric}", inplace=True, ascending=False)
            file_data = df[:top_n][metric].values
            data.append(file_data[~np.isnan(file_data)])
            labels.append(f"{file_[:-4]}")
        plt.boxplot(x=data, labels=labels)
        # colors = ["black", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
        # for i, patch in enumerate(boxplot["boxes"]):
        #     patch.set_facecolor(colors[i % 7])
        plt.xticks(rotation=45, ha="right")
        # plt.title(f"{self.read_path}: {metric}; best {top_n} results")
        plt.title(f"{self.file}")
        plt.ylabel(metric)
        plt.xlabel("Detector")
        plt.tight_layout()
        if show:
            plt.tight_layout()
            plt.show()
        else:
            plt.tight_layout()
            Path(self.write_root).mkdir(parents=True, exist_ok=True)
            plt.savefig(f"{self.write_root}/{metric[:3]}_boxes.eps", format="eps")
            plt.clf()

    def plot_scatter_metrics_per_file(
        self, x_metric="lpd (ht)", y_metric="acc (ht-dd)", show: bool = False
    ):
        for file_ in self.crawler.crawl():
            full_path = os.path.join(self.read_path, file_)
            df = pd.read_csv(full_path)
            df.dropna(inplace=True, subset=[x_metric, y_metric])
            if len(df) > 0:
                detector = os.path.basename(os.path.splitext(file_)[0])
                slope, intercept, r_value, p_value, std_err = linregress(
                    df[x_metric], df[y_metric]
                )
                if show:
                    plt.scatter(x=df[x_metric], y=df[y_metric], label=detector)
                    plt.legend()
                    plt.xlabel(x_metric)
                    plt.ylabel(y_metric)
                    plt.title(f"{self.read_path}/{detector}: R²={r_value}")
                    plt.tight_layout()
                    plt.show()
                else:
                    print(f"{full_path}: R²({x_metric}, {y_metric}) = {r_value}")

    def plot_scatter_metrics(
        self,
        x_metric="lpd (ht)",
        y_metric="acc (ht-dd)",
        markersize=30,
        log_y: bool = False,
        show=False,
        print_r2=False,
    ):
        results_x = defaultdict(list)
        results_y = defaultdict(list)
        for file_ in self.crawler.crawl():
            df = pd.read_csv(os.path.join(self.read_path, file_))
            df.dropna(inplace=True, subset=[x_metric, y_metric])
            detector = os.path.basename(os.path.splitext(file_)[0])
            results_x[detector] += list(df[x_metric])
            results_y[detector] += list(df[y_metric])

        x = []
        y = []
        for key in results_x.keys():
            x += results_x[key]
            y += results_y[key]
            if not print_r2:
                plt.scatter(
                    x=results_x[key],
                    y=results_y[key],
                    label=key,
                    c=self.colors[key],
                    marker=self.markers[key],
                    s=markersize,
                )
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        if print_r2:
            print(f"{self.read_path}: R²({x_metric}, {y_metric}) = {r_value}")
        else:
            # plt.plot(x, intercept + slope + np.array(x), "r")
            plt.rcParams.update({"font.size": 12})
            plt.legend()
            if log_y:
                plt.yscale("log")
            plt.grid()
            if show:
                plt.title(f"{self.read_path}: R²={r_value}")
                plt.xlabel(x_metric)
                plt.ylabel(y_metric)
                plt.tight_layout()
                plt.show()
            else:
                plt.title(self.file)
                if "acc" in x_metric:
                    plt.xlabel("mean accuracy (Hoeffding Tree)")
                else:
                    plt.xlabel("mean lift-per-drift (Hoeffding Tree)")
                plt.ylabel("mean number of detected drifts")
                file_name = (
                    f"{self.file}_{x_metric.split(' ')[0]}_{y_metric.split(' ')[0]}.eps"
                )
                write_path = os.path.join(self.write_root, file_name)
                Path(self.write_root).mkdir(parents=True, exist_ok=True)
                plt.tight_layout()
                plt.savefig(write_path, format="eps")
                plt.clf()

    def failure_bar_plot(self, all_counts, show: bool = False):
        fig, ax = plt.subplots()
        width = 0.1
        for i, detector in enumerate(all_counts):
            x = np.array(list(all_counts[detector].keys())) - 0.35
            y = list(all_counts[detector].values())
            offset = width * i
            ax.bar(
                x + offset, y, width, label=detector, color=self.colors[detector]
            )
        ax.legend()
        ax.set_xticks(np.arange(11))
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 20, 40, 60, 80, 100])
        ax.grid()
        ax.set_title("Failure rates")
        ax.set_ylabel("Configurations [%]")
        ax.set_xlabel("Number of failures")
        plt.rcParams.update({"font.size": 12})
        if show:
            plt.show()
            plt.tight_layout()
        else:
            Path(self.write_root).mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(f"{self.write_root}/failure_rates.eps", format="eps")
            plt.clf()

    @staticmethod
    def _get_samples(detector, row):
        if detector in ["csddm", "spll", "udetect", "ibdd", "bndm", "ocdd"]:
            return row["n_samples"]
        elif detector == "d3":
            return row["n_reference_samples"]
        raise NotImplementedError(
            f"Reset period cannot be determined for detector {detector}"
        )

    @staticmethod
    def _get_all_samples(detector, row):
        if detector in ["csddm", "spll", "udetect", "ibdd", "bndm"]:
            return 2 * row["n_samples"]
        elif detector == "d3":
            return row["n_reference_samples"] * (1 + row["recent_samples_proportion"])
        elif detector in ["ocdd"]:
            return row["n_samples"]
        raise NotImplementedError(
            f"Reset period cannot be determined for detector {detector}"
        )

    @staticmethod
    def _get_reset_period(detector, row):
        if detector in ["csddm", "spll", "udetect"]:
            return row["n_samples"]
        elif detector == "ocdd":
            return row["n_samples"] * row["threshold"]
        elif detector == "bndm":
            return 2 * row["n_samples"]
        elif detector == "ibdd":
            return 0
        elif detector == "d3":
            return row["n_reference_samples"]
        raise NotImplementedError(
            f"Reset period cannot be determined for detector {detector}"
        )
