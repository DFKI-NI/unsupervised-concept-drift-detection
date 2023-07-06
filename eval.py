import os

from eval.cleaner import Cleaner
from eval.parser import SummaryToDetectorParser, SummariesToAverageParser
from eval.plotter import SummaryPlotter
from eval.summarize import Summarizer


def main():
    show = False
    cleaner = Cleaner(
        read_root="results",
        write_clean_root="results_clean",
        write_repeats_root="results_periodic",
        write_no_detections_root="results_no_detections"
    )
    cleaner.filter_results()
    summarizer = Summarizer(read_root="results_clean", write_root="results_summarized")
    summarizer.summarize()

    # MTR ANALYSIS
    parser = SummaryPlotter(read_root="results_summarized", file="InsectsAbruptBalanced")
    parser.plot_scatter_metrics_per_file(x_metric="acc (ht-dd) (mean)", y_metric="mtr (mean)")
    parser.plot_scatter_metrics_per_file(x_metric="lpd (ht) (mean)", y_metric="mtr (mean)")
    parser = SummaryPlotter(read_root="results_summarized", file="InsectsAbruptBalanced", write_root="results_figures")
    parser.plot_top_metric_boxes(metric="mtr (mean)", show=show)

    # LPD ANALYSIS
    for file_ in sorted(os.listdir("results_summarized")):
        parser = SummaryPlotter(read_root="results_summarized", file=file_)
        parser.plot_scatter_metrics(x_metric="lpd (ht) (mean)", y_metric="lpd (nb) (mean)", show=show)
    for file_ in sorted(os.listdir("results_summarized")):
        parser = SummaryPlotter(read_root="results_summarized", file=file_)
        parser.plot_scatter_metrics(x_metric="lpd (ht) (mean)", y_metric="acc (ht-dd) (mean)", show=show)
    for file_ in sorted(os.listdir("results_summarized")):
        parser = SummaryPlotter(read_root="results_summarized", file=file_)
        parser.plot_scatter_samples(metric="lpd (ht) (mean)")
    for file_ in sorted(os.listdir("results_summarized")):
        parser = SummaryPlotter(read_root="results_summarized", file=file_, write_root="results_figures")
        parser.plot_scatter_metrics(x_metric="acc (ht-dd) (mean)", y_metric="drifts (mean)", log_y=True, show=show)
        parser.plot_scatter_metrics(x_metric="lpd (ht) (mean)", y_metric="drifts (mean)", log_y=True, show=show)

    # ACC ANALYSIS
    for file_ in sorted(os.listdir("results_summarized")):
        parser = SummaryPlotter(read_root="results_summarized", file=file_)
        parser.plot_scatter_samples(metric="acc (ht-dd) (mean)")

    parser = SummaryToDetectorParser("results_summarized", "results_best")
    parser.get_top_n_configurations("bndm", n_configs=10000, metric="lpd (ht)")
    parser.get_top_n_configurations("csddm", n_configs=10000, metric="lpd (ht)")
    parser.get_top_n_configurations("d3", n_configs=10000, metric="lpd (ht)")
    parser.get_top_n_configurations("ibdd", n_configs=10000, metric="lpd (ht)")
    parser.get_top_n_configurations("ocdd", n_configs=10000, metric="lpd (ht)")
    parser.get_top_n_configurations("spll", n_configs=10000, metric="lpd (ht)")
    parser.get_top_n_configurations("udetect", n_configs=10000, metric="lpd (ht)")
    parser.get_top_n_configurations("bndm", n_configs=10000, metric="acc (ht-dd)")
    parser.get_top_n_configurations("csddm", n_configs=10000, metric="acc (ht-dd)")
    parser.get_top_n_configurations("d3", n_configs=10000, metric="acc (ht-dd)")
    parser.get_top_n_configurations("ibdd", n_configs=10000, metric="acc (ht-dd)")
    parser.get_top_n_configurations("ocdd", n_configs=10000, metric="acc (ht-dd)")
    parser.get_top_n_configurations("spll", n_configs=10000, metric="acc (ht-dd)")
    parser.get_top_n_configurations("udetect", n_configs=10000, metric="acc (ht-dd)")

    parser = SummariesToAverageParser("results_summarized", "results_best")
    detectors = ["bndm", "csddm", "d3", "ibdd", "ocdd", "spll", "udetect"]
    all_counts = {}
    for detector in detectors:
        parser.get_average_rank_per_config(detector, metric="acc (ht-dd)")
        counts = parser.get_average_rank_per_config(detector, metric="lpd (ht)")
        all_counts[detector] = counts
    plotter = SummaryPlotter("", "", "results_figures")
    plotter.failure_bar_plot(all_counts, show=show)


if __name__ == "__main__":
    main()
