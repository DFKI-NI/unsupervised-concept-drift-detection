import csv
import os.path

from metrics.metrics import ExperimentResult


class ExperimentLogger:
    """
    ExperimentLogger logs the results of each tested configuration by storing the configuration, the metrics and the
    detected drifts in a file named after the tested detector in a folder named after the used data stream.
    """
    def __init__(self, stream, model, experiment_name, config_keys):
        """
        Init a new ExperimentLogger.

        :param stream: the data stream of the experiment
        :param model: the detector under test
        :param experiment_name: the name of the file
        :param config_keys: the names of the model's configuration parameters
        """
        self.stream = stream
        self.stream_name = stream.__class__.__name__
        self.model = model
        self.experiment_name = experiment_name
        self.file_name = f"{model}_{experiment_name}.csv"
        self.path = os.path.join("results", self.stream_name)
        self.full_path = os.path.join(self.path, self.file_name)
        self.columns = config_keys + [
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
        ]
        if hasattr(stream, "drifts"):
            self.columns += ["mtr", "mtfa", "mtd", "mdr"]
        self.columns.append("drifts")
        self._create_log()

    def _create_log(self):
        """
        Create a new log if it doesn't exist in 'results/<data stream>/<detector name>_<experiment_name>.csv'.
        """
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        if not os.path.exists(self.full_path):
            with open(self.full_path, "w") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.columns)
                writer.writeheader()

    def log(self, config, results: ExperimentResult, drifts):
        """
        Log the experiment results and detected drifts of the given configuration.
        :param config: the configuration
        :param results: the results
        :param drifts: the detected drifts
        """
        row = {
            **config,
            **results.to_dict(hasattr(self.stream, "drifts")),
            "drifts": drifts,
        }
        with open(self.full_path, "a") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.columns)
            writer.writerow(row)
