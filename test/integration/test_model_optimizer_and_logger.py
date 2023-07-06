import os
import unittest
from unittest.mock import patch
from uuid import uuid4

from metrics.metrics import ExperimentResult
from optimization.model_optimizer import ModelOptimizer
from optimization.parameter import Parameter


class TestDetector:
    def __init__(self, drift_positions, seed=None):
        self.i = -1
        self.drifts = drift_positions

    def update(self, features):
        self.i += 1
        return self.i in self.drifts


class TestStream:
    def __init__(self, length):
        self.length = length

    def __iter__(self):
        for i in range(self.length):
            yield i, 0


class OptimizeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base_name = f"INTEGRATION-TEST-{str(uuid4())}"

    @patch("optimization.config_generator.time")
    @patch("optimization.model_optimizer.get_metrics")
    @patch("optimization.model_optimizer.Classifiers")
    def test_one_run(
        self, mock_classifiers, mock_get_metrics, mock_config_gen_time
    ):
        mock_config_gen_time.time.return_value = 112244578
        mock_get_metrics.return_value = ExperimentResult(
            lpd=(1, 2), accuracies=[3, 4, 5, 6], f1_scores=[7, 8, 9, 10]
        )
        parameters = [Parameter("drift_positions", value=[10])]
        optimizer = ModelOptimizer(TestDetector, parameters, seeds=None, n_runs=1)
        optimizer.optimize(
            TestStream(20), f"{self.base_name}-ONE-RUN", n_training_samples=20
        )
        with open(
            os.path.join(
                "results/TestStream", f"TestDetector_{self.base_name}-ONE-RUN.csv"
            ),
            newline=None,
        ) as f:
            lines = f.readlines()
        self.assertEqual(
            "112244578,[10],1,2,3,4,5,6,7,8,9,10,[10]\n", lines[1]
        )

    @patch("optimization.model_optimizer.get_metrics")
    @patch("optimization.model_optimizer.Classifiers")
    def test_ten_runs(self, mock_classifiers, mock_get_metrics):
        name = f"{self.base_name}-TEN-RUNS"
        parameters = [Parameter("drift_positions", values=[[i] for i in range(10)])]
        optimizer = ModelOptimizer(TestDetector, parameters, seeds=None, n_runs=10)
        optimizer.optimize(TestStream(20), name, n_training_samples=20)
        filenames = os.listdir("results/TestStream")
        self.assertEqual(1, len(filenames))
        for filename in os.listdir("results/TestStream"):
            if name in filename:
                with open(os.path.join("results/TestStream", filename), newline=None) as f:
                    lines = f.readlines()
                    self.assertEqual(101, len(lines))

    def tearDown(self) -> None:
        for filename in os.listdir("results/TestStream"):
            if self.base_name in filename:
                os.remove(os.path.join("results/TestStream", filename))
            if len(os.listdir("results/TestStream")) == 0:
                os.rmdir("results/TestStream")


if __name__ == "__main__":
    unittest.main()
