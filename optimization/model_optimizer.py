import time
from typing import List, Optional

from metrics.metrics import get_metrics
from .classifiers import Classifiers
from .config_generator import ConfigGenerator
from .logger import ExperimentLogger
from .parameter import Parameter


class ModelOptimizer:
    def __init__(
        self,
        base_model: callable,
        parameters: List[Parameter],
        n_runs: int,
        seeds: Optional[List[int]] = None,
    ):
        """
        Init a new ModelOptimizer.

        :param base_model: a callable of the detector under test
        :param parameters: the configuration parameters
        :param n_runs: the number of test runs for each configuration
        :param seeds: the seeds or None
        """
        self.base_model = base_model
        self.configs = ConfigGenerator(parameters, seeds=seeds)
        self.classifiers = None
        self.n_runs = n_runs

    def _model_generator(self):
        """
        A generator that yields initialized models using configurations provided by the ConfigGenerator.

        :return: the initialized models
        """
        for config in self.configs:
            yield self.base_model(**config), config

    def optimize(self, stream, experiment_name, n_training_samples, verbose=False):
        """
        Optimize the model on the given data stream and log the results using the ExperimentLogger.

        :param stream: the data stream
        :param experiment_name: the name of the experiment
        :param n_training_samples: the number of training samples
        """
        for run in range(self.n_runs):
            logger = ExperimentLogger(
                stream=stream,
                model=self.base_model.__name__,
                experiment_name=experiment_name,
                config_keys=self.configs.get_parameter_names(),
            )
            for model, config in self._model_generator():
                if verbose:
                    print(f"{logger.model}: {config}")
                self.classifiers = Classifiers()
                drifts = []
                labels = []
                predictions = []
                train_steps = 0
                total_update_time = 0.0
                t_run_start = time.perf_counter()
                for i, (x, y) in enumerate(stream):
                    if i != 0:
                        predictions.append(self.classifiers.predict(x))
                        labels.append(y)
                    t0 = time.perf_counter()
                    updated = model.update(x)
                    total_update_time += time.perf_counter() - t0
                    if updated:
                        drifts.append(i)
                        self.classifiers.reset()
                        train_steps = 0
                    self.classifiers.fit(x, y, nonadaptive=i < n_training_samples)
                    train_steps += 1
                metrics = get_metrics(stream, drifts, labels, predictions)
                logger.log(config, metrics, drifts)
                total_run_time = time.perf_counter() - t_run_start
                print(f"  [Timing] total run time            : {total_run_time:.3f}s")
                print(f"  [Timing] total update() time       : {total_update_time:.3f}s")
                if hasattr(model, 'total_detect_drift_time'):
                    print(f"  [Timing] total _detect_drift() time: {model.total_detect_drift_time:.3f}s")
                if hasattr(model, 'total_aspt_time'):
                    print(f"  [Timing] total _aspt_test() time   : {model.total_aspt_time:.3f}s")
