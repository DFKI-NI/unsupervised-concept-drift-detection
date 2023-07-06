import unittest

import numpy as np

from detectors.base import UnsupervisedDriftDetector
from optimization.model_optimizer import ModelOptimizer
from optimization.parameter import Parameter


class TestModel(UnsupervisedDriftDetector):
    def __init__(self, seed, param):
        super().__init__(seed)

    def update(self, features: dict) -> bool:
        pass


class ReproducibilityTest(unittest.TestCase):
    @staticmethod
    def _get_model_optimizer(param_values, seeds=None):
        optimizer = ModelOptimizer(
            base_model=TestModel,
            parameters=[Parameter("param", values=param_values)],
            seeds=seeds,
            n_runs=0,
        )
        return optimizer

    def test_seed_usage(self):
        for _ in range(10):
            param_values = np.zeros(np.random.randint(4, 33))
            optimizer = self._get_model_optimizer(param_values)
            seeds = []
            for model, config in optimizer._model_generator():
                self.assertEqual(model.seed, config["seed"])
                seeds.append(model.seed)

            optimizer = self._get_model_optimizer(param_values, seeds)
            for seed, (model, config) in zip(seeds, optimizer._model_generator()):
                self.assertEqual(model.seed, config["seed"])
                self.assertEqual(seed, model.seed)

    def test_deterministic_configs(self):
        for _ in range(10):
            param_values = np.random.random(np.random.randint(4, 33))
            optimizer = self._get_model_optimizer(param_values)
            seeds = []
            configs = []
            for model, config in optimizer._model_generator():
                self.assertEqual(model.seed, config["seed"])
                seeds.append(model.seed)
                configs.append(config)

            optimizer = self._get_model_optimizer(param_values, seeds)
            for i, (model, config) in enumerate(optimizer._model_generator()):
                self.assertEqual(model.seed, config["seed"])
                self.assertEqual(seeds[i], model.seed)
                self.assertDictEqual(configs[i], config)


if __name__ == "__main__":
    unittest.main()
