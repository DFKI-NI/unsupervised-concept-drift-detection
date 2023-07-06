import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from optimization.model_optimizer import ModelOptimizer


class ModelGeneratorTest(unittest.TestCase):
    @staticmethod
    def _setup_optimizer(model_config):
        mock_model = MagicMock()
        mock_model.__name__ = MagicMock()
        mock_model.return_value = MagicMock()
        mock_config = MagicMock()
        mock_config.__iter__.return_value = model_config
        mock_stream = MagicMock()
        model_optimizer = ModelOptimizer(
            base_model=mock_model,
            parameters=MagicMock(),
            seeds=None,
            n_runs=1,
        )
        return model_optimizer, mock_model

    @patch("optimization.model_optimizer.ConfigGenerator")
    @patch("optimization.model_optimizer.ExperimentLogger")
    def test_initial_model(self, mock_logger, mock_configs):
        for _ in range(5):
            model_config = [
                {
                    "a": np.random.random(),
                    "b": np.random.random(),
                    "c": np.random.random(),
                }
            ]
            mock_configs.return_value = model_config
            model_optimizer, mock_model = self._setup_optimizer(model_config)
            i = -1
            for i, (model, _) in enumerate(model_optimizer._model_generator()):
                self.assertEqual(mock_model.return_value, model)
                mock_model.assert_called_with(**model_config[i])
            self.assertEqual(i + 1, len(model_config))

    @patch("optimization.model_optimizer.ConfigGenerator")
    @patch("optimization.model_optimizer.ExperimentLogger")
    def test_initial_config(self, mock_logger, mock_configs):
        for _ in range(5):
            model_config = [
                {
                    "a": np.random.random(),
                    "b": np.random.random(),
                    "c": np.random.random(),
                }
            ]
            mock_configs.return_value = model_config
            model_optimizer, mock_model = self._setup_optimizer(model_config)
            i = -1
            for i, (_, config) in enumerate(model_optimizer._model_generator()):
                self.assertDictEqual(model_config[0], config)
            self.assertEqual(i + 1, len(model_config))

    @patch("optimization.model_optimizer.ConfigGenerator")
    @patch("optimization.model_optimizer.ExperimentLogger")
    def test_full_sweep_models(self, mock_logger, mock_configs):
        model_config = [
            {"a": np.random.random(), "b": np.random.random(), "c": np.random.random()},
            {"d": np.random.random(), "e": np.random.random()},
            {"f": np.random.random()},
        ]
        mock_configs.return_value = model_config
        model_optimizer, mock_model = self._setup_optimizer(model_config)
        for config, (model, _) in zip(model_config, model_optimizer._model_generator()):
            self.assertEqual(mock_model.return_value, model)
            mock_model.assert_called_with(**config)

    @patch("optimization.model_optimizer.ConfigGenerator")
    @patch("optimization.model_optimizer.ExperimentLogger")
    def test_full_sweep_configs(self, mock_logger, mock_configs):
        model_config = [
            {"a": np.random.random(), "b": np.random.random(), "c": np.random.random()},
            {"d": np.random.random(), "e": np.random.random()},
            {"f": np.random.random(), "g": np.random.random()},
        ]
        mock_configs.return_value = model_config
        model_optimizer, mock_model = self._setup_optimizer(model_config)
        for config, (_, used_config) in zip(
            model_config, model_optimizer._model_generator()
        ):
            self.assertDictEqual(config, used_config)


if __name__ == "__main__":
    unittest.main()
