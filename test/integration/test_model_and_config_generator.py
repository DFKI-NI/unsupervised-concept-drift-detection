import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from optimization.model_optimizer import ModelOptimizer
from optimization.parameter import Parameter


class ModelGeneratorConfigGeneratorIntegrationTest(unittest.TestCase):
    @staticmethod
    def _setup_optimizer(model_config):
        mock_model = MagicMock()
        mock_model.return_value = MagicMock()
        mock_model.__name__ = MagicMock()
        model_optimizer = ModelOptimizer(
            base_model=mock_model,
            parameters=model_config,
            seeds=[0] * 6,
            n_runs=1,
        )
        return model_optimizer, mock_model

    @patch("optimization.model_optimizer.ExperimentLogger")
    def test_single_model(self, mock_logger):
        model_config = [Parameter("a", value=27)]
        model_optimizer, mock_model = self._setup_optimizer(model_config)
        for model, config in model_optimizer._model_generator():
            expected_config = {"a": 27, "seed": 0}
            self.assertDictEqual(expected_config, config)
            mock_model.assert_called_with(**config)
            mock_model.assert_called_with(**expected_config)

    @patch("optimization.model_optimizer.ExperimentLogger")
    def test_single_model_random_values(self, mock_logger):
        for _ in range(5):
            model_config = [
                Parameter("a", value=np.random.random()),
                Parameter("b", value=np.random.random()),
                Parameter("c", value=np.random.random()),
            ]
            model_optimizer, mock_model = self._setup_optimizer(model_config)
            for model, config in model_optimizer._model_generator():
                expected_config = {
                    parameter.name: parameter.value for parameter in model_config
                }
                expected_config["seed"] = 0
                self.assertDictEqual(expected_config, config)
                mock_model.assert_called_with(**config)
                mock_model.assert_called_with(**expected_config)

    @patch("optimization.model_optimizer.ExperimentLogger")
    def test_six_models(self, mock_logger):
        model_config = [
            Parameter("a", value=1),
            Parameter("b", value=2, max_value=9, step_size=7),
            Parameter("c", value=0, max_value=4, n_values=3),
        ]
        model_optimizer, mock_model = self._setup_optimizer(model_config)
        models_and_configs = list(model_optimizer._model_generator())
        self.assertDictEqual(
            {"a": 1, "b": 2, "c": 0, "seed": 0}, models_and_configs[0][1]
        )
        self.assertDictEqual(
            {"a": 1, "b": 2, "c": 0, "seed": 0}, mock_model.call_args_list[0].kwargs
        )
        self.assertDictEqual(
            {"a": 1, "b": 2, "c": 2, "seed": 0}, models_and_configs[1][1]
        )
        self.assertDictEqual(
            {"a": 1, "b": 2, "c": 2, "seed": 0}, mock_model.call_args_list[1].kwargs
        )
        self.assertDictEqual(
            {"a": 1, "b": 2, "c": 4, "seed": 0}, models_and_configs[2][1]
        )
        self.assertDictEqual(
            {"a": 1, "b": 2, "c": 4, "seed": 0}, mock_model.call_args_list[2].kwargs
        )
        self.assertDictEqual(
            {"a": 1, "b": 9, "c": 0, "seed": 0}, models_and_configs[3][1]
        )
        self.assertDictEqual(
            {"a": 1, "b": 9, "c": 0, "seed": 0}, mock_model.call_args_list[3].kwargs
        )
        self.assertDictEqual(
            {"a": 1, "b": 9, "c": 2, "seed": 0}, models_and_configs[4][1]
        )
        self.assertDictEqual(
            {"a": 1, "b": 9, "c": 2, "seed": 0}, mock_model.call_args_list[4].kwargs
        )
        self.assertDictEqual(
            {"a": 1, "b": 9, "c": 4, "seed": 0}, models_and_configs[5][1]
        )
        self.assertDictEqual(
            {"a": 1, "b": 9, "c": 4, "seed": 0}, mock_model.call_args_list[5].kwargs
        )

    @patch("optimization.model_optimizer.ExperimentLogger")
    def test_repeat_runs(self, mock_logger):
        model_config = [
            Parameter("a", value=1),
            Parameter("b", value=2, max_value=9, step_size=7),
            Parameter("c", value=0, max_value=4, n_values=3),
        ]
        model_optimizer, mock_model = self._setup_optimizer(model_config)
        models_and_configs = list(model_optimizer._model_generator())
        models_and_configs2 = list(model_optimizer._model_generator())
        self.assertEqual(models_and_configs, models_and_configs2)


if __name__ == "__main__":
    unittest.main()
