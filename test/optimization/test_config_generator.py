import random
import time
import unittest
from unittest.mock import patch

from optimization.config_generator import ConfigGenerator
from optimization.parameter import Parameter


class ConfigGeneratorTest(unittest.TestCase):
    def test_single_config(self):
        parameters = [Parameter("a", value=5)]
        configs = list(ConfigGenerator(parameters, seeds=[0]))
        self.assertEqual(1, len(configs))
        self.assertDictEqual({"a": 5, "seed": 0}, configs[0])

    def test_two_values(self):
        configs = list(
            ConfigGenerator([Parameter("a", value=9, max_value=10)], seeds=[0, 0])
        )
        self.assertEqual(2, len(configs))
        self.assertDictEqual({"a": 9, "seed": 0}, configs[0])
        self.assertDictEqual({"a": 10, "seed": 0}, configs[1])

    def test_two_parameters_two_values(self):
        configs = list(
            ConfigGenerator(
                [
                    Parameter("a", value=1, max_value=3),
                    Parameter("b", value=2, max_value=5),
                ],
                seeds=[0] * 4,
            )
        )
        self.assertEqual(4, len(configs))
        self.assertDictEqual({"a": 1, "b": 2, "seed": 0}, configs[0])
        self.assertDictEqual({"a": 1, "b": 5, "seed": 0}, configs[1])
        self.assertDictEqual({"a": 3, "b": 2, "seed": 0}, configs[2])
        self.assertDictEqual({"a": 3, "b": 5, "seed": 0}, configs[3])

    def test_three_parameters_mixed_values(self):
        configs = list(
            ConfigGenerator(
                [
                    Parameter("a", value=1),
                    Parameter("b", value=2, max_value=9, step_size=7),
                    Parameter("c", value=0, max_value=4, n_values=3),
                ],
                seeds=[0] * 6,
            )
        )
        self.assertEqual(6, len(configs))
        self.assertDictEqual({"a": 1, "b": 2, "c": 0, "seed": 0}, configs[0])
        self.assertDictEqual({"a": 1, "b": 2, "c": 2, "seed": 0}, configs[1])
        self.assertDictEqual({"a": 1, "b": 2, "c": 4, "seed": 0}, configs[2])
        self.assertDictEqual({"a": 1, "b": 9, "c": 0, "seed": 0}, configs[3])
        self.assertDictEqual({"a": 1, "b": 9, "c": 2, "seed": 0}, configs[4])
        self.assertDictEqual({"a": 1, "b": 9, "c": 4, "seed": 0}, configs[5])

    def test_repeat_configs(self):
        config_generator = ConfigGenerator(
            [
                Parameter("a", value=1),
                Parameter("b", value=2, max_value=9, step_size=7),
                Parameter("c", value=0, max_value=4, n_values=3),
            ],
            seeds=[0] * 6,
        )
        configs = list(config_generator)
        configs2 = list(config_generator)
        self.assertEqual(configs, configs2)


class GetParameterNamesTest(unittest.TestCase):
    def test_one_parameter(self):
        configs = ConfigGenerator([Parameter("a", value=0)], seeds=None)
        names = configs.get_parameter_names()
        self.assertListEqual(["seed", "a"], names)

    def test_multiple_parameters(self):
        expected_names = ["a", "test", "lorem ipsum dolor sit amet", "0"]
        parameters = [Parameter(name, value=0) for name in expected_names]
        configs = ConfigGenerator(parameters, seeds=None)
        names = configs.get_parameter_names()
        expected_names = sorted(expected_names)
        expected_names.insert(0, "seed")
        self.assertListEqual(expected_names, names)


class SortParametersTest(unittest.TestCase):
    def test_no_seed_sorted(self):
        parameters = [Parameter(chr(i + 97), value=0) for i in range(5)]
        configs = ConfigGenerator(parameters, seeds=None)
        names = configs.get_parameter_names()
        self.assertListEqual(["seed", "a", "b", "c", "d", "e"], names)

    def test_no_seed_shuffled(self):
        parameter_names = ["a", "b", "c", "d", "e"]
        for _ in range(5):
            random.shuffle(parameter_names)
            parameters = [Parameter(name, value=0) for name in parameter_names]
            configs = ConfigGenerator(parameters, seeds=None)
            names = configs.get_parameter_names()
            self.assertListEqual(["seed", "a", "b", "c", "d", "e"], names)

    def test_with_seed_sorted(self):
        parameters = [Parameter(chr(i + 97), value=0) for i in range(5)]
        configs = ConfigGenerator(parameters, seeds=[0, 1, 2, 3])
        names = configs.get_parameter_names()
        self.assertListEqual(["seed", "a", "b", "c", "d", "e"], names)

    def test_with_seed_shuffled(self):
        parameter_names = ["a", "b", "c", "d", "e"]
        for _ in range(5):
            random.shuffle(parameter_names)
            parameters = [Parameter(name, value=0) for name in parameter_names]
            configs = ConfigGenerator(parameters, seeds=[0, 1, 2, 3])
            names = configs.get_parameter_names()
            self.assertListEqual(["seed", "a", "b", "c", "d", "e"], names)


class SeedGenerationTest(unittest.TestCase):
    @patch("optimization.config_generator.time")
    def test_seed_generation(self, time_mock):
        time_mock.time.return_value = 25
        parameters = [Parameter("a", value=0)]
        configs = ConfigGenerator(parameters)
        for config in configs:
            self.assertEqual(25, config["seed"])

    def test_different_seeds(self):
        parameters = [Parameter(f"{i}", value=i) for i in range(10)]
        configs = ConfigGenerator(parameters)
        seeds = []
        for config in configs:
            seeds.append(config["seed"])
            time.sleep(1)
        for i, seed in enumerate(seeds):
            for j in range(i + 1, len(seeds)):
                self.assertNotEqual(seed, seeds[j])

    @patch("optimization.config_generator.time")
    def test_time_called(self, time_mock):
        parameters = [Parameter("a", value=0)]
        configs = ConfigGenerator(parameters)
        for config in configs:
            time_mock.time.assert_called()


if __name__ == "__main__":
    unittest.main()
