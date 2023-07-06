import unittest

import numpy as np

from optimization.parameter import Parameter


class ParameterTest(unittest.TestCase):
    def test_name(self):
        for _ in range(10):
            name = str(np.random.random())
            p = Parameter(name, value=0)
            self.assertEqual(name, p.name)

    def test_single_value(self):
        p = Parameter("a", value=5)
        values = list(p)
        self.assertListEqual([5], values)

    def test_single_string(self):
        p = Parameter("a", value="test value")
        values = list(p)
        self.assertListEqual(["test value"], values)

    def test_single_float(self):
        value = np.random.random()
        p = Parameter("a", value=value)
        values = list(p)
        self.assertListEqual([value], values)

    def test_two_values(self):
        p = Parameter("a", value=1, max_value=2)
        values = list(p)
        self.assertListEqual([1, 2], values)

    def test_3_values(self):
        p = Parameter("a", value=1, max_value=3, n_values=3)
        values = list(p)
        self.assertListEqual([1, 2, 3], values)

    def test_10_values(self):
        p = Parameter("a", value=11, max_value=20, n_values=10)
        values = list(p)
        self.assertListEqual([11, 12, 13, 14, 15, 16, 17, 18, 19, 20], values)

    def test_5_step_size(self):
        p = Parameter("a", value=10, max_value=12, step_size=5)
        values = list(p)
        self.assertListEqual([10], values)

    def test_5_step_size_multiple_steps(self):
        p = Parameter("a", value=0, max_value=20, step_size=5)
        values = list(p)
        self.assertListEqual([0, 5, 10, 15, 20], values)

    def test_float_step_size(self):
        p = Parameter("a", value=0, max_value=1, step_size=0.2)
        values = list(p)
        for i, value in enumerate(values):
            self.assertAlmostEqual(i * 0.2, value)

    def test_float_step_size_offset(self):
        p = Parameter("a", value=2, max_value=3, step_size=0.2)
        values = list(p)
        for i, value in enumerate(values):
            self.assertAlmostEqual(2 + i * 0.2, value)

    def test_n_steps_with_step_size(self):
        p = Parameter("a", value=3, step_size=2, n_values=5)
        values = list(p)
        self.assertListEqual([3, 5, 7, 9, 11], values)

    def test_list_values(self):
        values = list(np.random.random(20))
        p = Parameter("a", values=values)
        self.assertListEqual(values, list(p))


if __name__ == "__main__":
    unittest.main()
