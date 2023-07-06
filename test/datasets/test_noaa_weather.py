import unittest

from datasets import NOAAWeather


class NOAAWeatherTest(unittest.TestCase):
    def setUp(self):
        self.stream = NOAAWeather()

    def test_n_features_and_n_samples(self):
        i = 0
        for i, (x, y) in enumerate(self.stream):
            self.assertEqual(self.stream.n_features, len(x))
            self.assertEqual(8, len(x))
        self.assertEqual(self.stream.n_samples, i + 1)
        self.assertEqual(18_159, i + 1)


if __name__ == "__main__":
    unittest.main()
