import unittest

from datasets import Airlines


class AirlinesTest(unittest.TestCase):
    def setUp(self):
        self.stream = Airlines(directory_path="datasets/files")

    def test_n_features_and_n_samples(self):
        i = 0
        for i, (x, y) in enumerate(self.stream):
            self.assertEqual(self.stream.n_features, len(x))
            self.assertEqual(7, len(x))
        self.assertEqual(self.stream.n_samples, i + 1)
        self.assertEqual(539_383, i + 1)


if __name__ == "__main__":
    unittest.main()
