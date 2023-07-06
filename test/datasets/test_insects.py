"""This module tests the INSECTS datasets."""
import unittest

from datasets import (
    InsectsAbruptBalanced,
    InsectsAbruptImbalanced,
    InsectsIncrementalBalanced,
    InsectsIncrementalImbalanced,
    InsectsGradualBalanced,
    InsectsGradualImbalanced,
    InsectsIncrementalAbruptBalanced,
    InsectsIncrementalAbruptImbalanced,
    InsectsIncrementalReoccurringBalanced,
    InsectsIncrementalReoccurringImbalanced,
)


class AbruptBalancedTest(unittest.TestCase):
    """This class tests the abrupt balanced INSECTS dataset."""

    def setUp(self):
        self.stream = InsectsAbruptBalanced()

    def test_file(self):
        """Test the file path."""
        self.assertTrue("abrupt_balanced" in self.stream.full_path)

    def test_samples(self):
        """Test the number of samples and sample structure."""
        self.assertEqual(self.stream.n_samples, 52848)
        labels = set()
        i = 0
        for i, (features, label) in enumerate(self.stream):
            if i == 0:
                self.assertEqual(len(features), 33)
            labels.add(label)
        self.assertEqual(len(labels), 6)
        self.assertEqual(i + 1, self.stream.n_samples)


class IncrementalBalancedTest(unittest.TestCase):
    """This class tests the incremental balanced INSECTS dataset."""

    def setUp(self):
        self.stream = InsectsIncrementalBalanced()

    def test_file(self):
        """Test the file path."""
        self.assertTrue("incremental_balanced" in self.stream.full_path)

    def test_samples(self):
        """Test the number of samples and sample structure."""
        self.assertEqual(self.stream.n_samples, 57018)
        labels = set()
        i = 0
        for i, (features, label) in enumerate(self.stream):
            if i == 0:
                self.assertEqual(len(features), 33)
            labels.add(label)
        self.assertEqual(len(labels), 6)
        self.assertEqual(i + 1, self.stream.n_samples)


class IncrementalGradualBalancedTest(unittest.TestCase):
    """This class tests the incremental-gradual balanced INSECTS dataset."""

    def setUp(self):
        self.stream = InsectsGradualBalanced()

    def test_file(self):
        """Test the file path."""
        self.assertTrue("gradual_balanced" in self.stream.full_path)

    def test_samples(self):
        """Test the number of samples and sample structure."""
        self.assertEqual(self.stream.n_samples, 24150)
        labels = set()
        i = 0
        for i, (features, label) in enumerate(self.stream):
            if i == 0:
                self.assertEqual(len(features), 33)
            labels.add(label)
        self.assertEqual(len(labels), 6)
        self.assertEqual(i + 1, self.stream.n_samples)


class IncrementalAbruptBalancedTest(unittest.TestCase):
    """This class tests the incremental-abrupt balanced INSECTS dataset."""

    def setUp(self):
        self.stream = InsectsIncrementalAbruptBalanced()

    def test_file(self):
        """Test the file path."""
        self.assertTrue("incremental-abrupt_balanced" in self.stream.full_path)

    def test_samples(self):
        """Test the number of samples and sample structure."""
        self.assertEqual(self.stream.n_samples, 79986)
        labels = set()
        i = 0
        for i, (features, label) in enumerate(self.stream):
            if i == 0:
                self.assertEqual(len(features), 33)
            labels.add(label)
        self.assertEqual(len(labels), 6)
        self.assertEqual(i + 1, self.stream.n_samples)


class IncrementalReoccurringBalancedTest(unittest.TestCase):
    """This class tests the incremental-reoccurring balanced INSECTS dataset."""

    def setUp(self):
        self.stream = InsectsIncrementalReoccurringBalanced()

    def test_file(self):
        """Test the file path."""
        self.assertTrue("incremental-reoccurring_balanced" in self.stream.full_path)

    def test_samples(self):
        """Test the number of samples and sample structure."""
        self.assertEqual(self.stream.n_samples, 79986)
        labels = set()
        i = 0
        for i, (features, label) in enumerate(self.stream):
            if i == 0:
                self.assertEqual(len(features), 33)
            labels.add(label)
        self.assertEqual(len(labels), 6)
        self.assertEqual(i + 1, self.stream.n_samples)
