import unittest
from unittest.mock import MagicMock, patch

from optimization.logger import ExperimentLogger


class ExperimentLoggerTest(unittest.TestCase):
    common_columns = [
        "lpd (ht)",
        "lpd (nb)",
        "acc (ht-no dd)",
        "acc (nb-no dd)",
        "acc (ht-dd)",
        "acc (nb-dd)",
        "f1 (ht-no dd)",
        "f1 (nb-no dd)",
        "f1 (ht-dd)",
        "f1 (nb-dd)",
    ]

    @patch("optimization.logger.os.makedirs")
    @patch("optimization.logger.os.path.exists")
    @patch("builtins.open")
    def test_init_stream_with_drifts(self, mock_open, mock_exists, mock_makedirs):
        mock_exists.return_value = False
        experiment_name = "test_init_unsupervised"
        config_keys = ["key1", "key2", "key3", "key4"]
        logger = ExperimentLogger(
            MagicMock(), MagicMock(), experiment_name, config_keys
        )
        self.assertTrue(experiment_name in logger.file_name)
        mock_open.assert_called_with(logger.full_path, "w")
        self.assertListEqual(
            ["key1", "key2", "key3", "key4"]
            + self.common_columns
            + ["mtr", "mtfa", "mtd", "mdr", "drifts"],
            logger.columns,
        )

    @patch("optimization.logger.os.makedirs")
    @patch("optimization.logger.os.path.exists")
    @patch("builtins.open")
    def test_init_stream_without_drifts(self, mock_open, mock_exists, mock_makedirs):
        mock_exists.return_value = False
        experiment_name = "test_init_unsupervised"
        config_keys = ["key1", "key2", "key3", "key4"]
        mock_stream = MagicMock()
        del mock_stream.drifts
        logger = ExperimentLogger(
            mock_stream, MagicMock(), experiment_name, config_keys
        )
        self.assertTrue(experiment_name in logger.file_name)
        mock_open.assert_called_with(logger.full_path, "w")
        self.assertListEqual(
            ["key1", "key2", "key3", "key4"]
            + self.common_columns
            + ["drifts"],
            logger.columns,
        )

    @patch("optimization.logger.os.path.exists")
    @patch("optimization.logger.csv.DictWriter")
    @patch("builtins.open")
    def test_log_row(self, mock_open, mock_dictwriter, mock_exists):
        mock_exists.return_value = True
        mock_dictwriter.return_value = MagicMock()
        config_keys = ["key1", "key2"]
        logger = ExperimentLogger(MagicMock(), MagicMock(), "test_log_row", config_keys)
        results = MagicMock()
        results.to_dict.return_value = {"metric1": 0}
        logger.log({"key1": 1, "key2": 2}, results, [1, 2, 3])
        mock_dictwriter.return_value.writerow.assert_called_with(
            {
                "key1": 1,
                "key2": 2,
                "metric1": 0,
                "drifts": [1, 2, 3],
            }
        )


if __name__ == "__main__":
    unittest.main()
