import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation_training.cloud import _pad_window  # noqa: E402
from segmentation_training.cli import main as training_main  # noqa: E402


class SegmentationTrainingCloudTests(unittest.TestCase):
    def test_cloud_confirm_train_calls_training_runner(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            config = root / "config.yaml"
            bundle = root / "bundle"
            run_dir = root / "run"
            config.write_text("task_id: task-afc7f2c25f8f\n", encoding="utf-8")
            with patch("segmentation_training.cli.run_cloud_training") as runner:
                runner.return_value = {"status": "completed"}

                exit_code = training_main(
                    [
                        "train",
                        "--config",
                        str(config),
                        "--bundle-dir",
                        str(bundle),
                        "--run-dir",
                        str(run_dir),
                        "--cloud-confirm",
                    ]
                )

            self.assertEqual(exit_code, 0)
            runner.assert_called_once()

    def test_pad_window_makes_small_training_samples_batchable(self):
        channels = np.ones((2, 148, 152), dtype=np.float32)
        label = np.ones((148, 152), dtype=np.uint8)

        padded_channels = _pad_window(channels, 256, fill_value=0, np=np)
        padded_label = _pad_window(label, 256, fill_value=255, np=np)

        self.assertEqual(padded_channels.shape, (2, 256, 256))
        self.assertEqual(padded_label.shape, (256, 256))
        self.assertEqual(float(padded_channels[:, :148, :152].sum()), float(channels.sum()))
        self.assertTrue((padded_label[148:, :] == 255).all())


if __name__ == "__main__":
    unittest.main()
