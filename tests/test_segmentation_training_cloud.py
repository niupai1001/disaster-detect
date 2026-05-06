import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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


if __name__ == "__main__":
    unittest.main()

