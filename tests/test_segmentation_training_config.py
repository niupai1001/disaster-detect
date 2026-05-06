import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation_training.config import load_config, validate_config  # noqa: E402


class SegmentationTrainingConfigTests(unittest.TestCase):
    def test_all_e0_e3_configs_validate(self):
        config_dir = Path(__file__).resolve().parents[1] / "configs" / "segmentation_training"

        for path in sorted(config_dir.glob("e*.yaml")):
            with self.subTest(path=path.name):
                config = load_config(path)
                self.assertEqual(config["task_id"], "task-afc7f2c25f8f")
                self.assertEqual(config["input_channels"], ["F16", "F17"])

    def test_test_split_selection_is_rejected(self):
        config = load_config(
            Path(__file__).resolve().parents[1] / "configs" / "segmentation_training" / "e1_binary_c5_unet.yaml"
        )
        config["split_policy"]["allow_test_split_for_selection"] = True

        with self.assertRaisesRegex(ValueError, "Test split"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()

