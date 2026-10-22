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

    def test_e1b_e1c_configs_are_one_factor_diagnostics(self):
        config_dir = Path(__file__).resolve().parents[1] / "configs" / "segmentation_training"
        control = load_config(config_dir / "e1_binary_c5_unet.yaml")
        sampler = load_config(config_dir / "e1b_binary_c5_unet_foreground_sampler.yaml")
        weighted_ce = load_config(config_dir / "e1c_binary_c5_unet_weighted_ce.yaml")
        ce_dice = load_config(config_dir / "e1c_binary_c5_unet_ce_dice.yaml")

        self.assertEqual(sampler["diagnostics"]["changed_factor"], "sampler")
        self.assertEqual(weighted_ce["diagnostics"]["changed_factor"], "loss_weighting")
        self.assertEqual(ce_dice["diagnostics"]["changed_factor"], "loss_family")
        self.assertEqual(sampler["training"]["loss"], control["training"]["loss"])
        self.assertEqual(weighted_ce["training"]["sampler"], sampler["training"]["sampler"])
        self.assertEqual(ce_dice["training"]["sampler"], sampler["training"]["sampler"])
        self.assertEqual(weighted_ce["training"]["loss"], "weighted_cross_entropy")
        self.assertEqual(ce_dice["training"]["loss"], "cross_entropy_dice")

    def test_test_split_selection_is_rejected(self):
        config = load_config(
            Path(__file__).resolve().parents[1] / "configs" / "segmentation_training" / "e1_binary_c5_unet.yaml"
        )
        config["split_policy"]["allow_test_split_for_selection"] = True

        with self.assertRaisesRegex(ValueError, "Test split"):
            validate_config(config)

    def test_test_split_in_selection_splits_is_rejected(self):
        config = load_config(
            Path(__file__).resolve().parents[1] / "configs" / "segmentation_training" / "e1_binary_c5_unet.yaml"
        )
        config["split_policy"]["selection_splits"].append("test")

        with self.assertRaisesRegex(ValueError, "Test split"):
            validate_config(config)

    def test_invalid_sampler_policy_is_rejected(self):
        config = load_config(
            Path(__file__).resolve().parents[1] / "configs" / "segmentation_training" / "e1_binary_c5_unet.yaml"
        )
        config["training"]["sampler"]["train_policy"] = "mystery"

        with self.assertRaisesRegex(ValueError, "train_policy"):
            validate_config(config)

    def test_v2_multichannel_config_validates_when_model_input_count_matches(self):
        config = load_config(
            Path(__file__).resolve().parents[1] / "configs" / "segmentation_training" / "e1_binary_c5_unet.yaml"
        )
        config["task_id"] = "task-2ba5416843b1"
        config["experiment_id"] = "V2_binary_c5_unet_post_optical"
        config["input_channels"] = ["F01", "F02", "F03", "F07", "F11", "F12"]
        config["model"]["input_channels"] = 6

        validate_config(config)

    def test_v3_indices_config_validates_with_raw_and_derived_channels(self):
        config = load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "segmentation_training"
            / "v3_binary_c5_unet_post_optical_indices_smoke.yaml"
        )

        self.assertEqual(config["model"]["input_channels"], 11)
        self.assertIn("NDVI", config["input_channels"])
        self.assertIn("BRIGHTNESS", config["input_channels"])

    def test_model_input_channel_count_must_match_input_channels(self):
        config = load_config(
            Path(__file__).resolve().parents[1] / "configs" / "segmentation_training" / "e1_binary_c5_unet.yaml"
        )
        config["task_id"] = "task-2ba5416843b1"
        config["input_channels"] = ["F01", "F02", "F03"]
        config["model"]["input_channels"] = 2

        with self.assertRaisesRegex(ValueError, "model.input_channels"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
