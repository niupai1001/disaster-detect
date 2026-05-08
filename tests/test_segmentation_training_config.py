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

    def test_rtx4080_config_enables_cuda_performance_options(self):
        config = load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "segmentation_training"
            / "v3_binary_c5_unet_post_optical_indices_rtx4080.yaml"
        )

        self.assertEqual(config["model"]["base_channels"], 64)
        self.assertEqual(config["training"]["batch_size"], 16)
        self.assertTrue(config["training"]["performance"]["mixed_precision"])
        self.assertTrue(config["training"]["performance"]["channels_last"])
        self.assertTrue(config["training"]["performance"]["cudnn_benchmark"])
        self.assertTrue(config["training"]["performance"]["cache_records"])
        self.assertIn(0.9, config["metrics"]["threshold_sweep"])
        self.assertIn("checkpoints/best_mean_iou.pt", config["metrics"]["required_outputs"])

    def test_rtx4080_stable_config_reduces_overprediction_pressure(self):
        config = load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "segmentation_training"
            / "v3_binary_c5_unet_post_optical_indices_rtx4080_stable.yaml"
        )

        self.assertEqual(config["training"]["learning_rate"], 0.0003)
        self.assertEqual(config["training"]["dice_weight"], 0.5)
        self.assertEqual(config["training"]["class_weights"]["max_weight"], 10.0)
        self.assertEqual(config["training"]["sampler"]["foreground_probability"], 0.7)
        self.assertTrue(config["training"]["performance"]["cache_records"])

    def test_v6_11ch_augmented_diagnostic_config_enables_preview_and_channel_contribution(self):
        config = load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "segmentation_training"
            / "v6_11ch_unet_aug_diagnostics"
            / "c5_11ch_unet_aug_rtx4080.yaml"
        )

        self.assertEqual(len(config["input_channels"]), 11)
        self.assertTrue(config["training"]["augmentation"]["enabled"])
        self.assertEqual(config["metrics"]["preview"]["max_items"], "all")
        self.assertEqual(config["metrics"]["preview"]["worst_false_positive_items"], 24)
        self.assertIn("indices", config["metrics"]["channel_contribution"]["groups"])

    def test_c2_landslide_9ch_smoke_config_matches_paired_chip_channels(self):
        config = load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "segmentation_training"
            / "c2_landslide_9ch_unet_smoke.yaml"
        )

        self.assertEqual(config["class_scope"], "binary_c2")
        self.assertEqual(
            config["input_channels"],
            ["B", "G", "R", "NIR", "ELEVATION", "SLOPE", "ASPECT", "CURVATURE", "TPI"],
        )
        self.assertEqual(config["model"]["input_channels"], 9)
        self.assertEqual(config["metrics"]["class_ids"], [0, 1])

    def test_model_input_channel_count_must_match_input_channels(self):
        config = load_config(
            Path(__file__).resolve().parents[1] / "configs" / "segmentation_training" / "e1_binary_c5_unet.yaml"
        )
        config["task_id"] = "task-2ba5416843b1"
        config["input_channels"] = ["F01", "F02", "F03"]
        config["model"]["input_channels"] = 2

        with self.assertRaisesRegex(ValueError, "model.input_channels"):
            validate_config(config)

    def test_unknown_model_family_is_rejected(self):
        config = load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "segmentation_training"
            / "v3_binary_c5_unet_post_optical_indices_rtx4080_stable.yaml"
        )
        config["model"]["family"] = "mystery"

        with self.assertRaisesRegex(ValueError, "Unsupported model.family"):
            validate_config(config)

    def test_architecture_benchmark_configs_validate_and_preserve_shared_protocol(self):
        config_dir = Path(__file__).resolve().parents[1] / "configs" / "segmentation_training"
        baseline = load_config(config_dir / "v3_binary_c5_unet_post_optical_indices_rtx4080_stable.yaml")
        architecture_dir = config_dir / "architectures"
        expected = {
            "c5_11ch_resunet_rtx4080.yaml": "resunet",
            "c5_11ch_attention_unet_rtx4080.yaml": "attention_unet",
            "c5_11ch_unetpp_rtx4080.yaml": "unet_plus_plus",
            "c5_11ch_deeplabv3plus_rtx4080.yaml": "deeplabv3_plus",
            "c5_11ch_transformer_unet_rtx4080.yaml": "transformer_unet",
        }

        for filename, family in expected.items():
            with self.subTest(filename=filename):
                config = load_config(architecture_dir / filename)
                self.assertEqual(config["model"]["family"], family)
                self.assertEqual(config["task_id"], baseline["task_id"])
                self.assertEqual(config["class_scope"], baseline["class_scope"])
                self.assertEqual(config["input_channels"], baseline["input_channels"])
                self.assertEqual(config["split_policy"], baseline["split_policy"])
                self.assertEqual(config["training"], baseline["training"])
                self.assertEqual(config["metrics"], baseline["metrics"])
                self.assertEqual(config["cloud"], baseline["cloud"])

    def test_architecture_smoke_configs_are_short_validation_only_preflights(self):
        config_dir = Path(__file__).resolve().parents[1] / "configs" / "segmentation_training"
        smoke_dir = config_dir / "architecture_smoke"
        expected_families = {
            "unet",
            "resunet",
            "attention_unet",
            "unet_plus_plus",
            "deeplabv3_plus",
            "transformer_unet",
        }
        seen = set()

        for path in sorted(smoke_dir.glob("*.yaml")):
            config = load_config(path)
            seen.add(config["model"]["family"])
            self.assertEqual(config["training"]["max_epochs"], 1)
            self.assertLessEqual(config["training"]["batch_size"], 2)
            self.assertLessEqual(config["cloud"]["max_train_samples"], 8)
            self.assertLessEqual(config["cloud"]["max_validation_samples"], 4)
            self.assertEqual(config["split_policy"]["test"], "sealed")
            self.assertNotIn("test", config["split_policy"]["selection_splits"])

        self.assertEqual(seen, expected_families)

    def test_unet_channel_ablation_configs_validate_channel_contracts(self):
        config_dir = Path(__file__).resolve().parents[1] / "configs" / "segmentation_training"
        ablation_dir = config_dir / "unet_channel_ablation"
        expected_channels = {
            "c5_unet_rgb_optical_smoke.yaml": ("F03", "F02", "F01"),
            "c5_unet_false_color_smoke.yaml": ("F07", "F03", "F02"),
            "c5_unet_post_optical_7ch_smoke.yaml": ("F01", "F02", "F03", "F04", "F07", "F11", "F12"),
            "c5_unet_indices_only_smoke.yaml": ("NDVI", "NBR", "NDMI", "BRIGHTNESS"),
            "c5_unet_11ch_indices_smoke.yaml": (
                "F01",
                "F02",
                "F03",
                "F04",
                "F07",
                "F11",
                "F12",
                "NDVI",
                "NBR",
                "NDMI",
                "BRIGHTNESS",
            ),
        }

        for smoke_name, channels in expected_channels.items():
            full_name = smoke_name.replace("_smoke.yaml", "_rtx4080.yaml")
            with self.subTest(filename=smoke_name):
                smoke = load_config(ablation_dir / smoke_name)
                self.assertEqual(tuple(smoke["input_channels"]), channels)
                self.assertEqual(smoke["model"]["family"], "unet")
                self.assertEqual(smoke["model"]["input_channels"], len(channels))
                self.assertEqual(smoke["training"]["max_epochs"], 1)
                self.assertLessEqual(smoke["training"]["batch_size"], 2)
                self.assertLessEqual(smoke["cloud"]["max_train_samples"], 8)
                self.assertLessEqual(smoke["cloud"]["max_validation_samples"], 4)
                self.assertEqual(smoke["split_policy"]["test"], "sealed")
                self.assertNotIn("test", smoke["split_policy"]["selection_splits"])

            with self.subTest(filename=full_name):
                full = load_config(ablation_dir / full_name)
                self.assertEqual(tuple(full["input_channels"]), channels)
                self.assertEqual(full["model"]["family"], "unet")
                self.assertEqual(full["model"]["input_channels"], len(channels))
                self.assertEqual(full["training"]["max_epochs"], 80)
                self.assertEqual(full["cloud"]["expected_accelerator"], "cuda")
                self.assertEqual(full["split_policy"]["test"], "sealed")
                self.assertNotIn("test", full["split_policy"]["selection_splits"])

    def test_channel_mapping_probe_configs_validate_post_disaster_single_phase_contracts(self):
        probe_dir = Path(__file__).resolve().parents[1] / "configs" / "segmentation_training" / "channel_mapping_probe"
        expected_channels = {
            "c5_deeplabv3plus_s2_corrected_11ch_smoke.yaml": 11,
            "c5_deeplabv3plus_s2_corrected_11ch_rtx4080.yaml": 11,
            "c5_deeplabv3plus_s2_water_burn_15ch_smoke.yaml": 15,
            "c5_deeplabv3plus_s2_water_burn_15ch_rtx4080.yaml": 15,
            "c5_deeplabv3plus_s2_rededge_bais2_19ch_smoke.yaml": 19,
            "c5_deeplabv3plus_s2_rededge_bais2_19ch_rtx4080.yaml": 19,
        }

        for filename, channel_count in expected_channels.items():
            with self.subTest(filename=filename):
                config = load_config(probe_dir / filename)
                self.assertEqual(config["model"]["family"], "deeplabv3_plus")
                self.assertEqual(config["model"]["input_channels"], channel_count)
                self.assertEqual(len(config["input_channels"]), channel_count)
                self.assertEqual(config["split_policy"]["test"], "sealed")
                self.assertNotIn("test", config["split_policy"]["selection_splits"])

    def test_p14_fine_tune_configs_resume_from_p13_best_checkpoint(self):
        probe_dir = Path(__file__).resolve().parents[1] / "configs" / "segmentation_training" / "p14_15ch_finetune"
        expected = {
            "c5_deeplabv3plus_15ch_base32_ft_lr1e4_smoke.yaml": (1, 0.0001),
            "c5_deeplabv3plus_15ch_base32_ft_lr5e5_smoke.yaml": (1, 0.00005),
            "c5_deeplabv3plus_15ch_base32_ft_lr1e4_80ep.yaml": (80, 0.0001),
            "c5_deeplabv3plus_15ch_base32_ft_lr5e5_80ep.yaml": (80, 0.00005),
        }

        for filename, (max_epochs, learning_rate) in expected.items():
            with self.subTest(filename=filename):
                config = load_config(probe_dir / filename)
                self.assertEqual(config["model"]["family"], "deeplabv3_plus")
                self.assertEqual(config["model"]["base_channels"], 32)
                self.assertEqual(config["model"]["input_channels"], 15)
                self.assertEqual(len(config["input_channels"]), 15)
                self.assertEqual(config["training"]["max_epochs"], max_epochs)
                self.assertEqual(config["training"]["batch_size"], 8)
                self.assertEqual(config["training"]["learning_rate"], learning_rate)
                self.assertEqual(config["training"]["resume_from_checkpoint"], "${P13_BEST_CHECKPOINT}")
                self.assertEqual(config["split_policy"]["test"], "sealed")
                self.assertNotIn("test", config["split_policy"]["selection_splits"])
                self.assertIn("diagnostics/p14_hard_window_gate.csv", config["metrics"]["required_outputs"])

    def test_resume_from_checkpoint_config_value_must_be_non_empty_when_present(self):
        config = load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "segmentation_training"
            / "channel_mapping_probe"
            / "c5_deeplabv3plus_s2_water_burn_15ch_smoke.yaml"
        )
        config["training"]["resume_from_checkpoint"] = " "

        with self.assertRaisesRegex(ValueError, "resume_from_checkpoint"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
