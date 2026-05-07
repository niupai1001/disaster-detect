import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation_training.models import build_model  # noqa: E402


class SegmentationTrainingModelTests(unittest.TestCase):
    def test_registry_builds_all_architecture_families_for_11_channel_inputs(self):
        import torch

        families = [
            "unet",
            "resunet",
            "attention_unet",
            "unet_plus_plus",
            "deeplabv3_plus",
            "transformer_unet",
        ]

        for family in families:
            with self.subTest(family=family):
                model = build_model(
                    {"family": family, "base_channels": 4, "transformer_layers": 1, "num_heads": 2},
                    input_channels=11,
                    output_classes=2,
                )
                model.eval()
                with torch.no_grad():
                    logits = model(torch.zeros(2, 11, 64, 64))

                self.assertEqual(tuple(logits.shape), (2, 2, 64, 64))

    def test_architecture_families_preserve_training_window_shape(self):
        import torch

        families = [
            "unet",
            "resunet",
            "attention_unet",
            "unet_plus_plus",
            "deeplabv3_plus",
            "transformer_unet",
        ]

        for family in families:
            with self.subTest(family=family):
                model = build_model(
                    {
                        "family": family,
                        "base_channels": 2,
                        "transformer_layers": 1,
                        "num_heads": 2,
                        "token_grid": 4,
                    },
                    input_channels=11,
                    output_classes=2,
                )
                model.eval()
                with torch.no_grad():
                    logits = model(torch.zeros(1, 11, 256, 256))

                self.assertEqual(tuple(logits.shape), (1, 2, 256, 256))

    def test_registry_rejects_unknown_architecture_family(self):
        with self.assertRaisesRegex(ValueError, "Unsupported model family"):
            build_model({"family": "mystery"}, input_channels=11, output_classes=2)


if __name__ == "__main__":
    unittest.main()
