import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation_training.losses import build_segmentation_loss, class_weights_from_counts  # noqa: E402


class SegmentationTrainingLossTests(unittest.TestCase):
    def test_class_weights_from_counts_upweights_sparse_foreground_with_clipping(self):
        weights = class_weights_from_counts([1000, 10], max_weight=20.0)

        self.assertEqual(weights[0], 1.0)
        self.assertGreater(weights[1], weights[0])
        self.assertLessEqual(weights[1], 20.0)

    def test_class_weights_keep_missing_foreground_finite(self):
        weights = class_weights_from_counts([1000, 0], max_weight=20.0)

        self.assertEqual(weights, [1.0, 20.0])

    def test_weighted_cross_entropy_dice_builds_finite_loss(self):
        import torch

        loss_fn = build_segmentation_loss(
            loss_name="weighted_cross_entropy_dice",
            ignore_index=255,
            class_weights=[1.0, 3.0],
            dice_weight=1.0,
        )
        logits = torch.randn(1, 2, 4, 4)
        labels = torch.zeros((1, 4, 4), dtype=torch.long)
        labels[:, 1:3, 1:3] = 1

        loss = loss_fn(logits, labels)

        self.assertTrue(bool(torch.isfinite(loss).item()))


if __name__ == "__main__":
    unittest.main()
