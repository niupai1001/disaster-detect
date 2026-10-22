import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation_training.sampler import (  # noqa: E402
    choose_training_window,
    foreground_window_coverage,
    window_foreground_stats,
)


class SegmentationTrainingSamplerTests(unittest.TestCase):
    def test_choose_training_window_can_bias_toward_foreground(self):
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[9:11, 9:11] = 2
        rng = np.random.default_rng(7)

        window = choose_training_window(
            mask,
            size=4,
            rng=rng,
            foreground_class_ids=[2],
            foreground_probability=1.0,
            min_foreground_pixels=1,
        )
        stats = window_foreground_stats(mask, window, foreground_class_ids=[2])

        self.assertGreaterEqual(stats["foreground_pixels"], 1)
        self.assertGreater(stats["foreground_fraction"], 0.0)

    def test_foreground_window_coverage_reports_positive_and_empty_windows(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[1:3, 1:3] = 2

        rows = foreground_window_coverage(
            [("positive", mask), ("empty", np.zeros((8, 8), dtype=np.uint8))],
            foreground_class_ids=[2],
            size=4,
        )

        self.assertEqual(rows[0]["sample_id"], "positive")
        self.assertGreater(rows[0]["full_mask_foreground_pixels"], 0)
        self.assertGreater(rows[0]["foreground_center_window_pixels"], 0)
        self.assertEqual(rows[1]["full_mask_foreground_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
