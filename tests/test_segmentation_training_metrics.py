import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation_training.metrics import compute_confusion_matrix, per_class_metrics, summarize_area  # noqa: E402


class SegmentationTrainingMetricsTests(unittest.TestCase):
    def test_ignore_pixels_do_not_affect_confusion_or_area(self):
        labels = np.array([[0, 1, 1], [2, 2, 255]], dtype=np.uint8)
        predictions = np.array([[0, 1, 2], [2, 0, 1]], dtype=np.uint8)

        matrix = compute_confusion_matrix(labels, predictions, class_ids=[0, 1, 2], ignore_index=255)
        metrics = per_class_metrics(matrix, class_ids=[0, 1, 2])
        area = summarize_area(labels, predictions, class_ids=[0, 1, 2], ignore_index=255)

        self.assertEqual(matrix.tolist(), [[1, 0, 0], [0, 1, 1], [1, 0, 1]])
        self.assertAlmostEqual(metrics[1].iou, 0.5)
        self.assertAlmostEqual(metrics[2].dice, 0.5)
        self.assertEqual(area[1]["label_pixels"], 2)
        self.assertEqual(area[1]["predicted_pixels"], 1)


if __name__ == "__main__":
    unittest.main()

