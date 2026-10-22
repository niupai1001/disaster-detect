import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation_training.metrics import (  # noqa: E402
    compute_confusion_matrix,
    decode_prediction,
    encode_label,
    per_class_metrics,
    prediction_summary_rows,
    summarize_area,
    threshold_sweep,
)


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

    def test_binary_c5_encode_decode_preserves_source_class_ids_and_ignore(self):
        label = np.array([[0, 2, 255], [2, 0, 1]], dtype=np.uint8)

        encoded = encode_label(label, [0, 2], ignore_index=255)
        decoded = decode_prediction(encoded, [0, 2])

        self.assertEqual(encoded.tolist(), [[0, 1, 255], [1, 0, 255]])
        self.assertEqual(decoded.tolist(), [[0, 2, 255], [2, 0, 255]])

    def test_prediction_summary_detects_all_background_binary_c5_failure(self):
        labels = [
            np.array([[0, 2], [2, 255]], dtype=np.uint8),
            np.array([[0, 0], [0, 2]], dtype=np.uint8),
        ]
        predictions = [
            np.array([[0, 0], [0, 0]], dtype=np.uint8),
            np.array([[0, 0], [0, 2]], dtype=np.uint8),
        ]

        rows = prediction_summary_rows(labels, predictions, sample_ids=["s1", "s2"], foreground_class_ids=[2])

        self.assertTrue(rows[0]["all_background_prediction"])
        self.assertEqual(rows[0]["label_foreground_pixels"], 2)
        self.assertEqual(rows[0]["predicted_foreground_pixels"], 0)
        self.assertEqual(rows[0]["foreground_recall"], 0.0)
        self.assertFalse(rows[1]["all_background_prediction"])
        self.assertEqual(rows[1]["predicted_label_area_ratio"], 1.0)

    def test_threshold_sweep_recovers_weak_foreground_probabilities(self):
        labels = np.array([[0, 2], [2, 255]], dtype=np.uint8)
        foreground_probability = np.array([[0.02, 0.35], [0.40, 0.99]], dtype=np.float32)

        rows = threshold_sweep(
            labels,
            foreground_probability,
            foreground_class_id=2,
            thresholds=[0.5, 0.3],
            ignore_index=255,
        )

        self.assertEqual(rows[0]["threshold"], 0.5)
        self.assertEqual(rows[0]["recall"], 0.0)
        self.assertEqual(rows[1]["threshold"], 0.3)
        self.assertEqual(rows[1]["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
