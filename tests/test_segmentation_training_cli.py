import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation_training.cli import main as training_main  # noqa: E402


class SegmentationTrainingCliTests(unittest.TestCase):
    def test_summarize_runs_creates_review_first_pack_by_default(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            runs_dir = root / "runs"
            self._write_run(
                runs_dir / "run-a",
                experiment_id="exp-a",
                final_mean_iou=0.21,
                final_recall=0.75,
                best_epoch=3,
                best_mean_iou=0.33,
                foreground_precision=0.4,
                foreground_dice=0.52,
                label_pixels=10,
                predicted_pixels=25,
            )
            self._write_run(
                runs_dir / "run-b",
                experiment_id="exp-b",
                final_mean_iou=0.42,
                final_recall=0.5,
                best_epoch=8,
                best_mean_iou=0.44,
                foreground_precision=0.6,
                foreground_dice=0.55,
                label_pixels=20,
                predicted_pixels=18,
            )

            exit_code = training_main(
                [
                    "summarize-runs",
                    "--runs-dir",
                    str(runs_dir),
                    "--output",
                    str(runs_dir / "summary.md"),
                ]
            )

            self.assertEqual(exit_code, 0)
            review_dir = runs_dir / "review"
            self.assertTrue((review_dir / "review.md").exists())
            self.assertTrue((review_dir / "compact_metrics.csv").exists())
            self.assertTrue((review_dir / "raw_evidence.md").exists())
            self.assertTrue((review_dir / "figures" / "run-a__training_curves.png").exists())
            self.assertTrue((review_dir / "figures" / "run-b__validation_contact_sheet.png").exists())

            review_text = (review_dir / "review.md").read_text(encoding="utf-8")
            self.assertIn("Review-First Training Summary", review_text)
            self.assertIn("run-b", review_text)
            self.assertLess(review_text.index("run-b"), review_text.index("run-a"))

            with (review_dir / "compact_metrics.csv").open(encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual([row["run"] for row in rows], ["run-b", "run-a"])
            self.assertEqual(rows[0]["best_epoch"], "8")
            self.assertEqual(rows[0]["best_mean_iou"], "0.44")
            self.assertEqual(rows[0]["foreground_precision"], "0.6")
            self.assertEqual(rows[0]["pred_label_area_ratio"], "0.9")

            visible_files = {path.name for path in review_dir.iterdir() if path.is_file()}
            self.assertEqual(visible_files, {"review.md", "compact_metrics.csv", "raw_evidence.md"})

    def _write_run(
        self,
        run_dir: Path,
        *,
        experiment_id: str,
        final_mean_iou: float,
        final_recall: float,
        best_epoch: int,
        best_mean_iou: float,
        foreground_precision: float,
        foreground_dice: float,
        label_pixels: int,
        predicted_pixels: int,
    ) -> None:
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text(
            json.dumps({"mean_iou": final_mean_iou, "foreground_recall": final_recall}),
            encoding="utf-8",
        )
        (run_dir / "run_manifest.json").write_text(
            json.dumps({"experiment_id": experiment_id, "model": {"family": "unet"}, "test_split_read": False}),
            encoding="utf-8",
        )
        with (run_dir / "per_class_metrics.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["class_id", "precision", "recall", "dice", "iou"])
            writer.writeheader()
            writer.writerow({"class_id": "0", "precision": 0.9, "recall": 0.9, "dice": 0.9, "iou": 0.8})
            writer.writerow(
                {
                    "class_id": "2",
                    "precision": foreground_precision,
                    "recall": final_recall,
                    "dice": foreground_dice,
                    "iou": final_mean_iou,
                }
            )
        with (run_dir / "mask_area_summary.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["class_id", "label_pixels", "predicted_pixels"])
            writer.writeheader()
            writer.writerow({"class_id": "0", "label_pixels": 100, "predicted_pixels": 90})
            writer.writerow({"class_id": "2", "label_pixels": label_pixels, "predicted_pixels": predicted_pixels})
        (run_dir / "checkpoints").mkdir()
        (run_dir / "checkpoints" / "best_mean_iou.json").write_text(
            json.dumps({"epoch": best_epoch, "metric_value": best_mean_iou, "mean_iou": best_mean_iou}),
            encoding="utf-8",
        )
        (run_dir / "predictions").mkdir()
        (run_dir / "predictions" / "validation_contact_sheet.png").write_bytes(b"fake png")
        (run_dir / "training_curves.png").write_bytes(b"fake png")


if __name__ == "__main__":
    unittest.main()
