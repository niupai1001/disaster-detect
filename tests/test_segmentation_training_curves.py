import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation_training.cli import main as training_main  # noqa: E402
from segmentation_training.training_curves import parse_training_progress  # noqa: E402


class SegmentationTrainingCurvesTests(unittest.TestCase):
    def test_parse_training_progress_extracts_batch_and_validation_points(self):
        text = "\n".join(
            [
                "[2026-05-06T12:56:45+00:00] epoch 60/60 batch 50/69 (72.5%) loss=0.0548 avg_loss=0.0745",
                "[2026-05-06T12:56:48+00:00] epoch 60/60 validation loss=0.0769 mean_iou=0.0000 foreground_recall=0.0000",
            ]
        )

        parsed = parse_training_progress(text)

        self.assertEqual(parsed.batch_rows[0]["epoch"], 60)
        self.assertEqual(parsed.batch_rows[0]["batch"], 50)
        self.assertAlmostEqual(parsed.batch_rows[0]["loss"], 0.0548)
        self.assertEqual(parsed.validation_rows[0]["epoch"], 60)
        self.assertEqual(parsed.validation_rows[0]["foreground_recall"], 0.0)

    def test_plot_run_cli_writes_png_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            run_dir = root / "run"
            (run_dir / "logs").mkdir(parents=True)
            (run_dir / "logs" / "training_progress.log").write_text(
                "\n".join(
                    [
                        "[2026-05-06T12:00:00+00:00] epoch 1/2 batch 1/2 (50.0%) loss=0.5 avg_loss=0.5",
                        "[2026-05-06T12:00:01+00:00] epoch 1/2 batch 2/2 (100.0%) loss=0.4 avg_loss=0.45",
                        "[2026-05-06T12:00:02+00:00] epoch 1/2 validation loss=0.45 mean_iou=0.1 foreground_recall=0.2",
                    ]
                ),
                encoding="utf-8",
            )

            exit_code = training_main(["plot-run", "--run-dir", str(run_dir)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((run_dir / "training_curves.png").exists())
            self.assertTrue((run_dir / "training_curves.csv").exists())


if __name__ == "__main__":
    unittest.main()
