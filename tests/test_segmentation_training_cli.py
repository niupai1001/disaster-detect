from __future__ import annotations

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
            self.assertEqual(
                visible_files,
                {
                    "review.md",
                    "compact_metrics.csv",
                    "best_checkpoint_metrics.csv",
                    "threshold_operating_points.csv",
                    "validation_window_diagnostics.csv",
                    "hard_case_windows.csv",
                    "raw_evidence.md",
                },
            )

    def test_summarize_runs_adds_best_and_threshold_operating_point_tables(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            runs_dir = root / "runs"
            self._write_run(
                runs_dir / "run-final",
                experiment_id="exp-final",
                final_mean_iou=0.41,
                final_recall=0.88,
                best_epoch=8,
                best_mean_iou=0.41,
                foreground_precision=0.44,
                foreground_dice=0.58,
                label_pixels=100,
                predicted_pixels=200,
                threshold_rows=[
                    {
                        "threshold": "0.50",
                        "iou": "0.40",
                        "dice": "0.52",
                        "precision": "0.47",
                        "recall": "0.82",
                        "predicted_label_area_ratio": "3.2",
                    },
                    {
                        "threshold": "0.75",
                        "iou": "0.36",
                        "dice": "0.48",
                        "precision": "0.55",
                        "recall": "0.55",
                        "predicted_label_area_ratio": "1.5",
                    },
                ],
            )
            self._write_run(
                runs_dir / "run-best",
                experiment_id="exp-best",
                final_mean_iou=0.21,
                final_recall=0.87,
                best_epoch=7,
                best_mean_iou=0.44,
                foreground_precision=0.22,
                foreground_dice=0.35,
                label_pixels=100,
                predicted_pixels=400,
                threshold_rows=[
                    {
                        "threshold": "0.55",
                        "iou": "0.45",
                        "dice": "0.57",
                        "precision": "0.53",
                        "recall": "0.82",
                        "predicted_label_area_ratio": "11.8",
                    },
                    {
                        "threshold": "0.75",
                        "iou": "0.39",
                        "dice": "0.52",
                        "precision": "0.62",
                        "recall": "0.57",
                        "predicted_label_area_ratio": "1.4",
                    },
                ],
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
            review_text = (review_dir / "review.md").read_text(encoding="utf-8")
            self.assertIn("Ranked By Best Checkpoint", review_text)
            self.assertIn("Threshold-Calibrated Operating Points", review_text)
            self.assertIn("Area ratio band", review_text)

            best_rows = self._read_csv(review_dir / "best_checkpoint_metrics.csv")
            self.assertEqual([row["run"] for row in best_rows], ["run-best", "run-final"])

            threshold_rows = self._read_csv(review_dir / "threshold_operating_points.csv")
            self.assertEqual([row["run"] for row in threshold_rows], ["run-best", "run-final"])
            self.assertEqual(threshold_rows[0]["best_threshold"], "0.75")
            self.assertEqual(threshold_rows[0]["area_ratio_band"], "acceptable")
            self.assertEqual(threshold_rows[0]["selection_scope"], "validation_only")

    def test_summarize_runs_adds_validation_window_hard_case_tables(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            runs_dir = root / "runs"
            self._write_run(
                runs_dir / "run-a",
                experiment_id="exp-a",
                final_mean_iou=0.38,
                final_recall=0.75,
                best_epoch=10,
                best_mean_iou=0.41,
                foreground_precision=0.44,
                foreground_dice=0.55,
                label_pixels=100,
                predicted_pixels=160,
                validation_rows=[
                    {
                        "sample_id": "a-good",
                        "foreground_iou": "0.62",
                        "foreground_precision": "0.70",
                        "foreground_recall": "0.80",
                        "label_foreground_pixels": "100",
                        "predicted_foreground_pixels": "110",
                        "predicted_label_area_ratio": "1.1",
                        "all_background_prediction": "False",
                    },
                    {
                        "sample_id": "a-over",
                        "foreground_iou": "0.08",
                        "foreground_precision": "0.05",
                        "foreground_recall": "0.90",
                        "label_foreground_pixels": "100",
                        "predicted_foreground_pixels": "900",
                        "predicted_label_area_ratio": "9.0",
                        "all_background_prediction": "False",
                    },
                ],
            )
            self._write_run(
                runs_dir / "run-b",
                experiment_id="exp-b",
                final_mean_iou=0.34,
                final_recall=0.70,
                best_epoch=8,
                best_mean_iou=0.37,
                foreground_precision=0.38,
                foreground_dice=0.50,
                label_pixels=100,
                predicted_pixels=140,
                validation_rows=[
                    {
                        "sample_id": "b-good",
                        "foreground_iou": "0.50",
                        "foreground_precision": "0.58",
                        "foreground_recall": "0.72",
                        "label_foreground_pixels": "100",
                        "predicted_foreground_pixels": "100",
                        "predicted_label_area_ratio": "1.0",
                        "all_background_prediction": "False",
                    },
                    {
                        "sample_id": "b-zero",
                        "foreground_iou": "0.0",
                        "foreground_precision": "0.0",
                        "foreground_recall": "0.0",
                        "label_foreground_pixels": "100",
                        "predicted_foreground_pixels": "0",
                        "predicted_label_area_ratio": "0.0",
                        "all_background_prediction": "True",
                    },
                ],
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
            review_text = (review_dir / "review.md").read_text(encoding="utf-8")
            self.assertIn("Validation Window Diagnostics", review_text)
            self.assertIn("hard_case_windows.csv", review_text)

            diagnostics_rows = self._read_csv(review_dir / "validation_window_diagnostics.csv")
            self.assertEqual([row["run"] for row in diagnostics_rows], ["run-a", "run-b"])
            self.assertEqual(diagnostics_rows[0]["window_count"], "2")
            self.assertEqual(diagnostics_rows[0]["overexpanded_windows"], "1")
            self.assertEqual(diagnostics_rows[1]["zero_iou_windows"], "1")
            self.assertEqual(diagnostics_rows[1]["underpredicted_windows"], "1")

            hard_case_rows = self._read_csv(review_dir / "hard_case_windows.csv")
            self.assertEqual(hard_case_rows[0]["run"], "run-b")
            self.assertEqual(hard_case_rows[0]["sample_id"], "b-zero")
            self.assertIn("zero_iou", hard_case_rows[0]["issue_flags"])
            self.assertIn("all_background", hard_case_rows[0]["issue_flags"])

    def test_diagnostic_audit_writes_validation_only_review_pack(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            full_root = root / "full"
            benchmark_root = root / "benchmark"
            ablation_root = root / "ablation"
            self._write_diagnostic_run(
                full_root / "v6",
                run="v6",
                experiment_id="v6-11ch",
                model_family="unet",
                mean_iou=0.11,
                precision=0.12,
                recall=0.81,
                dice=0.20,
                area_ratio=7.0,
                test_split_read=False,
                threshold_rows=[
                    {"threshold": "0.50", "foreground_iou": "0.11", "foreground_precision": "0.12"},
                    {"threshold": "0.65", "foreground_iou": "0.14", "foreground_precision": "0.21"},
                ],
                worst_fp_rows=[
                    {"sample_id": "a", "false_positive_pixels": "42", "foreground_pixels": "1"},
                    {"sample_id": "b", "false_positive_pixels": "21", "foreground_pixels": "0"},
                ],
            )
            self._write_compact_metrics(
                full_root / "review" / "compact_metrics.csv",
                [
                    {
                        "run": "v6",
                        "experiment_id": "v6-11ch",
                        "model_family": "unet",
                        "best_epoch": "8",
                        "best_mean_iou": "0.25",
                        "mean_iou": "0.11",
                        "foreground_dice": "0.20",
                        "foreground_precision": "0.12",
                        "foreground_recall": "0.81",
                        "pred_label_area_ratio": "7.0",
                        "test_split_read": "False",
                        "run_dir": "workspace/runs/stale/v6",
                    }
                ],
            )
            self._write_diagnostic_run(
                benchmark_root / "deeplab",
                run="deeplab",
                experiment_id="deeplab",
                model_family="deeplabv3plus",
                mean_iou=0.40,
                precision=0.51,
                recall=0.63,
                dice=0.55,
                area_ratio=1.23,
                test_split_read=False,
                threshold_rows=[{"threshold": "0.50", "foreground_iou": "0.39", "foreground_precision": "0.51"}],
                worst_fp_rows=[],
            )
            self._write_compact_metrics(
                benchmark_root / "review" / "compact_metrics.csv",
                [
                    {
                        "run": "deeplab",
                        "experiment_id": "deeplab",
                        "model_family": "deeplabv3plus",
                        "best_epoch": "10",
                        "best_mean_iou": "0.41",
                        "mean_iou": "0.40",
                        "foreground_dice": "0.55",
                        "foreground_precision": "0.51",
                        "foreground_recall": "0.63",
                        "pred_label_area_ratio": "1.23",
                        "test_split_read": "False",
                        "run_dir": str(benchmark_root / "deeplab"),
                    }
                ],
            )
            self._write_diagnostic_run(
                ablation_root / "rgb",
                run="rgb",
                experiment_id="rgb",
                model_family="unet",
                mean_iou=0.29,
                precision=0.34,
                recall=0.66,
                dice=0.44,
                area_ratio=1.94,
                test_split_read=False,
                threshold_rows=[{"threshold": "0.50", "foreground_iou": "0.29", "foreground_precision": "0.34"}],
                worst_fp_rows=[],
            )
            self._write_compact_metrics(
                ablation_root / "review" / "compact_metrics.csv",
                [
                    {
                        "run": "rgb",
                        "experiment_id": "rgb",
                        "model_family": "unet",
                        "best_epoch": "5",
                        "best_mean_iou": "0.33",
                        "mean_iou": "0.29",
                        "foreground_dice": "0.44",
                        "foreground_precision": "0.34",
                        "foreground_recall": "0.66",
                        "pred_label_area_ratio": "1.94",
                        "test_split_read": "False",
                        "run_dir": str(ablation_root / "rgb"),
                    }
                ],
            )

            output_dir = root / "audit"
            exit_code = training_main(
                [
                    "diagnostic-audit",
                    "--run-root",
                    f"full={full_root}",
                    "--run-root",
                    f"benchmark={benchmark_root}",
                    "--run-root",
                    f"ablation={ablation_root}",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            expected_files = {
                "validation_operating_point_audit.md",
                "validation_operating_point_audit.csv",
                "threshold_aggregate_by_run.csv",
                "candidate_ranking_by_protocol.csv",
                "sealed_split_guard_report.md",
                "hard_negative_inventory.md",
                "hard_negative_inventory.csv",
                "hard_negative_taxonomy.yaml",
                "channel_normalization_audit.md",
                "controlled_probe_protocol.md",
                "controlled_probe_matrix.csv",
                "probe_acceptance_criteria.md",
                "implementation_scope_guard.md",
            }
            self.assertTrue(expected_files.issubset({path.name for path in output_dir.iterdir()}))

            guard = (output_dir / "sealed_split_guard_report.md").read_text(encoding="utf-8")
            self.assertIn("PASS", guard)
            self.assertIn("No sealed test reads were found", guard)

            ranking_rows = self._read_csv(output_dir / "candidate_ranking_by_protocol.csv")
            self.assertEqual(ranking_rows[0]["run"], "deeplab")
            self.assertEqual(ranking_rows[0]["area_ratio_band"], "acceptable")
            self.assertEqual(ranking_rows[-1]["area_ratio_band"], "blocked_overexpansion")

            threshold_rows = self._read_csv(output_dir / "threshold_aggregate_by_run.csv")
            self.assertEqual({row["run"] for row in threshold_rows}, {"v6", "deeplab", "rgb"})

            hard_negative = (output_dir / "hard_negative_inventory.md").read_text(encoding="utf-8")
            self.assertIn("validation evidence only", hard_negative)
            self.assertIn("must not convert validation examples into training candidates", hard_negative)
            hard_negative_rows = self._read_csv(output_dir / "hard_negative_inventory.csv")
            self.assertEqual([row["sample_id"] for row in hard_negative_rows], ["a", "b"])

            probe = (output_dir / "controlled_probe_protocol.md").read_text(encoding="utf-8")
            self.assertIn("Explicitly deferred", probe)
            self.assertIn("train", probe)
            self.assertIn("diagnostic artifacts pass QA", probe)

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
        threshold_rows: list[dict] | None = None,
        validation_rows: list[dict] | None = None,
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
        if threshold_rows is not None:
            diagnostics = run_dir / "diagnostics"
            diagnostics.mkdir(exist_ok=True)
            self._write_csv(diagnostics / "threshold_sweep.csv", threshold_rows)
        if validation_rows is not None:
            diagnostics = run_dir / "diagnostics"
            diagnostics.mkdir(exist_ok=True)
            self._write_csv(diagnostics / "validation_prediction_summary.csv", validation_rows)

    def _write_diagnostic_run(
        self,
        run_dir: Path,
        *,
        run: str,
        experiment_id: str,
        model_family: str,
        mean_iou: float,
        precision: float,
        recall: float,
        dice: float,
        area_ratio: float,
        test_split_read: bool,
        threshold_rows: list[dict],
        worst_fp_rows: list[dict],
    ) -> None:
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text(
            json.dumps({"mean_iou": mean_iou, "foreground_recall": recall}),
            encoding="utf-8",
        )
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "experiment_id": experiment_id,
                    "model": {"family": model_family},
                    "test_split_read": test_split_read,
                }
            ),
            encoding="utf-8",
        )
        with (run_dir / "per_class_metrics.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["class_id", "precision", "recall", "dice", "iou"])
            writer.writeheader()
            writer.writerow({"class_id": "2", "precision": precision, "recall": recall, "dice": dice, "iou": mean_iou})
        with (run_dir / "mask_area_summary.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["class_id", "label_pixels", "predicted_pixels"])
            writer.writeheader()
            writer.writerow({"class_id": "2", "label_pixels": 100, "predicted_pixels": int(area_ratio * 100)})
        diagnostics = run_dir / "diagnostics"
        diagnostics.mkdir()
        self._write_csv(diagnostics / "threshold_sweep.csv", threshold_rows)
        self._write_csv(diagnostics / "worst_false_positives.csv", worst_fp_rows)
        self._write_csv(
            diagnostics / "foreground_probability_summary.csv",
            [{"mean_probability": "0.42", "p95_probability": "0.88"}],
        )
        self._write_csv(
            diagnostics / "foreground_window_coverage.csv",
            [{"split": "validation", "foreground_pixels": "12"}],
        )
        (run_dir / "predictions" / "worst_false_positives").mkdir(parents=True)
        (run_dir / "predictions" / "worst_false_positives" / f"{run}.png").write_bytes(b"fake png")

    def _write_compact_metrics(self, path: Path, rows: list[dict]) -> None:
        fieldnames = [
            "run",
            "experiment_id",
            "model_family",
            "best_epoch",
            "best_mean_iou",
            "mean_iou",
            "foreground_dice",
            "foreground_precision",
            "foreground_recall",
            "pred_label_area_ratio",
            "test_split_read",
            "run_dir",
        ]
        path.parent.mkdir(parents=True)
        self._write_csv(path, rows, fieldnames=fieldnames)

    def _write_csv(self, path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if fieldnames is None:
            fieldnames = sorted({key for row in rows for key in row}) if rows else ["empty"]
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _read_csv(self, path: Path) -> list[dict]:
        with path.open(encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))


if __name__ == "__main__":
    unittest.main()
