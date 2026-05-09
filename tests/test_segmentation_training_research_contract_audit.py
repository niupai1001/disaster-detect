import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation_training.cli import main as training_main  # noqa: E402
from segmentation_training.research_contract_audit import AuditSource, build_research_contract_audit  # noqa: E402


class SegmentationTrainingResearchContractAuditTests(unittest.TestCase):
    def write_manifest(self, root: Path, name: str, fieldnames: list[str], rows: list[dict[str, str]]) -> Path:
        path = root / name / "manifests" / "cloud_model_input_manifest.csv"
        path.parent.mkdir(parents=True)
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_audit_blocks_when_research_columns_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            fieldnames = [
                "sample_id",
                "event_id",
                "class_id",
                "class_name",
                "split",
                "mask_path",
                "input_channels",
                "input_band_paths",
            ]
            c2 = self.write_manifest(
                root,
                "c2",
                fieldnames,
                [
                    {
                        "sample_id": "c2-a",
                        "event_id": "e1",
                        "class_id": "1",
                        "class_name": "C2_landslide",
                        "split": "train",
                        "mask_path": "masks/c2-a.tif",
                        "input_channels": "B;G",
                        "input_band_paths": "B:database/b.tif;G:database/g.tif",
                    }
                ],
            )
            c5 = self.write_manifest(
                root,
                "c5",
                fieldnames,
                [
                    {
                        "sample_id": "c5-a",
                        "event_id": "e2",
                        "class_id": "2",
                        "class_name": "C5_fire",
                        "split": "validation",
                        "mask_path": "masks/c5-a.tif",
                        "input_channels": "F01;F02",
                        "input_band_paths": "F01:database/f01.tif;F02:database/f02.tif",
                    }
                ],
            )

            audit = build_research_contract_audit(
                [
                    AuditSource("c2", c2, c2.parents[1], ("B", "G")),
                    AuditSource("c5", c5, c5.parents[1], ("F01", "F02")),
                ],
                output_dir=root / "audit",
            )

            self.assertEqual(audit["status"], "blocked")
            self.assertEqual(audit["total_records"], 2)
            self.assertEqual(audit["split_disaster_counts"]["train"]["C2"], 1)
            self.assertEqual(audit["split_disaster_counts"]["validation"]["C5"], 1)
            self.assertEqual(audit["missing_required_fields"]["disaster_id"], 2)
            self.assertEqual(audit["missing_required_fields"]["label_confidence"], 2)
            self.assertEqual(audit["missing_required_fields"]["ignore_mask_path"], 2)
            self.assertGreater(audit["contract_error_count"], 0)
            self.assertTrue((root / "audit" / "audit.json").exists())
            self.assertTrue((root / "audit" / "audit.md").exists())
            self.assertTrue((root / "audit" / "contract_errors.csv").exists())

    def test_audit_passes_when_research_contract_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            (root / "c2" / "masks").mkdir(parents=True)
            (root / "c2" / "masks" / "ignore.tif").write_bytes(b"ignore")
            fieldnames = [
                "sample_id",
                "event_id",
                "class_id",
                "class_name",
                "split",
                "mask_path",
                "input_channels",
                "input_band_paths",
                "disaster_id",
                "label_confidence",
                "ignore_mask_path",
            ]
            c2 = self.write_manifest(
                root,
                "c2",
                fieldnames,
                [
                    {
                        "sample_id": "c2-a",
                        "event_id": "e1",
                        "class_id": "1",
                        "class_name": "C2_landslide",
                        "split": "train",
                        "mask_path": "masks/c2-a.tif",
                        "input_channels": "B;G",
                        "input_band_paths": "B:database/b.tif;G:database/g.tif",
                        "disaster_id": "C2",
                        "label_confidence": "medium",
                        "ignore_mask_path": "masks/ignore.tif",
                    }
                ],
            )
            c5 = self.write_manifest(
                root,
                "c5",
                fieldnames,
                [
                    {
                        "sample_id": "c5-a",
                        "event_id": "e2",
                        "class_id": "2",
                        "class_name": "C5_fire",
                        "split": "validation",
                        "mask_path": "masks/c5-a.tif",
                        "input_channels": "F01;F02",
                        "input_band_paths": "F01:database/f01.tif;F02:database/f02.tif",
                        "disaster_id": "C5",
                        "label_confidence": "0.8",
                        "ignore_mask_path": "",
                    }
                ],
            )

            audit = build_research_contract_audit(
                [
                    AuditSource("c2", c2, c2.parents[1], ("B", "G")),
                    AuditSource("c5", c5, c5.parents[1], ("F01", "F02")),
                ],
                output_dir=root / "audit",
            )
            written = json.loads((root / "audit" / "audit.json").read_text(encoding="utf-8"))

            self.assertEqual(audit["status"], "ready_for_e1")
            self.assertEqual(audit["contract_error_count"], 0)
            self.assertEqual(written["label_confidence_counts"]["medium"], 1)
            self.assertEqual(written["label_confidence_counts"]["numeric"], 1)
            self.assertEqual(written["ignore_mask_declared_count"], 1)

    def test_research_contract_audit_cli_accepts_bundle_sources(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            c2 = self.write_manifest(
                root,
                "c2_bundle",
                [
                    "sample_id",
                    "event_id",
                    "class_id",
                    "class_name",
                    "split",
                    "mask_path",
                    "input_channels",
                    "input_band_paths",
                ],
                [
                    {
                        "sample_id": "c2-a",
                        "event_id": "e1",
                        "class_id": "1",
                        "class_name": "C2_landslide",
                        "split": "train",
                        "mask_path": "masks/c2-a.tif",
                        "input_channels": "B;G",
                        "input_band_paths": "B:database/b.tif;G:database/g.tif",
                    }
                ],
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = training_main(
                    [
                        "research-contract-audit",
                        "--bundle-source",
                        f"c2|{c2.parents[1]}|B,G",
                        "--required-disaster",
                        "C2",
                        "--output-dir",
                        str(root / "audit"),
                    ]
                )

            self.assertEqual(exit_code, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["status"], "blocked")
            self.assertEqual(summary["total_records"], 1)


if __name__ == "__main__":
    unittest.main()
