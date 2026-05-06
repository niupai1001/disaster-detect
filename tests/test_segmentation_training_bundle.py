import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation_training.bundle import build_training_bundle  # noqa: E402
from segmentation_training.manifest import load_model_input_manifest, validate_record_paths  # noqa: E402


class SegmentationTrainingBundleTests(unittest.TestCase):
    def make_contract(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        contract = root / "contract"
        (contract / "masks").mkdir(parents=True)
        (contract / "reports").mkdir()
        (contract / "masks" / "s1.tif").write_bytes(b"mask")
        (contract / "class_map.json").write_text(json.dumps({"1": {"name": "C2"}}), encoding="utf-8")
        (contract / "channels.json").write_text(json.dumps({"channels": [{"name": "F16"}, {"name": "F17"}]}), encoding="utf-8")
        (contract / "reports" / "channel_alignment_report.csv").write_text("status\nok\n", encoding="utf-8")
        with (contract / "model_input_manifest.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "sample_id",
                    "event_id",
                    "class_id",
                    "class_name",
                    "split",
                    "mask_path",
                    "input_channels",
                    "input_band_paths",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "sample_id": "s1",
                    "event_id": "e1",
                    "class_id": "1",
                    "class_name": "C2_debris_flow",
                    "split": "train",
                    "mask_path": "masks/s1.tif",
                    "input_channels": "F16;F17",
                    "input_band_paths": "F16:database/a.tif;F17:database/b.tif",
                }
            )
        return root, contract

    def test_manifest_only_bundle_writes_expected_files(self):
        root, contract = self.make_contract()
        bundle_dir = root / "bundle"

        manifest = build_training_bundle(
            contract_dir=contract,
            bundle_dir=bundle_dir,
            mode="manifest-only",
            cloud_data_root=Path("/cloud/training_bundle"),
        )

        self.assertEqual(manifest["record_count"], 1)
        self.assertTrue((bundle_dir / "metadata" / "class_map.json").exists())
        self.assertTrue((bundle_dir / "metadata" / "channels.json").exists())
        self.assertTrue((bundle_dir / "metadata" / "bundle_manifest.json").exists())
        self.assertTrue((bundle_dir / "manifests" / "cloud_model_input_manifest.csv").exists())
        self.assertTrue((bundle_dir / "masks" / "s1.tif").exists())
        cloud_manifest = (bundle_dir / "manifests" / "cloud_model_input_manifest.csv").read_text(encoding="utf-8")
        self.assertIn("/cloud/training_bundle/masks/s1.tif", cloud_manifest)

    def test_local_bundle_validation_accepts_rebased_cloud_mask_paths(self):
        root, contract = self.make_contract()
        bundle_dir = root / "bundle"
        build_training_bundle(
            contract_dir=contract,
            bundle_dir=bundle_dir,
            mode="manifest-only",
            cloud_data_root=Path("/cloud/training_bundle"),
        )

        records = load_model_input_manifest(bundle_dir / "manifests" / "cloud_model_input_manifest.csv")
        errors = validate_record_paths(records, contract_dir=bundle_dir, require_bands=False)

        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
