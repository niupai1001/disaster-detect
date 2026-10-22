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
        self.assertTrue((bundle_dir / "configs" / "v2_binary_c5_unet_post_optical.yaml").exists())
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

    def test_v2_bundle_records_arbitrary_required_channels_and_version(self):
        root, contract = self.make_contract()
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
                    "input_channels": "F01;F02",
                    "input_band_paths": "F01:database/f01.tif;F02:database/f02.tif",
                }
            )

        manifest = build_training_bundle(
            contract_dir=contract,
            bundle_dir=root / "bundle_v2",
            required_channels=("F01", "F02"),
            cloud_data_root=Path("/cloud/training_bundle_v0_2"),
            bundle_version="0.2",
        )

        self.assertEqual(manifest["bundle_version"], "0.2")
        self.assertEqual(manifest["required_channels"], ["F01", "F02"])
        cloud_manifest = (root / "bundle_v2" / "manifests" / "cloud_model_input_manifest.csv").read_text(
            encoding="utf-8"
        )
        self.assertIn("/cloud/training_bundle_v0_2/database/f01.tif", cloud_manifest)

    def test_bundle_copies_contract_scoped_model_inputs(self):
        root, contract = self.make_contract()
        model_inputs = contract / "model_inputs"
        model_inputs.mkdir()
        (model_inputs / "s1_F01.tif").write_bytes(b"f01")
        (model_inputs / "s1_F02.tif").write_bytes(b"f02")
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
                    "input_channels": "F01;F02",
                    "input_band_paths": (
                        f"F01:{model_inputs / 's1_F01.tif'};"
                        f"F02:{model_inputs / 's1_F02.tif'}"
                    ),
                }
            )

        build_training_bundle(
            contract_dir=contract,
            bundle_dir=root / "bundle_model_inputs",
            required_channels=("F01", "F02"),
            cloud_data_root=Path("/cloud/training_bundle_v0_2"),
            bundle_version="0.2",
        )

        self.assertTrue((root / "bundle_model_inputs" / "model_inputs" / "s1_F01.tif").exists())
        cloud_manifest = (root / "bundle_model_inputs" / "manifests" / "cloud_model_input_manifest.csv").read_text(
            encoding="utf-8"
        )
        self.assertIn("F01:model_inputs/s1_F01.tif", cloud_manifest)


if __name__ == "__main__":
    unittest.main()
