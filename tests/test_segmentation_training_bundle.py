import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation_training.bundle import build_common_channel_bundle, build_training_bundle  # noqa: E402
from segmentation_training.manifest import load_model_input_manifest, validate_record_paths  # noqa: E402
from segmentation_training.cli import main as training_main  # noqa: E402


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
        records = load_model_input_manifest(bundle_dir / "manifests" / "cloud_model_input_manifest.csv")
        self.assertEqual(records[0].row["disaster_id"], "C2")
        self.assertEqual(records[0].row["label_confidence"], "unknown")
        self.assertEqual(records[0].row["ignore_mask_path"], "")

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

    def test_common_channel_bundle_merges_c2_and_c5_manifests_with_logical_channels(self):
        root, c2_bundle, c5_bundle = self.make_common_source_bundles()

        manifest = build_common_channel_bundle(
            c2_bundle_dir=c2_bundle,
            c5_bundle_dir=c5_bundle,
            output_bundle_dir=root / "merged",
            bundle_version="p15-test",
        )

        self.assertEqual(manifest["record_count"], 2)
        self.assertEqual(manifest["required_channels"], ["BLUE", "GREEN", "RED", "NIR"])
        records = load_model_input_manifest(
            root / "merged" / "manifests" / "cloud_model_input_manifest.csv",
            required_channels=("BLUE", "GREEN", "RED", "NIR"),
        )
        self.assertEqual([record.class_id for record in records], [1, 2])
        self.assertEqual([record.row["disaster_id"] for record in records], ["C2", "C5"])
        self.assertEqual([record.row["label_confidence"] for record in records], ["unknown", "unknown"])
        self.assertEqual([record.row["ignore_mask_path"] for record in records], ["", ""])
        self.assertEqual(records[0].input_channels, ("BLUE", "GREEN", "RED", "NIR"))
        self.assertEqual(records[0].input_band_paths["BLUE"], "/data/c2/a.tif#band=1")
        self.assertEqual(records[1].input_band_paths["NIR"], "/data/c5/f07.tif")

    def test_common_channel_bundle_cli_writes_merged_manifest(self):
        root, c2_bundle, c5_bundle = self.make_common_source_bundles()
        output_bundle = root / "merged_cli"

        exit_code = training_main(
            [
                "common-channel-bundle",
                "--c2-bundle-dir",
                str(c2_bundle),
                "--c5-bundle-dir",
                str(c5_bundle),
                "--bundle-dir",
                str(output_bundle),
                "--bundle-version",
                "p15-cli-test",
            ]
        )

        self.assertEqual(exit_code, 0)
        self.assertTrue((output_bundle / "manifests" / "cloud_model_input_manifest.csv").exists())
        manifest = json.loads((output_bundle / "metadata" / "bundle_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["bundle_version"], "p15-cli-test")

    def test_common_channel_bundle_interleaves_classes_within_each_split_for_smoke_sampling(self):
        root, c2_bundle, c5_bundle = self.make_common_source_bundles(c2_count=3, c5_count=3, split="validation")

        build_common_channel_bundle(
            c2_bundle_dir=c2_bundle,
            c5_bundle_dir=c5_bundle,
            output_bundle_dir=root / "merged_interleaved",
            bundle_version="p15-interleaved-test",
        )

        records = load_model_input_manifest(
            root / "merged_interleaved" / "manifests" / "cloud_model_input_manifest.csv",
            required_channels=("BLUE", "GREEN", "RED", "NIR"),
        )
        self.assertEqual([record.class_id for record in records[:6]], [1, 2, 1, 2, 1, 2])

    def make_common_source_bundles(self, *, c2_count: int = 1, c5_count: int = 1, split: str | None = None):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        c2_bundle = root / "c2"
        c5_bundle = root / "c5"
        for bundle in [c2_bundle, c5_bundle]:
            (bundle / "manifests").mkdir(parents=True)
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
        with (c2_bundle / "manifests" / "cloud_model_input_manifest.csv").open(
            "w", encoding="utf-8", newline=""
        ) as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for index in range(c2_count):
                writer.writerow(
                    {
                        "sample_id": f"landslide-{index + 1}",
                        "event_id": f"c2-event-{index + 1}",
                        "class_id": "1",
                        "class_name": "C2_debris_flow",
                        "split": split or "train",
                        "mask_path": f"/data/c2/masks/landslide-{index + 1}.tif",
                        "input_channels": "B;G;R;NIR;SLOPE",
                        "input_band_paths": "B:/data/c2/a.tif#band=1;G:/data/c2/a.tif#band=2;R:/data/c2/a.tif#band=3;NIR:/data/c2/a.tif#band=4;SLOPE:/data/c2/a.tif#band=6",
                    }
                )
        with (c5_bundle / "manifests" / "cloud_model_input_manifest.csv").open(
            "w", encoding="utf-8", newline=""
        ) as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for index in range(c5_count):
                writer.writerow(
                    {
                        "sample_id": f"fire-{index + 1}",
                        "event_id": f"c5-event-{index + 1}",
                        "class_id": "2",
                        "class_name": "C5_fire",
                        "split": split or "validation",
                        "mask_path": f"/data/c5/masks/fire-{index + 1}.tif",
                        "input_channels": "F01;F02;F03;F07;F11",
                        "input_band_paths": "F01:/data/c5/f01.tif;F02:/data/c5/f02.tif;F03:/data/c5/f03.tif;F07:/data/c5/f07.tif;F11:/data/c5/f11.tif",
                    }
                )
        return root, c2_bundle, c5_bundle


if __name__ == "__main__":
    unittest.main()
