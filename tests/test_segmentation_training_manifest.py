import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation_training.manifest import (  # noqa: E402
    filter_records,
    load_model_input_manifest,
    parse_band_paths,
    rebase_records,
)


class SegmentationTrainingManifestTests(unittest.TestCase):
    def write_manifest(self, rows):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        path = root / "model_input_manifest.csv"
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
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return root, path

    def test_parse_band_paths(self):
        parsed = parse_band_paths("F16:/tmp/a.tif;F17:/tmp/b.tif")

        self.assertEqual(parsed["F16"], "/tmp/a.tif")
        self.assertEqual(parsed["F17"], "/tmp/b.tif")

    def test_load_manifest_and_filter_class_scopes(self):
        _, path = self.write_manifest(
            [
                {
                    "sample_id": "s1",
                    "event_id": "e1",
                    "class_id": "1",
                    "class_name": "C2_debris_flow",
                    "split": "train",
                    "mask_path": "masks/s1.tif",
                    "input_channels": "F16;F17",
                    "input_band_paths": "F16:database/a.tif;F17:database/b.tif",
                },
                {
                    "sample_id": "s2",
                    "event_id": "e2",
                    "class_id": "2",
                    "class_name": "C5_fire",
                    "split": "validation",
                    "mask_path": "masks/s2.tif",
                    "input_channels": "F16;F17",
                    "input_band_paths": "F16:database/c.tif;F17:database/d.tif",
                },
            ]
        )

        records = load_model_input_manifest(path)

        self.assertEqual(records[0].input_channels, ("F16", "F17"))
        self.assertEqual(records[0].input_band_paths["F16"], "database/a.tif")
        self.assertEqual([record.sample_id for record in filter_records(records, class_scope="binary_c5")], ["s2"])

    def test_missing_required_channel_is_rejected(self):
        _, path = self.write_manifest(
            [
                {
                    "sample_id": "s1",
                    "event_id": "e1",
                    "class_id": "1",
                    "class_name": "C2_debris_flow",
                    "split": "train",
                    "mask_path": "masks/s1.tif",
                    "input_channels": "F16",
                    "input_band_paths": "F16:database/a.tif",
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "missing required channels"):
            load_model_input_manifest(path)

    def test_rebase_records_limits_to_required_channels(self):
        _, path = self.write_manifest(
            [
                {
                    "sample_id": "s1",
                    "event_id": "e1",
                    "class_id": "1",
                    "class_name": "C2_debris_flow",
                    "split": "train",
                    "mask_path": "masks/s1.tif",
                    "input_channels": "F01;F16;F17",
                    "input_band_paths": "F01:database/x.tif;F16:database/a.tif;F17:database/b.tif",
                }
            ]
        )

        rebased = rebase_records(
            load_model_input_manifest(path),
            local_root=Path("/local/project"),
            target_root=Path("/cloud/data"),
        )

        self.assertEqual(rebased[0].input_channels, ("F16", "F17"))
        self.assertEqual(rebased[0].input_band_paths["F16"], "/cloud/data/database/a.tif")


if __name__ == "__main__":
    unittest.main()

