import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation_training.manifest import (  # noqa: E402
    filter_records,
    load_model_input_manifest,
    parse_band_paths,
    rebase_records,
    split_band_reference,
    validate_research_contract,
    validate_record_paths,
    validate_record_raster_grids,
    write_research_manifest,
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

    def test_band_reference_suffix_preserves_path_and_band_index(self):
        path, band_index = split_band_reference("bands/multiband.tif#band=3")

        self.assertEqual(path, "bands/multiband.tif")
        self.assertEqual(band_index, 3)

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

    def test_validate_record_paths_checks_each_band_path(self):
        root, path = self.write_manifest(
            [
                {
                    "sample_id": "s1",
                    "event_id": "e1",
                    "class_id": "2",
                    "class_name": "C5_fire",
                    "split": "train",
                    "mask_path": "masks/s1.tif",
                    "input_channels": "F16;F17",
                    "input_band_paths": "F16:bands/missing.tif;F17:bands/f17.tif",
                }
            ]
        )
        (root / "masks").mkdir()
        (root / "bands").mkdir()
        (root / "masks" / "s1.tif").write_bytes(b"mask")
        (root / "bands" / "f17.tif").write_bytes(b"band")

        errors = validate_record_paths(load_model_input_manifest(path), contract_dir=root)

        self.assertEqual(len(errors), 1)
        self.assertIn("missing F16 raster", errors[0])

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

    def test_validate_record_raster_grids_rejects_mask_grid_mismatch(self):
        root, path = self.write_manifest(
            [
                {
                    "sample_id": "s1",
                    "event_id": "e1",
                    "class_id": "2",
                    "class_name": "C5_fire",
                    "split": "train",
                    "mask_path": "masks/s1.tif",
                    "input_channels": "F16;F17",
                    "input_band_paths": "F16:bands/f16.tif;F17:bands/f17.tif",
                }
            ]
        )
        (root / "masks").mkdir()
        (root / "bands").mkdir()
        with rasterio.open(
            root / "masks" / "s1.tif",
            "w",
            driver="GTiff",
            height=4,
            width=4,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=from_origin(0, 1, 0.3, 0.3),
        ) as dst:
            dst.write(np.ones((1, 4, 4), dtype="uint8"))
        for name in ["f16.tif", "f17.tif"]:
            with rasterio.open(
                root / "bands" / name,
                "w",
                driver="GTiff",
                height=10,
                width=10,
                count=1,
                dtype="uint16",
                crs="EPSG:4326",
                transform=from_origin(0, 1, 0.1, 0.1),
            ) as dst:
                dst.write(np.ones((1, 10, 10), dtype="uint16"))

        errors = validate_record_raster_grids(load_model_input_manifest(path), contract_dir=root)

        self.assertEqual(len(errors), 1)
        self.assertIn("s1", errors[0])
        self.assertIn("mask grid mismatch", errors[0])

    def test_validate_record_raster_grids_accepts_multiband_band_references(self):
        root, path = self.write_manifest(
            [
                {
                    "sample_id": "s1",
                    "event_id": "e1",
                    "class_id": "1",
                    "class_name": "C2_debris_flow",
                    "split": "train",
                    "mask_path": "masks/s1.tif",
                    "input_channels": "CH01;CH02",
                    "input_band_paths": "CH01:bands/multi.tif#band=1;CH02:bands/multi.tif#band=2",
                }
            ]
        )
        (root / "masks").mkdir()
        (root / "bands").mkdir()
        with rasterio.open(
            root / "masks" / "s1.tif",
            "w",
            driver="GTiff",
            height=4,
            width=4,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=from_origin(0, 1, 0.1, 0.1),
        ) as dst:
            dst.write(np.ones((1, 4, 4), dtype="uint8"))
        with rasterio.open(
            root / "bands" / "multi.tif",
            "w",
            driver="GTiff",
            height=4,
            width=4,
            count=2,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(0, 1, 0.1, 0.1),
        ) as dst:
            dst.write(np.ones((2, 4, 4), dtype="float32"))

        records = load_model_input_manifest(path, required_channels=("CH01", "CH02"))
        errors = validate_record_raster_grids(records, contract_dir=root, required_channels=("CH01", "CH02"))

        self.assertEqual(errors, [])

    def test_validate_research_contract_accepts_disaster_metadata_and_ignore_masks(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        path = root / "model_input_manifest.csv"
        (root / "masks").mkdir()
        (root / "ignore").mkdir()
        for name in ["c2.tif", "c5.tif", "c2_ignore.tif"]:
            (root / "masks" / name).write_bytes(b"mask")
        (root / "ignore" / "c5_ignore.tif").write_bytes(b"ignore")
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
        rows = [
            {
                "sample_id": "c2-train",
                "event_id": "e1",
                "class_id": "1",
                "class_name": "C2_landslide",
                "split": "train",
                "mask_path": "masks/c2.tif",
                "input_channels": "B;G",
                "input_band_paths": "B:database/b.tif;G:database/g.tif",
                "disaster_id": "C2",
                "label_confidence": "medium",
                "ignore_mask_path": "masks/c2_ignore.tif",
            },
            {
                "sample_id": "c5-validation",
                "event_id": "e2",
                "class_id": "2",
                "class_name": "C5_fire",
                "split": "validation",
                "mask_path": "masks/c5.tif",
                "input_channels": "B;G",
                "input_band_paths": "B:database/b.tif;G:database/g.tif",
                "disaster_id": "C5",
                "label_confidence": "0.85",
                "ignore_mask_path": "ignore/c5_ignore.tif",
            },
        ]
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        records = load_model_input_manifest(path, required_channels=("B", "G"))
        errors = validate_research_contract(records, contract_dir=root, required_disasters=("C2", "C5"))

        self.assertEqual(errors, [])

    def test_validate_research_contract_rejects_disaster_mismatch_bad_confidence_and_split_leakage(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
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
            "disaster_id",
            "label_confidence",
        ]
        rows = [
            {
                "sample_id": "same-sample",
                "event_id": "e1",
                "class_id": "1",
                "class_name": "C2_landslide",
                "split": "train",
                "mask_path": "masks/c2.tif",
                "input_channels": "B;G",
                "input_band_paths": "B:database/b.tif;G:database/g.tif",
                "disaster_id": "C5",
                "label_confidence": "certain",
            },
            {
                "sample_id": "same-sample",
                "event_id": "e1",
                "class_id": "1",
                "class_name": "C2_landslide",
                "split": "validation",
                "mask_path": "masks/c2b.tif",
                "input_channels": "B;G",
                "input_band_paths": "B:database/b.tif;G:database/g.tif",
                "disaster_id": "C2",
                "label_confidence": "high",
            },
        ]
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        records = load_model_input_manifest(path, required_channels=("B", "G"))
        errors = validate_research_contract(records, required_disasters=("C2", "C5"))

        self.assertTrue(any("disaster_id C5 does not match class_id 1" in error for error in errors))
        self.assertTrue(any("invalid label_confidence" in error for error in errors))
        self.assertTrue(any("appears in multiple splits" in error for error in errors))

    def test_write_research_manifest_adds_missing_research_fields(self):
        root, path = self.write_manifest(
            [
                {
                    "sample_id": "s1",
                    "event_id": "e1",
                    "class_id": "1",
                    "class_name": "C2_landslide",
                    "split": "train",
                    "mask_path": "masks/s1.tif",
                    "input_channels": "B;G",
                    "input_band_paths": "B:database/b.tif;G:database/g.tif",
                }
            ]
        )
        output = root / "research_manifest.csv"
        records = load_model_input_manifest(path, required_channels=("B", "G"))

        write_research_manifest(output, records)

        upgraded = load_model_input_manifest(output, required_channels=("B", "G"))
        self.assertEqual(upgraded[0].row["disaster_id"], "C2")
        self.assertEqual(upgraded[0].row["label_confidence"], "unknown")
        self.assertEqual(upgraded[0].row["ignore_mask_path"], "")
        self.assertEqual(validate_research_contract(upgraded, required_disasters=("C2",)), [])


if __name__ == "__main__":
    unittest.main()
