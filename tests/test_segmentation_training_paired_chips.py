import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation_training.paired_chips import LANDSLIDE_9CH_CHANNELS, build_paired_chip_contract  # noqa: E402


class SegmentationTrainingPairedChipTests(unittest.TestCase):
    def test_build_paired_chip_contract_writes_multiband_manifest_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            mask_dir = root / "masks"
            image_dir.mkdir()
            mask_dir.mkdir()
            for sample_id in [0, 1]:
                with rasterio.open(
                    image_dir / f"sample_{sample_id}.tif",
                    "w",
                    driver="GTiff",
                    height=4,
                    width=4,
                    count=len(LANDSLIDE_9CH_CHANNELS),
                    dtype="float32",
                    crs="EPSG:4326",
                    transform=from_origin(0, 1, 0.1, 0.1),
                ) as dst:
                    dst.write(np.ones((len(LANDSLIDE_9CH_CHANNELS), 4, 4), dtype="float32"))
                with rasterio.open(
                    mask_dir / f"sample_{sample_id}_mask.tif",
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
                mask_dir / "sample_99_mask.tif",
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

            result = build_paired_chip_contract(
                image_dir=image_dir,
                mask_dir=mask_dir,
                output_dir=root / "contract",
                train_fraction=0.5,
            )

            self.assertEqual(result.paired_rows, 2)
            self.assertEqual(result.unmatched_masks, 1)
            with (root / "contract" / "model_input_manifest.csv").open(encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            channels = json.loads((root / "contract" / "channels.json").read_text(encoding="utf-8"))

        self.assertEqual(rows[0]["class_id"], "1")
        self.assertEqual(rows[0]["class_name"], "C2_debris_flow")
        self.assertEqual(rows[0]["disaster_id"], "C2")
        self.assertEqual(rows[0]["label_confidence"], "unknown")
        self.assertEqual(rows[0]["ignore_mask_path"], "")
        self.assertEqual(rows[0]["input_channels"], ";".join(LANDSLIDE_9CH_CHANNELS))
        self.assertIn("#band=9", rows[0]["input_band_paths"])
        self.assertEqual([item["name"] for item in channels["channels"]], list(LANDSLIDE_9CH_CHANNELS))


if __name__ == "__main__":
    unittest.main()
