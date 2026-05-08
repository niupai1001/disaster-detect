import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation_training.data import read_raster_array  # noqa: E402


class SegmentationTrainingDataTests(unittest.TestCase):
    def test_read_raster_array_supports_band_reference_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "multi.tif"
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=2,
                width=2,
                count=2,
                dtype="uint8",
                crs="EPSG:4326",
                transform=from_origin(0, 1, 0.1, 0.1),
            ) as dst:
                dst.write(np.stack([np.ones((2, 2), dtype="uint8"), np.full((2, 2), 5, dtype="uint8")]))

            array = read_raster_array(f"{path}#band=2")

        self.assertTrue(np.array_equal(array, np.full((2, 2), 5, dtype="uint8")))


if __name__ == "__main__":
    unittest.main()
