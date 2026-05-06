import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation_training.preview import write_contact_sheet, write_prediction_preview  # noqa: E402


class SegmentationTrainingPreviewTests(unittest.TestCase):
    def test_preview_and_contact_sheet_are_written(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            channels = np.stack(
                [
                    np.arange(16, dtype=np.float32).reshape(4, 4),
                    np.arange(16, 32, dtype=np.float32).reshape(4, 4),
                ],
                axis=0,
            )
            label = np.array([[0, 1, 1, 0], [0, 2, 2, 0], [0, 0, 255, 0], [1, 0, 0, 2]], dtype=np.uint8)
            prediction = np.array([[0, 1, 0, 0], [0, 2, 1, 0], [0, 0, 255, 0], [2, 0, 0, 2]], dtype=np.uint8)
            preview_a = write_prediction_preview(
                channels=channels,
                label=label,
                prediction=prediction,
                output_path=root / "preview_a.png",
                title="s1",
            )
            preview_b = write_prediction_preview(
                channels=channels,
                label=label,
                prediction=label,
                output_path=root / "preview_b.png",
            )
            sheet = write_contact_sheet([preview_a, preview_b], root / "sheet.png")

            self.assertTrue(preview_a.exists())
            self.assertTrue(sheet.exists())
            with Image.open(sheet) as sheet_image, Image.open(preview_a) as preview_image:
                self.assertGreater(sheet_image.size[0], preview_image.size[0])


if __name__ == "__main__":
    unittest.main()
