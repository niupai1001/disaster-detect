import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation_training.manifest import load_model_input_manifest, validate_research_contract  # noqa: E402
from segmentation_training.multiscene import build_c5_multiscene_manifest  # noqa: E402


class SegmentationTrainingMultisceneTests(unittest.TestCase):
    def write_manifest(self, rows):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        manifest = root / "cloud_model_input_manifest.csv"
        fieldnames = [
            "sample_id",
            "event_id",
            "event_date",
            "class_id",
            "class_name",
            "split",
            "mask_path",
            "input_channels",
            "input_band_paths",
            "candidate_scenes",
            "selected_scene_id",
            "quality_score",
            "quality_score_status",
            "quality_score_detail",
            "disaster_id",
            "label_confidence",
            "ignore_mask_path",
        ]
        with manifest.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return root, manifest

    def test_build_c5_multiscene_manifest_expands_topk_without_changing_split(self):
        scene_json = (
            '[{"scene_id":"IM20200101","image_date":"20200101","days_after_event":"0","temporal_role":"same_day",'
            '"band_paths":{"F01":"database/a_f01.tif","F02":"database/a_f02.tif","F03":"database/a_f03.tif"}},'
            '{"scene_id":"IM20200105","image_date":"20200105","days_after_event":"4","temporal_role":"post",'
            '"band_paths":{"F01":"database/b_f01.tif","F02":"database/b_f02.tif","F03":"database/b_f03.tif"}},'
            '{"scene_id":"IM20191230","image_date":"20191230","days_after_event":"-2","temporal_role":"pre",'
            '"band_paths":{"F01":"database/pre_f01.tif","F02":"database/pre_f02.tif","F03":"database/pre_f03.tif"}},'
            '{"scene_id":"IM20200215","image_date":"20200215","days_after_event":"45","temporal_role":"post",'
            '"band_paths":{"F01":"database/late_f01.tif","F02":"database/late_f02.tif","F03":"database/late_f03.tif"}}]'
        )
        root, manifest = self.write_manifest(
            [
                {
                    "sample_id": "sample-a",
                    "event_id": "event-a",
                    "event_date": "20200101",
                    "class_id": "2",
                    "class_name": "C5_fire",
                    "split": "train",
                    "mask_path": "/data/base/masks/sample-a.tif",
                    "input_channels": "F01;F02;F03",
                    "input_band_paths": "F01:model_inputs/a_f01.tif;F02:model_inputs/a_f02.tif;F03:model_inputs/a_f03.tif",
                    "candidate_scenes": scene_json,
                    "selected_scene_id": "IM20200105",
                    "quality_score": "0.01",
                    "quality_score_status": "scored",
                    "quality_score_detail": "selected_scene_quality",
                    "disaster_id": "C5",
                    "label_confidence": "unknown",
                    "ignore_mask_path": "",
                }
            ]
        )
        output = root / "expanded.csv"

        summary = build_c5_multiscene_manifest(
            manifest_path=manifest,
            output_path=output,
            required_channels=("F01", "F02", "F03"),
            top_k=2,
            cloud_data_root=Path("/data/base"),
        )

        records = load_model_input_manifest(output, required_channels=("F01", "F02", "F03"))
        self.assertEqual(summary["expanded_records"], 2)
        self.assertEqual([record.split for record in records], ["train", "train"])
        self.assertEqual([record.row["parent_sample_id"] for record in records], ["sample-a", "sample-a"])
        self.assertEqual([record.row["selected_scene_id"] for record in records], ["IM20200105", "IM20200101"])
        self.assertEqual(records[0].sample_id, "sample-a__IM20200105")
        self.assertIn("/data/base/database/b_f01.tif", records[0].input_band_paths["F01"])
        self.assertEqual(validate_research_contract(records, required_disasters=("C5",)), [])

    def test_build_c5_multiscene_manifest_reports_parent_split_leakage(self):
        scene_json = (
            '[{"scene_id":"IM20200101","image_date":"20200101","days_after_event":"0","temporal_role":"same_day",'
            '"band_paths":{"F01":"database/a_f01.tif"}}]'
        )
        root, manifest = self.write_manifest(
            [
                {
                    "sample_id": "same-parent",
                    "event_id": "event-a",
                    "event_date": "20200101",
                    "class_id": "2",
                    "class_name": "C5_fire",
                    "split": "train",
                    "mask_path": "/data/base/masks/a.tif",
                    "input_channels": "F01",
                    "input_band_paths": "F01:model_inputs/a_f01.tif",
                    "candidate_scenes": scene_json,
                    "selected_scene_id": "IM20200101",
                    "quality_score": "0",
                    "quality_score_status": "scored",
                    "quality_score_detail": "",
                    "disaster_id": "C5",
                    "label_confidence": "unknown",
                    "ignore_mask_path": "",
                },
                {
                    "sample_id": "same-parent",
                    "event_id": "event-a",
                    "event_date": "20200101",
                    "class_id": "2",
                    "class_name": "C5_fire",
                    "split": "validation",
                    "mask_path": "/data/base/masks/a.tif",
                    "input_channels": "F01",
                    "input_band_paths": "F01:model_inputs/a_f01.tif",
                    "candidate_scenes": scene_json,
                    "selected_scene_id": "IM20200101",
                    "quality_score": "0",
                    "quality_score_status": "scored",
                    "quality_score_detail": "",
                    "disaster_id": "C5",
                    "label_confidence": "unknown",
                    "ignore_mask_path": "",
                },
            ]
        )

        with self.assertRaisesRegex(ValueError, "parent sample split leakage"):
            build_c5_multiscene_manifest(
                manifest_path=manifest,
                output_path=root / "expanded.csv",
                required_channels=("F01",),
                top_k=1,
                cloud_data_root=Path("/data/base"),
            )


if __name__ == "__main__":
    unittest.main()
