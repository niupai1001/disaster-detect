import csv
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation_contract import build_phase_a_contract  # noqa: E402
from segmentation_contract.post_scene import select_post_disaster_scene  # noqa: E402
from segmentation_contract.phase_b import (
    _normalize_to_uint8,
    build_channel_alignment_audit,
    build_model_input_previews,
    build_post_disaster_model_input_manifest,
    build_phase_b_masks,
    build_phase_b_preflight,
    build_visual_qa,
)  # noqa: E402
from segmentation_contract.cli import main as segmentation_contract_main  # noqa: E402


class SegmentationContractPhaseATests(unittest.TestCase):
    def make_label_csv(self, rows):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        csv_path = root / "labels.csv"
        fieldnames = ["grid_id", "poly_id", "evt_date", "category", "conf_lvl", "geometry_wkt"]
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return root, csv_path

    def read_csv_rows(self, path):
        with path.open(encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    def test_phase_a_writes_class_map_with_c3_as_ignore_not_background(self):
        root, csv_path = self.make_label_csv(
            [
                {
                    "grid_id": "46RGT",
                    "poly_id": "100",
                    "evt_date": "2022/7/7",
                    "category": "3",
                    "conf_lvl": "3",
                    "geometry_wkt": "POLYGON ((0 0, 1 0, 1 1, 0 0))",
                }
            ]
        )

        result = build_phase_a_contract([csv_path], root / "contract")

        class_map = json.loads((result.output_dir / "class_map.json").read_text(encoding="utf-8"))
        manifest_rows = self.read_csv_rows(result.output_dir / "samples_manifest.csv")

        self.assertEqual(set(class_map), {"0", "1", "2", "255"})
        self.assertEqual(class_map["255"]["name"], "ignore")
        self.assertEqual(manifest_rows[0]["class_id"], "255")
        self.assertEqual(manifest_rows[0]["class_name"], "ignore")
        self.assertIn("taxonomy_pending", manifest_rows[0]["qa_flags"])

    def test_channels_preserve_declared_order_independent_of_input_order(self):
        root, csv_path = self.make_label_csv([])

        result = build_phase_a_contract(
            [csv_path],
            root / "contract",
            channel_names=["F12", "F01", "F07"],
        )

        channels = json.loads((result.output_dir / "channels.json").read_text(encoding="utf-8"))

        self.assertEqual([item["name"] for item in channels["channels"]], ["F12", "F01", "F07"])
        self.assertEqual([item["index"] for item in channels["channels"]], [0, 1, 2])

    def test_malformed_and_missing_required_rows_are_audited(self):
        root, csv_path = self.make_label_csv(
            [
                {
                    "grid_id": "46RGT",
                    "poly_id": "101",
                    "evt_date": "2022/7/7",
                    "category": "2",
                    "conf_lvl": "3",
                    "geometry_wkt": "NOT_A_WKT",
                },
                {
                    "grid_id": "46RGT",
                    "poly_id": "102",
                    "evt_date": "",
                    "category": "5",
                    "conf_lvl": "3",
                    "geometry_wkt": "POLYGON ((0 0, 1 0, 1 1, 0 0))",
                },
            ]
        )

        result = build_phase_a_contract([csv_path], root / "contract")

        skipped_rows = self.read_csv_rows(result.output_dir / "skipped_label_rows.csv")
        qa_summary = (result.output_dir / "qa_summary.md").read_text(encoding="utf-8")

        self.assertEqual(len(skipped_rows), 2)
        self.assertEqual({row["action"] for row in skipped_rows}, {"skip"})
        self.assertIn("invalid_wkt", {row["parse_error"] for row in skipped_rows})
        self.assertIn("missing_event_id", {row["parse_error"] for row in skipped_rows})
        self.assertIn("malformed_rows: 1", qa_summary)
        self.assertIn("skipped_rows: 2", qa_summary)

    def test_event_level_split_does_not_split_same_event(self):
        root, csv_path = self.make_label_csv(
            [
                {
                    "grid_id": "46RGT",
                    "poly_id": str(idx),
                    "evt_date": event_date,
                    "category": "2",
                    "conf_lvl": "3",
                    "geometry_wkt": "POLYGON ((0 0, 1 0, 1 1, 0 0))",
                }
                for idx, event_date in enumerate(
                    ["2022/7/7", "2022/7/7", "2021/1/1", "2020/1/1", "2019/1/1"],
                    start=1,
                )
            ]
        )

        result = build_phase_a_contract([csv_path], root / "contract")

        manifest_rows = self.read_csv_rows(result.output_dir / "samples_manifest.csv")
        splits_by_event = {}
        for row in manifest_rows:
            splits_by_event.setdefault(row["event_id"], set()).add(row["split"])

        self.assertTrue(all(len(splits) == 1 for splits in splits_by_event.values()))
        self.assertIn("leakage_check: pass", (result.output_dir / "qa_summary.md").read_text(encoding="utf-8"))

    def test_event_ids_are_zero_padded_and_raster_band_paths_are_linked(self):
        root, csv_path = self.make_label_csv(
            [
                {
                    "grid_id": "46RGT",
                    "poly_id": "405",
                    "evt_date": "2022/7/7",
                    "category": "2",
                    "conf_lvl": "3",
                    "geometry_wkt": "POLYGON ((0 0, 1 0, 1 1, 0 0))",
                }
            ]
        )
        raster_path = root / "T_C2_46RGT_405_EV20220707_IM20220731_F17.tif"
        raster_path.write_text("fake raster placeholder", encoding="utf-8")

        result = build_phase_a_contract([csv_path], root / "contract", raster_paths=[raster_path])

        manifest_rows = self.read_csv_rows(result.output_dir / "samples_manifest.csv")

        self.assertEqual(manifest_rows[0]["event_id"], "C2_46RGT_EV20220707")
        self.assertEqual(manifest_rows[0]["scene_id"], "IM20220731")
        self.assertEqual(manifest_rows[0]["source_raster_id"], "C2_46RGT_405_EV20220707")
        self.assertIn("F17:", manifest_rows[0]["source_band_paths"])
        self.assertIn(str(raster_path), manifest_rows[0]["source_band_paths"])

    def test_phase_a_candidate_scenes_do_not_mix_bands_across_image_dates(self):
        root, csv_path = self.make_label_csv(
            [
                {
                    "grid_id": "46RGT",
                    "poly_id": "405",
                    "evt_date": "2022/7/7",
                    "category": "5",
                    "conf_lvl": "3",
                    "geometry_wkt": "POLYGON ((0 0, 1 0, 1 1, 0 0))",
                }
            ]
        )
        raster_paths = [
            root / "T_C5_46RGT_405_EV20220707_IM20220708_F01.tif",
            root / "T_C5_46RGT_405_EV20220707_IM20220709_F02.tif",
        ]
        for path in raster_paths:
            path.write_text("fake raster placeholder", encoding="utf-8")

        result = build_phase_a_contract([csv_path], root / "contract", raster_paths=raster_paths)

        manifest_rows = self.read_csv_rows(result.output_dir / "samples_manifest.csv")
        candidate_scenes = json.loads(manifest_rows[0]["candidate_scenes"])

        self.assertEqual([scene["image_date"] for scene in candidate_scenes], ["20220708", "20220709"])
        self.assertEqual(set(candidate_scenes[0]["band_paths"]), {"F01"})
        self.assertEqual(set(candidate_scenes[1]["band_paths"]), {"F02"})
        self.assertNotIn("F02:", manifest_rows[0]["source_band_paths"])

    def test_select_post_disaster_scene_requires_same_date_required_channels(self):
        row = {
            "sample_id": "s1",
            "event_date": "20220707",
            "split": "train",
            "candidate_scenes": json.dumps(
                [
                    {
                        "scene_id": "IM20220708",
                        "image_date": "20220708",
                        "days_after_event": "1",
                        "temporal_role": "post",
                        "band_paths": {"F01": "/tmp/f01.tif"},
                    },
                    {
                        "scene_id": "IM20220709",
                        "image_date": "20220709",
                        "days_after_event": "2",
                        "temporal_role": "post",
                        "band_paths": {"F02": "/tmp/f02.tif"},
                    },
                ]
            ),
        }

        result = select_post_disaster_scene(row, required_channels=("F01", "F02"))

        self.assertEqual(result["selection_status"], "blocked_missing_same_date_required_channels")
        self.assertEqual(result["selected_image_date"], "")

    def test_select_post_disaster_scene_blocks_pre_event_and_selects_earliest_valid_post_scene(self):
        row = {
            "sample_id": "s1",
            "event_date": "20220707",
            "split": "validation",
            "candidate_scenes": json.dumps(
                [
                    {
                        "scene_id": "IM20220701",
                        "image_date": "20220701",
                        "days_after_event": "-6",
                        "temporal_role": "pre",
                        "band_paths": {"F01": "/tmp/pre_f01.tif", "F02": "/tmp/pre_f02.tif"},
                    },
                    {
                        "scene_id": "IM20220710",
                        "image_date": "20220710",
                        "days_after_event": "3",
                        "temporal_role": "post",
                        "band_paths": {"F01": "/tmp/post_f01.tif", "F02": "/tmp/post_f02.tif"},
                    },
                ]
            ),
        }

        result = select_post_disaster_scene(row, required_channels=("F01", "F02"))

        self.assertEqual(result["selection_status"], "selected")
        self.assertEqual(result["selected_image_date"], "20220710")
        self.assertEqual(result["days_after_event"], "3")
        self.assertEqual(result["temporal_role"], "post")
        self.assertEqual(result["input_band_paths"], "F01:/tmp/post_f01.tif;F02:/tmp/post_f02.tif")

    def test_select_post_disaster_scene_can_rank_by_cloud_proxy_quality(self):
        row = {
            "sample_id": "s1",
            "event_date": "20220707",
            "split": "train",
            "candidate_scenes": json.dumps(
                [
                    {
                        "scene_id": "cloudy",
                        "image_date": "20220708",
                        "days_after_event": "1",
                        "temporal_role": "post",
                        "band_paths": {"F02": "cloudy_f02.tif", "F03": "cloudy_f03.tif", "F04": "cloudy_f04.tif"},
                    },
                    {
                        "scene_id": "clear",
                        "image_date": "20220710",
                        "days_after_event": "3",
                        "temporal_role": "post",
                        "band_paths": {"F02": "clear_f02.tif", "F03": "clear_f03.tif", "F04": "clear_f04.tif"},
                    },
                ]
            ),
        }

        result = select_post_disaster_scene(
            row,
            required_channels=("F02", "F03", "F04"),
            selection_strategy="quality_then_earliest",
            scene_quality_scores={
                "cloudy": {"quality_score": "0.9", "quality_score_status": "scored", "quality_score_detail": "cloudy"},
                "clear": {"quality_score": "0.1", "quality_score_status": "scored", "quality_score_detail": "clear"},
            },
        )

        self.assertEqual(result["selection_status"], "selected")
        self.assertEqual(result["selected_scene_id"], "clear")
        self.assertEqual(result["selected_image_date"], "20220710")
        self.assertEqual(result["quality_score"], "0.1")
        self.assertIn("quality_ranked_cloud_proxy", result["temporal_qa_flags"])

    def test_select_post_disaster_scene_blocks_test_split(self):
        row = {
            "sample_id": "s1",
            "event_date": "20220707",
            "split": "test",
            "candidate_scenes": json.dumps(
                [
                    {
                        "scene_id": "IM20220710",
                        "image_date": "20220710",
                        "days_after_event": "3",
                        "temporal_role": "post",
                        "band_paths": {"F01": "/tmp/post_f01.tif"},
                    }
                ]
            ),
        }

        result = select_post_disaster_scene(row, required_channels=("F01",))

        self.assertEqual(result["selection_status"], "blocked_sealed_test_split")

    def test_phase_b_preflight_creates_reports_with_dependency_status(self):
        root, csv_path = self.make_label_csv(
            [
                {
                    "grid_id": "46RGT",
                    "poly_id": "405",
                    "evt_date": "2022/7/7",
                    "category": "2",
                    "conf_lvl": "3",
                    "geometry_wkt": "POLYGON ((0 0, 1 0, 1 1, 0 0))",
                }
            ]
        )
        phase_a = build_phase_a_contract([csv_path], root / "contract")

        result = build_phase_b_preflight(phase_a.output_dir)

        self.assertIn(result.readiness, {"ready_for_geospatial_validation", "blocked_missing_geospatial_dependencies"})
        self.assertTrue((phase_a.output_dir / "masks").is_dir())
        self.assertTrue((phase_a.output_dir / "previews").is_dir())
        self.assertTrue((phase_a.output_dir / "reports").is_dir())
        self.assertTrue((phase_a.output_dir / "reports" / "geometry_validity_report.csv").exists())
        self.assertTrue((phase_a.output_dir / "reports" / "wkt_mask_alignment_report.csv").exists())
        dependency_report = (phase_a.output_dir / "reports" / "phase_b_dependency_report.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(f"phase_b_readiness: {result.readiness}", dependency_report)
        self.assertIn(f"shapely: {'available' if result.shapely_available else 'missing'}", dependency_report)
        self.assertIn(f"rasterio: {'available' if result.rasterio_available else 'missing'}", dependency_report)

    def test_cli_can_run_phase_b_preflight_after_phase_a(self):
        root, csv_path = self.make_label_csv(
            [
                {
                    "grid_id": "46RGT",
                    "poly_id": "405",
                    "evt_date": "2022/7/7",
                    "category": "5",
                    "conf_lvl": "3",
                    "geometry_wkt": "POLYGON ((0 0, 1 0, 1 1, 0 0))",
                }
            ]
        )
        output_dir = root / "contract"

        exit_code = segmentation_contract_main(
            [
                "--project-root",
                str(root),
                "--output-dir",
                str(output_dir),
                "--label-csv",
                str(csv_path),
                "--phase-b-preflight",
            ]
        )

        self.assertEqual(exit_code, 0)
        self.assertTrue((output_dir / "reports" / "phase_b_dependency_report.md").exists())

    def test_phase_b_builds_geospatial_mask_from_wkt_and_reference_raster(self):
        root, csv_path = self.make_label_csv(
            [
                {
                    "grid_id": "46RGT",
                    "poly_id": "405",
                    "evt_date": "2022/7/7",
                    "category": "2",
                    "conf_lvl": "3",
                    "geometry_wkt": "POLYGON ((0.2 0.2, 0.8 0.2, 0.8 0.8, 0.2 0.8, 0.2 0.2))",
                }
            ]
        )
        raster_path = root / "T_C2_46RGT_405_EV20220707_IM20220731_F01.tif"
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=10,
            width=10,
            count=1,
            dtype="uint16",
            crs="EPSG:4326",
            transform=from_origin(0, 1, 0.1, 0.1),
        ) as dst:
            dst.write(numpy.ones((1, 10, 10), dtype="uint16"))
        phase_a = build_phase_a_contract([csv_path], root / "contract", raster_paths=[raster_path])

        result = build_phase_b_masks(phase_a.output_dir)

        self.assertEqual(result.masks_written, 1)
        with rasterio.open(phase_a.output_dir / "masks" / "sample-000001.tif") as mask_ds:
            mask = mask_ds.read(1)
            self.assertEqual(mask_ds.crs.to_string(), "EPSG:4326")
            self.assertEqual(mask.max(), 1)
            self.assertGreater(int((mask == 1).sum()), 0)
        manifest_rows = self.read_csv_rows(phase_a.output_dir / "samples_manifest.csv")
        self.assertEqual(manifest_rows[0]["crs"], "EPSG:4326")
        self.assertEqual(manifest_rows[0]["height"], "10")
        self.assertEqual(manifest_rows[0]["width"], "10")
        self.assertEqual(manifest_rows[0]["nodata_policy"], "mask_nodata_255_background_0")
        geometry_rows = self.read_csv_rows(phase_a.output_dir / "reports" / "geometry_validity_report.csv")
        alignment_rows = self.read_csv_rows(phase_a.output_dir / "reports" / "wkt_mask_alignment_report.csv")
        self.assertEqual(geometry_rows[0]["geometry_status"], "valid")
        self.assertEqual(alignment_rows[0]["alignment_status"], "mask_written")

    def test_phase_b_uses_required_channel_grid_for_mask_reference(self):
        root, csv_path = self.make_label_csv(
            [
                {
                    "grid_id": "46RGT",
                    "poly_id": "405",
                    "evt_date": "2022/7/7",
                    "category": "5",
                    "conf_lvl": "3",
                    "geometry_wkt": "POLYGON ((0.2 0.2, 0.8 0.2, 0.8 0.8, 0.2 0.8, 0.2 0.2))",
                }
            ]
        )
        raster_specs = {
            "F13": (4, 4, from_origin(0, 1, 0.3, 0.3)),
            "F16": (10, 10, from_origin(0, 1, 0.1, 0.1)),
            "F17": (10, 10, from_origin(0, 1, 0.1, 0.1)),
        }
        raster_paths = []
        for band, (height, width, transform) in raster_specs.items():
            raster_path = root / f"T_C5_46RGT_405_EV20220707_IM20220731_{band}.tif"
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype="uint16",
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(numpy.ones((1, height, width), dtype="uint16"))
            raster_paths.append(raster_path)
        phase_a = build_phase_a_contract([csv_path], root / "contract", raster_paths=raster_paths)

        build_phase_b_masks(phase_a.output_dir)

        with rasterio.open(phase_a.output_dir / "masks" / "sample-000001.tif") as mask_ds:
            self.assertEqual((mask_ds.height, mask_ds.width), (10, 10))
            self.assertEqual(tuple(mask_ds.transform)[:6], tuple(raster_specs["F16"][2])[:6])
            self.assertEqual(mask_ds.nodata, 255)

    def test_phase_b_reports_missing_reference_raster_without_mask(self):
        root, csv_path = self.make_label_csv(
            [
                {
                    "grid_id": "46RGT",
                    "poly_id": "405",
                    "evt_date": "2022/7/7",
                    "category": "5",
                    "conf_lvl": "3",
                    "geometry_wkt": "POLYGON ((0.2 0.2, 0.8 0.2, 0.8 0.8, 0.2 0.8, 0.2 0.2))",
                }
            ]
        )
        phase_a = build_phase_a_contract([csv_path], root / "contract")

        result = build_phase_b_masks(phase_a.output_dir)

        self.assertEqual(result.masks_written, 0)
        alignment_rows = self.read_csv_rows(phase_a.output_dir / "reports" / "wkt_mask_alignment_report.csv")
        self.assertEqual(alignment_rows[0]["alignment_status"], "blocked_missing_reference_raster")

    def test_visual_qa_writes_training_manifest_excluding_ignore_and_preview_sheet(self):
        root, csv_path = self.make_label_csv(
            [
                {
                    "grid_id": "46RGT",
                    "poly_id": "405",
                    "evt_date": "2022/7/7",
                    "category": "5",
                    "conf_lvl": "3",
                    "geometry_wkt": "POLYGON ((0.2 0.2, 0.8 0.2, 0.8 0.8, 0.2 0.8, 0.2 0.2))",
                },
                {
                    "grid_id": "46RGT",
                    "poly_id": "406",
                    "evt_date": "2022/7/8",
                    "category": "3",
                    "conf_lvl": "3",
                    "geometry_wkt": "POLYGON ((0.2 0.2, 0.8 0.2, 0.8 0.8, 0.2 0.8, 0.2 0.2))",
                },
            ]
        )
        raster_paths = []
        for polygon_id, category, event_date in [("405", "5", "20220707"), ("406", "3", "20220708")]:
            raster_path = root / f"T_C{category}_46RGT_{polygon_id}_EV{event_date}_IM20220731_F01.tif"
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                height=10,
                width=10,
                count=1,
                dtype="uint16",
                crs="EPSG:4326",
                transform=from_origin(0, 1, 0.1, 0.1),
            ) as dst:
                dst.write(numpy.ones((1, 10, 10), dtype="uint16"))
            raster_paths.append(raster_path)
        phase_a = build_phase_a_contract([csv_path], root / "contract", raster_paths=raster_paths)
        build_phase_b_masks(phase_a.output_dir)

        result = build_visual_qa(phase_a.output_dir, max_items=2)

        training_rows = self.read_csv_rows(phase_a.output_dir / "training_manifest.csv")
        self.assertEqual(result.training_rows, 1)
        self.assertEqual(training_rows[0]["class_id"], "2")
        self.assertTrue((phase_a.output_dir / "previews" / "contact_sheet.png").exists())
        self.assertTrue((phase_a.output_dir / "reports" / "visual_qa_summary.md").exists())

    def test_visual_qa_training_manifest_excludes_test_split(self):
        root, csv_path = self.make_label_csv(
            [
                {
                    "grid_id": f"46RG{idx}",
                    "poly_id": str(500 + idx),
                    "evt_date": f"2022/7/{idx + 1}",
                    "category": "5",
                    "conf_lvl": "3",
                    "geometry_wkt": "POLYGON ((0.2 0.2, 0.8 0.2, 0.8 0.8, 0.2 0.8, 0.2 0.2))",
                }
                for idx in range(10)
            ]
        )
        raster_paths = []
        for idx in range(10):
            event_date = f"202207{idx + 1:02d}"
            grid = f"46RG{idx}"
            polygon = str(500 + idx)
            raster_path = root / f"T_C5_{grid}_{polygon}_EV{event_date}_IM20220731_F01.tif"
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                height=10,
                width=10,
                count=1,
                dtype="uint16",
                crs="EPSG:4326",
                transform=from_origin(0, 1, 0.1, 0.1),
            ) as dst:
                dst.write(numpy.ones((1, 10, 10), dtype="uint16"))
            raster_paths.append(raster_path)
        phase_a = build_phase_a_contract([csv_path], root / "contract", raster_paths=raster_paths)
        build_phase_b_masks(phase_a.output_dir, reference_channels=["F01"])

        build_visual_qa(phase_a.output_dir, max_items=10)

        training_rows = self.read_csv_rows(phase_a.output_dir / "training_manifest.csv")
        self.assertNotIn("test", {row["split"] for row in training_rows})

    def test_visual_qa_normalization_handles_nonfinite_values_without_warning(self):
        image = numpy.array([[numpy.nan, numpy.inf], [-numpy.inf, 1.0]], dtype="float32")

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            normalized = _normalize_to_uint8(numpy, image)

        self.assertEqual(normalized.dtype, numpy.uint8)
        self.assertTrue(numpy.isfinite(normalized).all())

    def test_channel_alignment_audit_keeps_only_required_aligned_channels(self):
        root, csv_path = self.make_label_csv(
            [
                {
                    "grid_id": "46RGT",
                    "poly_id": "405",
                    "evt_date": "2022/7/7",
                    "category": "5",
                    "conf_lvl": "3",
                    "geometry_wkt": "POLYGON ((0.2 0.2, 0.8 0.2, 0.8 0.8, 0.2 0.8, 0.2 0.2))",
                }
            ]
        )
        raster_paths = []
        for band in ["F16", "F17"]:
            raster_path = root / f"T_C5_46RGT_405_EV20220707_IM20220731_{band}.tif"
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                height=10,
                width=10,
                count=1,
                dtype="uint16",
                crs="EPSG:4326",
                transform=from_origin(0, 1, 0.1, 0.1),
            ) as dst:
                dst.write(numpy.ones((1, 10, 10), dtype="uint16"))
            raster_paths.append(raster_path)
        phase_a = build_phase_a_contract([csv_path], root / "contract", raster_paths=raster_paths)
        build_phase_b_masks(phase_a.output_dir)
        build_visual_qa(phase_a.output_dir, max_items=1)

        result = build_channel_alignment_audit(phase_a.output_dir, required_channels=["F16", "F17"])

        model_rows = self.read_csv_rows(phase_a.output_dir / "model_input_manifest.csv")
        audit_rows = self.read_csv_rows(phase_a.output_dir / "reports" / "channel_alignment_report.csv")
        self.assertEqual(result.model_input_rows, 1)
        self.assertEqual(model_rows[0]["input_channels"], "F16;F17")
        self.assertEqual(audit_rows[0]["channel_alignment_status"], "aligned")

    def test_post_disaster_model_input_manifest_selects_same_date_train_validation_channels(self):
        root, csv_path = self.make_label_csv(
            [
                {
                    "grid_id": "46RGT",
                    "poly_id": "405",
                    "evt_date": "2022/7/7",
                    "category": "5",
                    "conf_lvl": "3",
                    "geometry_wkt": "POLYGON ((0.2 0.2, 0.8 0.2, 0.8 0.8, 0.2 0.8, 0.2 0.2))",
                }
            ]
        )
        raster_paths = []
        for band in ["F01", "F02"]:
            raster_path = root / f"T_C5_46RGT_405_EV20220707_IM20220710_{band}.tif"
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                height=10,
                width=10,
                count=1,
                dtype="uint16",
                crs="EPSG:4326",
                transform=from_origin(0, 1, 0.1, 0.1),
            ) as dst:
                dst.write(numpy.ones((1, 10, 10), dtype="uint16"))
            raster_paths.append(raster_path)
        phase_a = build_phase_a_contract([csv_path], root / "contract", raster_paths=raster_paths)
        build_phase_b_masks(phase_a.output_dir, reference_channels=["F01"])
        build_visual_qa(phase_a.output_dir, max_items=1)

        result = build_post_disaster_model_input_manifest(phase_a.output_dir, required_channels=["F01", "F02"])

        model_rows = self.read_csv_rows(phase_a.output_dir / "model_input_manifest.csv")
        audit_rows = self.read_csv_rows(phase_a.output_dir / "reports" / "post_disaster_channel_audit.csv")
        self.assertEqual(result.model_input_rows, 1)
        self.assertEqual(model_rows[0]["input_channels"], "F01;F02")
        self.assertEqual(model_rows[0]["selected_image_date"], "20220710")
        self.assertEqual(model_rows[0]["days_after_event"], "3")
        self.assertEqual(model_rows[0]["selection_status"], "selected")
        self.assertEqual(audit_rows[0]["selection_status"], "selected")

    def test_post_disaster_model_input_manifest_prefers_lower_cloud_proxy_scene(self):
        root, csv_path = self.make_label_csv(
            [
                {
                    "grid_id": "46RGT",
                    "poly_id": "411",
                    "evt_date": "2022/7/7",
                    "category": "5",
                    "conf_lvl": "3",
                    "geometry_wkt": "POLYGON ((0.2 0.2, 0.8 0.2, 0.8 0.8, 0.2 0.8, 0.2 0.2))",
                }
            ]
        )
        raster_paths = []
        for image_date, value in [("20220708", 9000), ("20220710", 1200)]:
            for band in ["F02", "F03", "F04"]:
                raster_path = root / f"T_C5_46RGT_411_EV20220707_IM{image_date}_{band}.tif"
                with rasterio.open(
                    raster_path,
                    "w",
                    driver="GTiff",
                    height=10,
                    width=10,
                    count=1,
                    dtype="uint16",
                    crs="EPSG:4326",
                    transform=from_origin(0, 1, 0.1, 0.1),
                ) as dst:
                    dst.write(numpy.full((1, 10, 10), value, dtype="uint16"))
                raster_paths.append(raster_path)
        phase_a = build_phase_a_contract([csv_path], root / "contract", raster_paths=raster_paths)
        build_phase_b_masks(phase_a.output_dir, reference_channels=["F04"])
        build_visual_qa(phase_a.output_dir, max_items=1)

        build_post_disaster_model_input_manifest(
            phase_a.output_dir,
            required_channels=["F02", "F03", "F04"],
            selection_strategy="quality_then_earliest",
        )

        model_rows = self.read_csv_rows(phase_a.output_dir / "model_input_manifest.csv")
        quality_rows = self.read_csv_rows(phase_a.output_dir / "reports" / "post_disaster_scene_quality_report.csv")
        self.assertEqual(model_rows[0]["selected_image_date"], "20220710")
        self.assertEqual(model_rows[0]["selection_strategy"], "quality_then_earliest")
        self.assertEqual(model_rows[0]["quality_score_status"], "scored")
        self.assertIn("bright_fraction", model_rows[0]["quality_score_detail"])
        self.assertEqual(len(quality_rows), 2)
        self.assertEqual([row["selected"] for row in quality_rows], ["false", "true"])

    def test_post_disaster_model_input_manifest_resamples_channels_to_mask_grid(self):
        root, csv_path = self.make_label_csv(
            [
                {
                    "grid_id": "46RGT",
                    "poly_id": "412",
                    "evt_date": "2022/7/7",
                    "category": "5",
                    "conf_lvl": "3",
                    "geometry_wkt": "POLYGON ((0.2 0.2, 0.8 0.2, 0.8 0.8, 0.2 0.8, 0.2 0.2))",
                }
            ]
        )
        raster_paths = []
        for band, height, width, pixel_size in [
            ("F01", 10, 10, 0.1),
            ("F11", 5, 5, 0.2),
        ]:
            raster_path = root / f"T_C5_46RGT_412_EV20220707_IM20220731_{band}.tif"
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype="uint16",
                crs="EPSG:4326",
                transform=from_origin(0, 1, pixel_size, pixel_size),
            ) as dst:
                dst.write(numpy.ones((1, height, width), dtype="uint16"))
            raster_paths.append(raster_path)
        phase_a = build_phase_a_contract([csv_path], root / "contract", raster_paths=raster_paths)
        build_phase_b_masks(phase_a.output_dir, reference_channels=["F01"])
        build_visual_qa(phase_a.output_dir, max_items=1)

        build_post_disaster_model_input_manifest(phase_a.output_dir, required_channels=["F01", "F11"])

        model_rows = self.read_csv_rows(phase_a.output_dir / "model_input_manifest.csv")
        input_paths = dict(item.split(":", 1) for item in model_rows[0]["input_band_paths"].split(";"))
        self.assertIn("model_inputs", input_paths["F11"])
        with rasterio.open(phase_a.output_dir / "masks" / "sample-000001.tif") as mask_ds:
            mask_grid = (mask_ds.crs, mask_ds.transform, mask_ds.height, mask_ds.width)
        with rasterio.open(input_paths["F01"]) as f01_ds, rasterio.open(input_paths["F11"]) as f11_ds:
            self.assertEqual((f01_ds.crs, f01_ds.transform, f01_ds.height, f01_ds.width), mask_grid)
            self.assertEqual((f11_ds.crs, f11_ds.transform, f11_ds.height, f11_ds.width), mask_grid)
        self.assertIn("F11:", model_rows[0]["resampled_channel_paths"])
        self.assertEqual(model_rows[0]["resampling_policy"], "bilinear_to_mask_grid")

    def test_post_disaster_model_input_manifest_writes_derived_index_channels(self):
        root, csv_path = self.make_label_csv(
            [
                {
                    "grid_id": "46RGT",
                    "poly_id": "414",
                    "evt_date": "2022/7/7",
                    "category": "5",
                    "conf_lvl": "3",
                    "geometry_wkt": "POLYGON ((0.2 0.2, 0.8 0.2, 0.8 0.8, 0.2 0.8, 0.2 0.2))",
                }
            ]
        )
        raster_paths = []
        values = {"F02": 1000, "F03": 2000, "F04": 2000, "F07": 8000, "F11": 3000, "F12": 1000}
        for band, value in values.items():
            raster_path = root / f"T_C5_46RGT_414_EV20220707_IM20220731_{band}.tif"
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                height=10,
                width=10,
                count=1,
                dtype="uint16",
                crs="EPSG:4326",
                transform=from_origin(0, 1, 0.1, 0.1),
            ) as dst:
                dst.write(numpy.full((1, 10, 10), value, dtype="uint16"))
            raster_paths.append(raster_path)
        phase_a = build_phase_a_contract([csv_path], root / "contract", raster_paths=raster_paths)
        build_phase_b_masks(phase_a.output_dir, reference_channels=["F04"])
        build_visual_qa(phase_a.output_dir, max_items=1)

        build_post_disaster_model_input_manifest(
            phase_a.output_dir,
            required_channels=["F02", "F03", "F04", "F07", "F11", "F12"],
            derived_indices=["NDVI", "NBR", "NDMI", "BRIGHTNESS"],
        )

        model_rows = self.read_csv_rows(phase_a.output_dir / "model_input_manifest.csv")
        input_paths = dict(item.split(":", 1) for item in model_rows[0]["input_band_paths"].split(";"))
        self.assertEqual(
            model_rows[0]["input_channels"],
            "F02;F03;F04;F07;F11;F12;NDVI;NBR;NDMI;BRIGHTNESS",
        )
        self.assertIn("NDVI:", model_rows[0]["derived_index_paths"])
        self.assertIn("BRIGHTNESS:", model_rows[0]["derived_index_paths"])
        with rasterio.open(input_paths["NDVI"]) as ndvi_ds:
            ndvi = ndvi_ds.read(1)
            self.assertAlmostEqual(float(ndvi[0, 0]), 0.6, places=4)
        with rasterio.open(input_paths["BRIGHTNESS"]) as brightness_ds:
            brightness = brightness_ds.read(1)
            self.assertAlmostEqual(float(brightness[0, 0]), 0.166666, places=4)

    def test_model_input_previews_render_multichannel_composites(self):
        root, csv_path = self.make_label_csv(
            [
                {
                    "grid_id": "46RGT",
                    "poly_id": "413",
                    "evt_date": "2022/7/7",
                    "category": "5",
                    "conf_lvl": "3",
                    "geometry_wkt": "POLYGON ((0.2 0.2, 0.8 0.2, 0.8 0.8, 0.2 0.8, 0.2 0.2))",
                }
            ]
        )
        raster_paths = []
        base = numpy.arange(100, dtype="uint16").reshape(10, 10)
        for band, offset in [("F02", 0), ("F03", 100), ("F04", 200), ("F07", 300), ("F11", 400)]:
            raster_path = root / f"T_C5_46RGT_413_EV20220707_IM20220731_{band}.tif"
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                height=10,
                width=10,
                count=1,
                dtype="uint16",
                crs="EPSG:4326",
                transform=from_origin(0, 1, 0.1, 0.1),
            ) as dst:
                dst.write((base + offset)[None, :, :])
            raster_paths.append(raster_path)
        phase_a = build_phase_a_contract([csv_path], root / "contract", raster_paths=raster_paths)
        build_phase_b_masks(phase_a.output_dir, reference_channels=["F04"])
        build_visual_qa(phase_a.output_dir, max_items=1)
        build_post_disaster_model_input_manifest(
            phase_a.output_dir,
            required_channels=["F02", "F03", "F04", "F07", "F11"],
        )

        result = build_model_input_previews(phase_a.output_dir, max_items=1)

        self.assertEqual(result.preview_items, 1)
        self.assertTrue((phase_a.output_dir / "previews" / "model_input_contact_sheet.png").exists())
        self.assertTrue((phase_a.output_dir / "previews" / "model_input_sample-000001_composite.png").exists())
        summary = (phase_a.output_dir / "reports" / "model_input_visual_qa_summary.md").read_text(encoding="utf-8")
        self.assertIn("rgb_channels: F04;F03;F02", summary)
        self.assertIn("false_color_channels: F11;F07;F04", summary)

    def test_post_disaster_model_input_manifest_excludes_test_split(self):
        root, csv_path = self.make_label_csv(
            [
                {
                    "grid_id": f"46RG{idx}",
                    "poly_id": str(400 + idx),
                    "evt_date": f"2022/7/{idx + 1}",
                    "category": "5",
                    "conf_lvl": "3",
                    "geometry_wkt": "POLYGON ((0.2 0.2, 0.8 0.2, 0.8 0.8, 0.2 0.8, 0.2 0.2))",
                }
                for idx in range(10)
            ]
        )
        raster_paths = []
        for idx in range(10):
            event_date = f"202207{idx + 1:02d}"
            polygon = str(400 + idx)
            grid = f"46RG{idx}"
            for band in ["F01"]:
                raster_path = root / f"T_C5_{grid}_{polygon}_EV{event_date}_IM20220731_{band}.tif"
                with rasterio.open(
                    raster_path,
                    "w",
                    driver="GTiff",
                    height=10,
                    width=10,
                    count=1,
                    dtype="uint16",
                    crs="EPSG:4326",
                    transform=from_origin(0, 1, 0.1, 0.1),
                ) as dst:
                    dst.write(numpy.ones((1, 10, 10), dtype="uint16"))
                raster_paths.append(raster_path)
        phase_a = build_phase_a_contract([csv_path], root / "contract", raster_paths=raster_paths)
        build_phase_b_masks(phase_a.output_dir, reference_channels=["F01"])
        build_visual_qa(phase_a.output_dir, max_items=1)

        build_post_disaster_model_input_manifest(phase_a.output_dir, required_channels=["F01"])

        audit_rows = self.read_csv_rows(phase_a.output_dir / "reports" / "post_disaster_channel_audit.csv")
        model_rows = self.read_csv_rows(phase_a.output_dir / "model_input_manifest.csv")
        self.assertNotIn("test", {row["split"] for row in audit_rows})
        self.assertNotIn("test", {row["split"] for row in model_rows})

    def test_channel_alignment_audit_blocks_mask_grid_mismatch(self):
        root, csv_path = self.make_label_csv(
            [
                {
                    "grid_id": "46RGT",
                    "poly_id": "405",
                    "evt_date": "2022/7/7",
                    "category": "5",
                    "conf_lvl": "3",
                    "geometry_wkt": "POLYGON ((0.2 0.2, 0.8 0.2, 0.8 0.8, 0.2 0.8, 0.2 0.2))",
                }
            ]
        )
        raster_paths = []
        for band in ["F13", "F16", "F17"]:
            height = 4 if band == "F13" else 10
            width = 4 if band == "F13" else 10
            pixel_size = 0.3 if band == "F13" else 0.1
            raster_path = root / f"T_C5_46RGT_405_EV20220707_IM20220731_{band}.tif"
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype="uint16",
                crs="EPSG:4326",
                transform=from_origin(0, 1, pixel_size, pixel_size),
            ) as dst:
                dst.write(numpy.ones((1, height, width), dtype="uint16"))
            raster_paths.append(raster_path)
        phase_a = build_phase_a_contract([csv_path], root / "contract", raster_paths=raster_paths)
        build_phase_b_masks(phase_a.output_dir, reference_channels=["F13"])
        build_visual_qa(phase_a.output_dir, max_items=1)

        result = build_channel_alignment_audit(phase_a.output_dir, required_channels=["F16", "F17"])

        audit_rows = self.read_csv_rows(phase_a.output_dir / "reports" / "channel_alignment_report.csv")
        model_rows = self.read_csv_rows(phase_a.output_dir / "model_input_manifest.csv")
        self.assertEqual(result.model_input_rows, 0)
        self.assertEqual(model_rows, [])
        self.assertEqual(audit_rows[0]["channel_alignment_status"], "blocked_mask_grid_mismatch")


if __name__ == "__main__":
    unittest.main()
