import sys
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation_training.cloud import (  # noqa: E402
    _BestCheckpointTracker,
    _TrainingProgressLogger,
    _apply_channel_perturbation,
    _apply_training_augmentation,
    _build_record_cache,
    _build_trainable_model,
    _limit_records,
    _build_metrics_payload,
    _configure_torch_runtime,
    _foreground_window_coverage_rows,
    _load_resume_checkpoint,
    _maybe_write_training_curves,
    _normalize_band,
    _pad_window,
    _read_training_window,
    _read_validation_window,
    _resolve_resume_checkpoint,
    _write_p14_gate_reports,
    _target_has_valid_pixels,
    _write_run_outputs,
    prepare_cloud_run,
)
from segmentation_training.cli import main as training_main  # noqa: E402


class SegmentationTrainingCloudTests(unittest.TestCase):
    def test_configure_torch_runtime_enables_cuda_performance_flags_only_on_cuda(self):
        class FakeCuda:
            @staticmethod
            def is_available():
                return True

        class FakeCudnn:
            benchmark = False

        class FakeBackends:
            cudnn = FakeCudnn()

        class FakeTorch:
            cuda = FakeCuda()
            backends = FakeBackends()

        runtime = _configure_torch_runtime(
            FakeTorch,
            device="cuda",
            performance_config={"mixed_precision": True, "channels_last": True, "cudnn_benchmark": True},
        )

        self.assertTrue(runtime["mixed_precision"])
        self.assertTrue(runtime["channels_last"])
        self.assertTrue(runtime["cudnn_benchmark"])
        self.assertTrue(FakeTorch.backends.cudnn.benchmark)

    def test_configure_torch_runtime_keeps_cpu_safe(self):
        class FakeCudnn:
            benchmark = True

        class FakeBackends:
            cudnn = FakeCudnn()

        class FakeTorch:
            backends = FakeBackends()

        runtime = _configure_torch_runtime(
            FakeTorch,
            device="cpu",
            performance_config={"mixed_precision": True, "channels_last": True, "cudnn_benchmark": True},
        )

        self.assertFalse(runtime["mixed_precision"])
        self.assertFalse(runtime["channels_last"])
        self.assertFalse(runtime["cudnn_benchmark"])
        self.assertFalse(FakeTorch.backends.cudnn.benchmark)

    def test_best_checkpoint_tracker_saves_only_when_metric_improves(self):
        class FakeModel:
            def __init__(self):
                self.value = 0

            def state_dict(self):
                return {"value": self.value}

        class FakeTorch:
            saves = []

            @classmethod
            def save(cls, state, path):
                cls.saves.append((state, path.name))
                path.write_text(str(state["value"]), encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp_name:
            run_dir = Path(tmp_name)
            model = FakeModel()
            tracker = _BestCheckpointTracker(run_dir, metric_name="mean_iou")

            model.value = 1
            self.assertTrue(tracker.update(epoch=1, metrics={"mean_iou": 0.2}, model=model, torch=FakeTorch))
            model.value = 2
            self.assertFalse(tracker.update(epoch=2, metrics={"mean_iou": 0.1}, model=model, torch=FakeTorch))
            model.value = 3
            self.assertTrue(tracker.update(epoch=3, metrics={"mean_iou": 0.3}, model=model, torch=FakeTorch))

            self.assertEqual((run_dir / "checkpoints" / "best_mean_iou.pt").read_text(encoding="utf-8"), "3")
            metadata = (run_dir / "checkpoints" / "best_mean_iou.json").read_text(encoding="utf-8")
            self.assertIn('"epoch": 3', metadata)
            self.assertEqual([name for _, name in FakeTorch.saves], ["best_mean_iou.pt", "best_mean_iou.pt"])

    def test_resolve_resume_checkpoint_records_metadata_from_best_checkpoint_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            checkpoint = root / "best_mean_iou.pt"
            checkpoint.write_text("weights", encoding="utf-8")
            (root / "best_mean_iou.json").write_text(
                json.dumps({"epoch": 69, "metric_value": 0.409574}, sort_keys=True),
                encoding="utf-8",
            )
            config = {"training": {"resume_from_checkpoint": str(checkpoint)}}

            metadata = _resolve_resume_checkpoint(config)

        self.assertEqual(metadata["resume_from_checkpoint"], str(checkpoint))
        self.assertEqual(metadata["resume_source_epoch"], 69)
        self.assertEqual(metadata["resume_source_metric_value"], 0.409574)
        self.assertTrue(metadata["fine_tune"])

    def test_resolve_resume_checkpoint_rejects_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            config = {"training": {"resume_from_checkpoint": str(Path(tmp_name) / "missing.pt")}}

            with self.assertRaisesRegex(FileNotFoundError, "Resume checkpoint not found"):
                _resolve_resume_checkpoint(config)

    def test_load_resume_checkpoint_wraps_incompatible_state_error(self):
        class FakeModel:
            def load_state_dict(self, state):
                raise RuntimeError("size mismatch")

        class FakeTorch:
            @staticmethod
            def load(path, map_location):
                return {"bad.weight": "shape"}

        with tempfile.TemporaryDirectory() as tmp_name:
            checkpoint = Path(tmp_name) / "bad.pt"
            checkpoint.write_text("bad", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "incompatible"):
                _load_resume_checkpoint(
                    FakeModel(),
                    {"resume_from_checkpoint": str(checkpoint)},
                    device="cpu",
                    torch=FakeTorch,
                )

    def test_prepare_cloud_run_records_resume_metadata_without_reading_test_split(self):
        import csv

        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            bundle = root / "bundle"
            (bundle / "manifests").mkdir(parents=True)
            checkpoint = root / "best_mean_iou.pt"
            checkpoint.write_text("weights", encoding="utf-8")
            (root / "best_mean_iou.json").write_text(json.dumps({"epoch": 69, "metric_value": 0.409574}))
            with (bundle / "manifests" / "cloud_model_input_manifest.csv").open(
                "w", encoding="utf-8", newline=""
            ) as fh:
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
                for split in ["test", "train", "validation"]:
                    writer.writerow(
                        {
                            "sample_id": split,
                            "event_id": split,
                            "class_id": "2",
                            "class_name": "C5_fire",
                            "split": split,
                            "mask_path": f"masks/{split}.tif",
                            "input_channels": "F01;F02",
                            "input_band_paths": "F01:database/a.tif;F02:database/b.tif",
                        }
                    )
            config = root / "config.yaml"
            config.write_text(
                f"""
task_id: task-367c2761a456
experiment_id: P14_manifest_resume_test
seed: 20260508
class_scope: binary_c5
input_channels: [F01, F02]
split_policy:
  train: train
  validation: validation
  test: sealed
  selection_splits: [validation]
  allow_test_split_for_selection: false
model:
  family: unet
  input_channels: 2
  output_classes: 2
metrics:
  class_ids: [0, 2]
  ignore_index: 255
cloud:
  local_full_training_allowed: false
training:
  batch_size: 1
  max_epochs: 1
  resume_from_checkpoint: {checkpoint}
""",
                encoding="utf-8",
            )

            manifest = prepare_cloud_run(
                config_path=config,
                bundle_dir=bundle,
                run_dir=root / "run",
                command_line=["python", "-m", "segmentation_training", "train"],
            )

        self.assertFalse(manifest["test_split_read"])
        self.assertTrue(manifest["fine_tune"])
        self.assertEqual(manifest["resume_source_epoch"], 69)
        self.assertEqual(manifest["resume_source_metric_value"], 0.409574)

    def test_trainable_model_builder_uses_model_registry(self):
        calls = []

        def fake_build_model(model_config, *, input_channels, output_classes):
            calls.append((model_config, input_channels, output_classes))
            return "registered-model"

        config = {
            "input_channels": ["F01", "F02", "F03"],
            "model": {"family": "attention_unet", "base_channels": 4},
        }
        with patch("segmentation_training.cloud.build_model", fake_build_model):
            model = _build_trainable_model(config, source_class_ids=[0, 2])

        self.assertEqual(model, "registered-model")
        self.assertEqual(calls, [(config["model"], 3, 2)])

    def test_limit_records_applies_positive_cloud_smoke_limit_only(self):
        records = ["a", "b", "c"]

        self.assertEqual(_limit_records(records, None), records)
        self.assertEqual(_limit_records(records, 0), records)
        self.assertEqual(_limit_records(records, 2), ["a", "b"])

    def test_cloud_confirm_train_calls_training_runner(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            config = root / "config.yaml"
            bundle = root / "bundle"
            run_dir = root / "run"
            config.write_text("task_id: task-afc7f2c25f8f\n", encoding="utf-8")
            with patch("segmentation_training.cli.run_cloud_training") as runner:
                runner.return_value = {"status": "completed"}

                exit_code = training_main(
                    [
                        "train",
                        "--config",
                        str(config),
                        "--bundle-dir",
                        str(bundle),
                        "--run-dir",
                        str(run_dir),
                        "--cloud-confirm",
                    ]
                )

            self.assertEqual(exit_code, 0)
            runner.assert_called_once()

    def test_channel_contribution_requires_cloud_confirm_and_calls_runner(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            config = root / "config.yaml"
            bundle = root / "bundle"
            checkpoint = root / "best.pt"
            output_dir = root / "channel_contribution"
            config.write_text("task_id: task-afc7f2c25f8f\n", encoding="utf-8")
            checkpoint.write_text("fake", encoding="utf-8")
            without_confirm = training_main(
                [
                    "channel-contribution",
                    "--config",
                    str(config),
                    "--bundle-dir",
                    str(bundle),
                    "--checkpoint",
                    str(checkpoint),
                    "--output-dir",
                    str(output_dir),
                ]
            )
            self.assertEqual(without_confirm, 2)
            with patch("segmentation_training.cli.run_channel_contribution") as runner:
                runner.return_value = {"status": "completed"}

                exit_code = training_main(
                    [
                        "channel-contribution",
                        "--config",
                        str(config),
                        "--bundle-dir",
                        str(bundle),
                        "--checkpoint",
                        str(checkpoint),
                        "--output-dir",
                        str(output_dir),
                        "--cloud-confirm",
                    ]
                )

            self.assertEqual(exit_code, 0)
            runner.assert_called_once()

    def test_dry_run_validates_selection_records_without_test_split(self):
        import csv

        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            config = root / "e1.yaml"
            config.write_text(
                """
task_id: task-afc7f2c25f8f
experiment_id: E1_binary_c5_unet
seed: 20260505
class_scope: binary_c5
input_channels: [F16, F17]
split_policy:
  train: train
  validation: validation
  test: sealed
  selection_splits: [validation]
  allow_test_split_for_selection: false
model:
  family: unet
  input_channels: 2
  output_classes: 2
metrics:
  class_ids: [0, 2]
  ignore_index: 255
cloud:
  local_full_training_allowed: false
""",
                encoding="utf-8",
            )
            with (root / "model_input_manifest.csv").open("w", encoding="utf-8", newline="") as fh:
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
                for split in ["test", "train", "validation"]:
                    writer.writerow(
                        {
                            "sample_id": split,
                            "event_id": split,
                            "class_id": "2",
                            "class_name": "C5_fire",
                            "split": split,
                            "mask_path": f"masks/{split}.tif",
                            "input_channels": "F16;F17",
                            "input_band_paths": "F16:database/a.tif;F17:database/b.tif",
                        }
                    )

            with patch("segmentation_training.cli.validate_record_paths", return_value=[]) as path_validator, patch(
                "segmentation_training.cli.validate_record_raster_grids", return_value=[]
            ) as grid_validator:
                exit_code = training_main(
                    [
                        "dry-run",
                        "--config",
                        str(config),
                        "--contract-dir",
                        str(root),
                        "--max-samples",
                        "2",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual([record.sample_id for record in path_validator.call_args.args[0]], ["train", "validation"])
        self.assertEqual([record.sample_id for record in grid_validator.call_args.args[0]], ["train", "validation"])

    def test_probe_preflight_writes_cloud_smoke_commands_without_test_split(self):
        import csv

        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            bundle = root / "bundle"
            (bundle / "manifests").mkdir(parents=True)
            with (bundle / "manifests" / "cloud_model_input_manifest.csv").open(
                "w", encoding="utf-8", newline=""
            ) as fh:
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
                for split in ["test", "train", "validation"]:
                    writer.writerow(
                        {
                            "sample_id": split,
                            "event_id": split,
                            "class_id": "2",
                            "class_name": "C5_fire",
                            "split": split,
                            "mask_path": f"masks/{split}.tif",
                            "input_channels": "F01;F02;F03",
                            "input_band_paths": "F01:database/a.tif;F02:database/b.tif;F03:database/c.tif",
                        }
                    )
            config_a = root / "deeplab_smoke.yaml"
            config_b = root / "rgb_unet_smoke.yaml"
            for config_path, family, experiment_id in [
                (config_a, "deeplabv3_plus", "probe_deeplab_smoke"),
                (config_b, "unet", "probe_rgb_unet_smoke"),
            ]:
                config_path.write_text(
                    f"""
task_id: task-ead45ca02073
experiment_id: {experiment_id}
seed: 20260507
class_scope: binary_c5
input_channels: [F01, F02, F03]
split_policy:
  train: train
  validation: validation
  test: sealed
  selection_splits: [validation]
  allow_test_split_for_selection: false
model:
  family: {family}
  input_channels: 3
  output_classes: 2
  base_channels: 4
metrics:
  class_ids: [0, 2]
  ignore_index: 255
  threshold_sweep: [0.5]
cloud:
  local_full_training_allowed: false
  max_train_samples: 4
  max_validation_samples: 2
training:
  batch_size: 1
  max_epochs: 1
  window_size: 256
  loss: weighted_cross_entropy_dice
  sampler:
    train_policy: foreground_biased
    foreground_probability: 0.5
    min_foreground_pixels: 1
    validation_policy: foreground_center
""",
                    encoding="utf-8",
                )

            output_dir = root / "probe_preflight"
            with patch("segmentation_training.cli.validate_record_paths", return_value=[]) as path_validator, patch(
                "segmentation_training.cli.validate_record_raster_grids", return_value=[]
            ) as grid_validator:
                exit_code = training_main(
                    [
                        "probe-preflight",
                        "--bundle-dir",
                        str(bundle),
                        "--output-dir",
                        str(output_dir),
                        "--run-root",
                        "workspace/runs/segmentation/controlled_probe_smoke",
                        "--config",
                        str(config_a),
                        "--config",
                        str(config_b),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                [record.sample_id for record in path_validator.call_args_list[0].args[0]],
                ["train", "validation"],
            )
            self.assertEqual(
                [record.sample_id for record in grid_validator.call_args_list[0].args[0]],
                ["train", "validation"],
            )
            report = (output_dir / "probe_preflight_report.md").read_text(encoding="utf-8")
            self.assertIn("Probe Preflight Report", report)
            self.assertIn("test_split_read=false", report)
            self.assertIn("Full training remains blocked", report)
            commands = (output_dir / "cloud_smoke_commands.sh").read_text(encoding="utf-8")
            self.assertIn("probe_deeplab_smoke", commands)
            self.assertIn("--cloud-confirm", commands)
            matrix = (output_dir / "probe_preflight_matrix.csv").read_text(encoding="utf-8")
            self.assertIn("probe_rgb_unet_smoke", matrix)

    def test_model_preflight_instantiates_probe_models_without_data_reads(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            config_a = root / "deeplab_smoke.yaml"
            config_b = root / "rgb_unet_smoke.yaml"
            for config_path, family, experiment_id, channels in [
                (config_a, "deeplabv3_plus", "probe_deeplab_smoke", ["F01", "F02", "F03"]),
                (config_b, "unet", "probe_rgb_unet_smoke", ["F01", "F02", "F03"]),
            ]:
                config_path.write_text(
                    f"""
task_id: task-ead45ca02073
experiment_id: {experiment_id}
seed: 20260507
class_scope: binary_c5
input_channels: [{", ".join(channels)}]
split_policy:
  train: train
  validation: validation
  test: sealed
  selection_splits: [validation]
  allow_test_split_for_selection: false
model:
  family: {family}
  input_channels: {len(channels)}
  output_classes: 2
  base_channels: 2
metrics:
  class_ids: [0, 2]
  ignore_index: 255
  threshold_sweep: [0.5]
cloud:
  local_full_training_allowed: false
  max_train_samples: 4
  max_validation_samples: 2
training:
  batch_size: 1
  max_epochs: 1
  window_size: 64
  loss: weighted_cross_entropy_dice
  sampler:
    train_policy: foreground_biased
    foreground_probability: 0.5
    min_foreground_pixels: 1
    validation_policy: foreground_center
""",
                    encoding="utf-8",
                )

            output_dir = root / "model_preflight"
            exit_code = training_main(
                [
                    "model-preflight",
                    "--output-dir",
                    str(output_dir),
                    "--height",
                    "64",
                    "--width",
                    "64",
                    "--config",
                    str(config_a),
                    "--config",
                    str(config_b),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = (output_dir / "model_preflight_report.md").read_text(encoding="utf-8")
            self.assertIn("Model Preflight Report", report)
            self.assertIn("test_split_read=false", report)
            self.assertIn("does not read data", report)
            matrix = (output_dir / "model_preflight_matrix.csv").read_text(encoding="utf-8")
            self.assertIn("probe_deeplab_smoke", matrix)
            self.assertIn("1x2x64x64", matrix)

    def test_pad_window_makes_small_training_samples_batchable(self):
        channels = np.ones((2, 148, 152), dtype=np.float32)
        label = np.ones((148, 152), dtype=np.uint8)

        padded_channels = _pad_window(channels, 256, fill_value=0, np=np)
        padded_label = _pad_window(label, 256, fill_value=255, np=np)

        self.assertEqual(padded_channels.shape, (2, 256, 256))
        self.assertEqual(padded_label.shape, (256, 256))
        self.assertEqual(float(padded_channels[:, :148, :152].sum()), float(channels.sum()))
        self.assertTrue((padded_label[148:, :] == 255).all())

    def test_progress_logger_prints_and_writes_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            messages = []
            logger = _TrainingProgressLogger(Path(tmp_name), emit=messages.append)

            logger.log_epoch_metrics(epoch=2, max_epochs=5, train_loss=0.25, mean_iou=0.4, foreground_recall=0.5)

            self.assertIn("epoch 2/5", messages[0])
            self.assertIn("loss=0.2500", messages[0])
            self.assertIn("mean_iou=0.4000", messages[0])
            self.assertIn("foreground_recall=0.5000", messages[0])
            self.assertIn("epoch 2/5", (Path(tmp_name) / "logs" / "training_progress.log").read_text())

    def test_normalize_band_replaces_nan_and_inf(self):
        array = np.array([[0.0, np.nan], [np.inf, -np.inf]], dtype=np.float32)

        normalized = _normalize_band(array, np=np)

        self.assertTrue(np.isfinite(normalized).all())

    def test_training_augmentation_applies_spatial_transforms_to_channels_and_label(self):
        channels = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
        label = np.array([[0, 2, 0], [2, 0, 2]], dtype=np.uint8)

        augmented_channels, augmented_label = _apply_training_augmentation(
            channels,
            label,
            {
                "enabled": True,
                "horizontal_flip_probability": 1.0,
                "vertical_flip_probability": 1.0,
                "rotate90_probability": 0.0,
            },
            rng=np.random.default_rng(7),
            np=np,
        )

        self.assertTrue(np.array_equal(augmented_channels, channels[:, ::-1, ::-1]))
        self.assertTrue(np.array_equal(augmented_label, label[::-1, ::-1]))

    def test_record_cache_reads_selected_band_from_multiband_reference(self):
        import csv
        import rasterio
        from rasterio.transform import from_origin
        from segmentation_training.manifest import load_model_input_manifest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "masks").mkdir()
            (root / "bands").mkdir()
            with rasterio.open(
                root / "masks" / "s1.tif",
                "w",
                driver="GTiff",
                height=2,
                width=2,
                count=1,
                dtype="uint8",
                crs="EPSG:4326",
                transform=from_origin(0, 1, 0.1, 0.1),
            ) as dst:
                dst.write(np.ones((1, 2, 2), dtype="uint8"))
            selected_band = np.array([[10, 20], [40, 30]], dtype="float32")
            with rasterio.open(
                root / "bands" / "multi.tif",
                "w",
                driver="GTiff",
                height=2,
                width=2,
                count=2,
                dtype="float32",
                crs="EPSG:4326",
                transform=from_origin(0, 1, 0.1, 0.1),
            ) as dst:
                dst.write(np.stack([np.array([[1, 2], [3, 4]], dtype="float32"), selected_band]))
            manifest_path = root / "model_input_manifest.csv"
            with manifest_path.open("w", encoding="utf-8", newline="") as fh:
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
                        "input_channels": "CH02",
                        "input_band_paths": "CH02:bands/multi.tif#band=2",
                    }
                )

            [record] = load_model_input_manifest(manifest_path, required_channels=("CH02",))
            cache = _build_record_cache([record], bundle_dir=root, channels=("CH02",), rasterio=rasterio, np=np)

        self.assertTrue(np.array_equal(cache["s1"]["channels"][0], _normalize_band(selected_band, np=np)))

    def test_channel_perturbation_zeroes_named_channels_only(self):
        channels = np.arange(3 * 2 * 2, dtype=np.float32).reshape(3, 2, 2)

        perturbed = _apply_channel_perturbation(
            channels,
            input_channels=("F01", "F02", "NDVI"),
            perturbation={"channels": ["F02"], "mode": "zero"},
            np=np,
        )

        self.assertTrue(np.array_equal(perturbed[0], channels[0]))
        self.assertTrue((perturbed[1] == 0).all())
        self.assertTrue(np.array_equal(perturbed[2], channels[2]))

    def test_write_p14_gate_reports_marks_hard_windows_and_tiny_bins(self):
        rows = [
            {
                "sample_id": "sample-000946",
                "foreground_iou": 0.2,
                "label_foreground_pixels": 346,
                "predicted_foreground_pixels": 360,
                "predicted_label_area_ratio": 1.04,
                "foreground_precision": 0.5,
                "foreground_recall": 0.6,
            },
            {
                "sample_id": "sample-001232",
                "foreground_iou": 0.0,
                "label_foreground_pixels": 207,
                "predicted_foreground_pixels": 0,
                "predicted_label_area_ratio": 0.0,
                "foreground_precision": 0.0,
                "foreground_recall": 0.0,
            },
            {
                "sample_id": "sample-regular-tiny",
                "foreground_iou": 0.5,
                "label_foreground_pixels": 40,
                "predicted_foreground_pixels": 42,
                "predicted_label_area_ratio": 1.05,
                "foreground_precision": 0.7,
                "foreground_recall": 0.8,
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_name:
            run_dir = Path(tmp_name)
            _write_p14_gate_reports(run_dir, rows)
            hard_gate = (run_dir / "diagnostics" / "p14_hard_window_gate.csv").read_text(encoding="utf-8")
            tiny_gate = (run_dir / "diagnostics" / "p14_tiny_bin_gate.csv").read_text(encoding="utf-8")

        self.assertIn("sample-000946", hard_gate)
        self.assertIn("improved", hard_gate)
        self.assertIn("sample-001232", hard_gate)
        self.assertIn("regressed", hard_gate)
        self.assertIn("fg_lt_50", tiny_gate)
        self.assertIn("fg_lt_500", tiny_gate)

    def test_read_training_window_can_use_cached_arrays_without_rasterio(self):
        from types import SimpleNamespace

        label = np.zeros((6, 6), dtype=np.uint8)
        label[2:4, 2:4] = 2
        channels = np.stack([np.ones((6, 6), dtype=np.float32), np.full((6, 6), 2.0, dtype=np.float32)], axis=0)
        record = SimpleNamespace(sample_id="s1")

        batch_channels, batch_label = _read_training_window(
            record,
            bundle_dir=Path("unused"),
            channels=("F01", "F02"),
            window_size=4,
            sampler_config={"train_policy": "foreground_biased", "foreground_probability": 1.0},
            foreground_class_ids=[2],
            rng=np.random.default_rng(7),
            rasterio=None,
            RasterWindow=None,
            np=np,
            record_cache={"s1": {"channels": channels, "label": label}},
        )

        self.assertEqual(batch_channels.shape, (2, 4, 4))
        self.assertEqual(batch_label.shape, (4, 4))
        self.assertGreater(int((batch_label == 2).sum()), 0)

    def test_read_validation_window_can_use_cached_arrays_without_rasterio(self):
        from types import SimpleNamespace

        label = np.zeros((6, 6), dtype=np.uint8)
        label[4, 4] = 2
        channels = np.stack([np.ones((6, 6), dtype=np.float32), np.full((6, 6), 2.0, dtype=np.float32)], axis=0)
        record = SimpleNamespace(sample_id="s1")

        batch_channels, batch_label = _read_validation_window(
            record,
            bundle_dir=Path("unused"),
            channels=("F01", "F02"),
            window_size=4,
            sampler_config={"validation_policy": "foreground_center"},
            foreground_class_ids=[2],
            rasterio=None,
            RasterWindow=None,
            np=np,
            record_cache={"s1": {"channels": channels, "label": label}},
        )

        self.assertEqual(batch_channels.shape, (2, 4, 4))
        self.assertEqual(batch_label.shape, (4, 4))
        self.assertGreater(int((batch_label == 2).sum()), 0)

    def test_target_has_valid_pixels_rejects_all_ignore(self):
        self.assertFalse(_target_has_valid_pixels(np.full((8, 8), 255, dtype=np.uint8), np=np))
        self.assertTrue(_target_has_valid_pixels(np.array([[255, 1]], dtype=np.uint8), np=np))

    def test_metrics_payload_includes_prediction_summary_and_threshold_sweep(self):
        labels = [np.array([[0, 2], [2, 255]], dtype=np.uint8)]
        predictions = [np.array([[0, 0], [2, 0]], dtype=np.uint8)]
        probabilities = [np.array([[0.01, 0.25], [0.75, 0.0]], dtype=np.float32)]

        payload = _build_metrics_payload(
            labels,
            predictions,
            [0, 2],
            preview_items=[],
            sample_ids=["s1"],
            foreground_probabilities=probabilities,
            thresholds=[0.5, 0.2],
            np=np,
        )

        self.assertEqual(payload["prediction_summary"][0]["sample_id"], "s1")
        self.assertEqual(payload["prediction_summary"][0]["label_foreground_pixels"], 2)
        self.assertEqual(payload["threshold_sweep"][0]["threshold"], 0.5)
        self.assertEqual(payload["threshold_sweep"][1]["recall"], 1.0)

    def test_write_run_outputs_writes_diagnostics_files(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            run_dir = Path(tmp_name)
            payload = {
                "mean_iou": 0.25,
                "foreground_recall": 0.5,
                "per_class": [],
                "confusion_matrix": np.zeros((2, 2), dtype=np.int64),
                "area": {0: {"label_pixels": 3, "predicted_pixels": 4}, 2: {"label_pixels": 2, "predicted_pixels": 1}},
                "preview_items": [],
                "prediction_summary": [
                    {
                        "sample_id": "s1",
                        "label_foreground_pixels": 2,
                        "predicted_foreground_pixels": 1,
                        "predicted_label_area_ratio": 0.5,
                        "tp": 1,
                        "fp": 0,
                        "fn": 1,
                        "foreground_precision": 1.0,
                        "foreground_recall": 0.5,
                        "foreground_iou": 0.5,
                        "foreground_dice": 0.666,
                        "all_background_prediction": False,
                    }
                ],
                "threshold_sweep": [
                    {
                        "threshold": 0.5,
                        "label_foreground_pixels": 2,
                        "predicted_foreground_pixels": 1,
                        "predicted_label_area_ratio": 0.5,
                        "precision": 1.0,
                        "recall": 0.5,
                        "iou": 0.5,
                        "dice": 0.666,
                    }
                ],
                "foreground_probability_summary": [{"sample_id": "s1", "foreground_probability_max": 0.75}],
                "foreground_window_coverage": [{"sample_id": "s1", "full_mask_foreground_pixels": 2}],
            }

            _write_run_outputs(run_dir, payload, [0, 2])

            self.assertTrue((run_dir / "diagnostics" / "validation_prediction_summary.csv").exists())
            self.assertTrue((run_dir / "diagnostics" / "threshold_sweep.csv").exists())
            self.assertTrue((run_dir / "diagnostics" / "foreground_probability_summary.csv").exists())
            self.assertTrue((run_dir / "diagnostics" / "foreground_window_coverage.csv").exists())
            self.assertTrue((run_dir / "diagnostics" / "artifact_inventory.json").exists())
            self.assertTrue((run_dir / "diagnostics" / "binary_c5_encode_decode_audit.json").exists())

    def test_write_run_outputs_refreshes_inventory_after_prediction_contact_sheet(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            run_dir = Path(tmp_name)
            label = np.array([[0, 2], [0, 2]], dtype=np.uint8)
            prediction = np.array([[0, 2], [2, 0]], dtype=np.uint8)
            payload = {
                "mean_iou": 0.25,
                "foreground_recall": 0.5,
                "per_class": [],
                "confusion_matrix": np.zeros((2, 2), dtype=np.int64),
                "area": {0: {"label_pixels": 2, "predicted_pixels": 2}, 2: {"label_pixels": 2, "predicted_pixels": 2}},
                "preview_items": [("s1", np.ones((2, 2, 2), dtype=np.float32), label, prediction)],
                "prediction_summary": [],
                "threshold_sweep": [],
                "foreground_probability_summary": [],
                "foreground_window_coverage": [],
            }

            _write_run_outputs(run_dir, payload, [0, 2])

            inventory = json.loads((run_dir / "diagnostics" / "artifact_inventory.json").read_text(encoding="utf-8"))
            by_path = {item["path"]: item["exists"] for item in inventory["artifacts"]}
            self.assertTrue(by_path["predictions/validation_contact_sheet.png"])

    def test_write_run_outputs_can_export_all_previews_and_worst_false_positives(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            run_dir = Path(tmp_name)
            preview_items = []
            prediction_summary = []
            for index, fp in enumerate([4, 30, 12], start=1):
                sample_id = f"s{index}"
                label = np.array([[0, 2], [0, 0]], dtype=np.uint8)
                prediction = np.array([[2, 2], [2, 0]], dtype=np.uint8)
                preview_items.append((sample_id, np.ones((2, 2, 2), dtype=np.float32), label, prediction))
                prediction_summary.append(
                    {
                        "sample_id": sample_id,
                        "label_foreground_pixels": 1,
                        "predicted_foreground_pixels": fp + 1,
                        "predicted_label_area_ratio": float(fp + 1),
                        "tp": 1,
                        "fp": fp,
                        "fn": 0,
                        "foreground_precision": 1.0 / float(fp + 1),
                        "foreground_recall": 1.0,
                        "foreground_iou": 1.0 / float(fp + 1),
                        "foreground_dice": 2.0 / float(fp + 2),
                        "all_background_prediction": False,
                    }
                )
            payload = {
                "mean_iou": 0.25,
                "foreground_recall": 0.5,
                "per_class": [],
                "confusion_matrix": np.zeros((2, 2), dtype=np.int64),
                "area": {0: {"label_pixels": 3, "predicted_pixels": 1}, 2: {"label_pixels": 1, "predicted_pixels": 3}},
                "preview_items": preview_items,
                "prediction_summary": prediction_summary,
                "threshold_sweep": [],
                "foreground_probability_summary": [],
                "foreground_window_coverage": [],
            }

            _write_run_outputs(
                run_dir,
                payload,
                [0, 2],
                preview_config={"max_items": "all", "worst_false_positive_items": 2},
            )

            self.assertEqual(len(list((run_dir / "predictions" / "per_sample").glob("*.png"))), 3)
            worst_files = sorted((run_dir / "predictions" / "worst_false_positives").glob("*.png"))
            self.assertEqual(len(worst_files), 2)
            self.assertIn("s2", worst_files[0].name)
            worst_summary = (run_dir / "diagnostics" / "worst_false_positives.csv").read_text(encoding="utf-8")
            self.assertIn("s2", worst_summary.splitlines()[1])

    def test_maybe_write_training_curves_refreshes_existing_inventory_after_curves(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            run_dir = Path(tmp_name)
            (run_dir / "logs").mkdir()
            (run_dir / "logs" / "training_progress.log").write_text(
                "\n".join(
                    [
                        "[2026-05-06T12:00:00+00:00] epoch 1/1 batch 1/1 (100.0%) loss=0.5 avg_loss=0.5",
                        "[2026-05-06T12:00:01+00:00] epoch 1/1 validation loss=0.5 mean_iou=0.2 foreground_recall=0.3",
                    ]
                ),
                encoding="utf-8",
            )
            payload = {
                "mean_iou": 0.25,
                "foreground_recall": 0.5,
                "per_class": [],
                "confusion_matrix": np.zeros((2, 2), dtype=np.int64),
                "area": {0: {"label_pixels": 2, "predicted_pixels": 2}, 2: {"label_pixels": 2, "predicted_pixels": 2}},
                "preview_items": [],
                "prediction_summary": [],
                "threshold_sweep": [],
                "foreground_probability_summary": [],
                "foreground_window_coverage": [],
            }
            _write_run_outputs(run_dir, payload, [0, 2])

            result = _maybe_write_training_curves(run_dir)

            self.assertEqual(result["status"], "written")
            inventory = json.loads((run_dir / "diagnostics" / "artifact_inventory.json").read_text(encoding="utf-8"))
            by_path = {item["path"]: item["exists"] for item in inventory["artifacts"]}
            self.assertTrue(by_path["training_curves.png"])
            self.assertTrue(by_path["training_curves.csv"])

    def test_maybe_write_training_curves_writes_visual_feedback_from_progress_log(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            run_dir = Path(tmp_name)
            (run_dir / "logs").mkdir()
            (run_dir / "logs" / "training_progress.log").write_text(
                "\n".join(
                    [
                        "[2026-05-06T12:00:00+00:00] epoch 1/1 batch 1/1 (100.0%) loss=0.5 avg_loss=0.5",
                        "[2026-05-06T12:00:01+00:00] epoch 1/1 validation loss=0.5 mean_iou=0.2 foreground_recall=0.3",
                    ]
                ),
                encoding="utf-8",
            )

            result = _maybe_write_training_curves(run_dir)

            self.assertEqual(result["status"], "written")
            self.assertTrue((run_dir / "training_curves.png").exists())
            self.assertTrue((run_dir / "training_curves.csv").exists())

    def test_prepare_cloud_run_manifest_does_not_summarize_test_split(self):
        import csv

        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            bundle = root / "bundle"
            (bundle / "manifests").mkdir(parents=True)
            config = root / "e1.yaml"
            config.write_text(
                """
task_id: task-afc7f2c25f8f
experiment_id: E1_binary_c5_unet
seed: 20260505
class_scope: binary_c5
input_channels: [F16, F17]
split_policy:
  train: train
  validation: validation
  test: sealed
  selection_splits: [validation]
  allow_test_split_for_selection: false
model:
  family: unet
  input_channels: 2
  output_classes: 2
metrics:
  class_ids: [0, 2]
  ignore_index: 255
cloud:
  local_full_training_allowed: false
""",
                encoding="utf-8",
            )
            with (bundle / "manifests" / "cloud_model_input_manifest.csv").open("w", encoding="utf-8", newline="") as fh:
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
                for split in ["train", "validation", "test"]:
                    writer.writerow(
                        {
                            "sample_id": split,
                            "event_id": split,
                            "class_id": "2",
                            "class_name": "C5_fire",
                            "split": split,
                            "mask_path": f"masks/{split}.tif",
                            "input_channels": "F16;F17",
                            "input_band_paths": "F16:database/a.tif;F17:database/b.tif",
                        }
                    )

            manifest = prepare_cloud_run(
                config_path=config,
                bundle_dir=bundle,
                run_dir=root / "run",
                command_line=["python", "-m", "segmentation_training", "train"],
            )

        self.assertEqual(manifest["split_counts"], {"train": 1, "validation": 1})
        self.assertFalse(manifest["test_split_read"])

    def test_foreground_window_coverage_rows_exclude_test_split(self):
        import rasterio
        from rasterio.transform import from_origin
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as tmp_name:
            bundle_dir = Path(tmp_name)
            masks_dir = bundle_dir / "masks"
            masks_dir.mkdir()
            positive = np.zeros((8, 8), dtype=np.uint8)
            positive[1:3, 1:3] = 2
            for name, array in {"train.tif": positive, "test.tif": positive}.items():
                with rasterio.open(
                    masks_dir / name,
                    "w",
                    driver="GTiff",
                    height=8,
                    width=8,
                    count=1,
                    dtype="uint8",
                    transform=from_origin(0, 8, 1, 1),
                ) as ds:
                    ds.write(array, 1)
            records = [
                SimpleNamespace(sample_id="train", split="train", mask_path="masks/train.tif"),
                SimpleNamespace(sample_id="test", split="test", mask_path="masks/test.tif"),
            ]

            rows = _foreground_window_coverage_rows(
                records,
                bundle_dir=bundle_dir,
                foreground_class_ids=[2],
                window_size=4,
                rasterio=rasterio,
                np=np,
            )

        self.assertEqual([row["sample_id"] for row in rows], ["train"])
        self.assertGreater(rows[0]["full_mask_foreground_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
