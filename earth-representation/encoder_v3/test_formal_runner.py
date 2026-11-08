import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from .formal_runner import panel_validator, run_route
from .formal_panels import build_validation_panel_document
from .formal_schedule import build_formal_candidate_schedule
from .formal_test_fixtures import populate
from .participation_layout import AreaEquivalentHillRate
from .participation_encoder import merge_recipe
from .prepare_pilot import bounds
from .route_control import RouteProgress, frozen_identity
from .route_step import RouteSchedule
from .scheduled_data import FrozenABSchedule
from .training import restore_checkpoint, save_checkpoint
from .validation_panels import FrozenValidationPanels
from .validation_metric import ValidationUnavailable


class ScheduleFixture:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def prepare(self, index, *, cell_indexed, retain_layouts):
        if not 0 <= index < self.length:
            raise ValueError("outside fixture")
        return dict(views=[index], AB_input_signature=hashlib.sha256(str(index).encode()).hexdigest())


class RunnerChecks(unittest.TestCase):
    def config(self):
        config = json.loads((Path(__file__).with_name("config.json")).read_text(encoding="utf-8"))
        config["required_before_training"] = {key: None for key in config["required_before_training"]}
        populate(config)
        config["m9_task_head_training_authorized"] = False
        return config

    def test_route_schedule_explicit_controls_and_state(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)
        schedule = RouteSchedule(optimizer, peak_lr=1e-3, threshold=.1, warmup_updates=2, validation_interval=3, max_updates=5)
        schedule.before_update(0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5e-4)
        schedule.before_update(1)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-3)
        schedule.before_update(2)
        schedule.validate(1.0, completed_update=3)
        schedule.before_update(3)
        state = schedule.state_dict()
        restored = RouteSchedule(optimizer, peak_lr=1e-3, threshold=.1, warmup_updates=2, validation_interval=3, max_updates=5)
        restored.load_state_dict(state)
        self.assertEqual(restored.last_update, 3)
        with self.assertRaisesRegex(ValueError, "route cursor"):
            schedule.before_update(5)

    def test_runner_advances_only_through_frozen_schedule_and_limit(self):
        progress = RouteProgress("shared", 0, "a" * 64)
        calls = []

        def fake_update(**kwargs):
            calls.append(kwargs["expected_input_signature"])
            return replace(kwargs["progress"], logical_update=kwargs["progress"].logical_update + 1), {"logical_update": kwargs["progress"].logical_update + 1, "status": "fake"}

        with patch("encoder_v3.formal_runner.move_views", side_effect=lambda views, device: views), \
             patch("encoder_v3.formal_runner.execute_route_update", side_effect=fake_update):
            with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
                result = run_route(
                    config=self.config(), frozen_ids={}, progress=progress, modules={}, optimizer=None,
                    scheduler=None, schedule=ScheduleFixture(5), directory=Path(folder), device="cpu",
                    control=dict(max_updates=2, max_wall_seconds=10.0, cell_indexed=False, retain_layouts=False),
                )
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["updates"], 2)
        self.assertEqual(result["progress"].logical_update, 2)
        self.assertEqual(result["reason"], "update_limit")

    def test_runner_rejects_checkpoint_input_mismatch_and_jepa(self):
        progress = RouteProgress("shared", 0, "a" * 64)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            with self.assertRaisesRegex(ValueError, "checkpoint expectation"):
                run_route(
                    config=self.config(), frozen_ids={}, progress=progress, modules={}, optimizer=None,
                    scheduler=None, schedule=ScheduleFixture(2), directory=Path(folder), device="cpu",
                    control=dict(max_updates=1, max_wall_seconds=10.0, cell_indexed=False, retain_layouts=False),
                    expected_input_signature="b" * 64,
                )
            with self.assertRaisesRegex(ValueError, "JEPA route"):
                run_route(
                    config=self.config(), frozen_ids={}, progress=RouteProgress("A+B+JEPA", 2000, "a" * 64, prefix_update=2000, prefix_checkpoint_sha256="c" * 64),
                    modules={}, optimizer=None, scheduler=None, schedule=ScheduleFixture(2),
                    directory=Path(folder), device="cpu",
                    control=dict(max_updates=1, max_wall_seconds=10.0, cell_indexed=False, retain_layouts=False),
                )

    def test_candidate_schedule_builder_is_loader_compatible(self):
        study = json.loads((Path(__file__).with_name("reports") / "pilot_study_candidate.json").read_text(encoding="utf-8"))
        members = study["training_members"]

        class Dataset:
            role = "TRAIN"
            identities = study["source_identities"]
            rows = members

        curve = AreaEquivalentHillRate(.8092248954506467, 4.010953026221307, 1.219268281571556)
        document = build_formal_candidate_schedule(Dataset(), members, updates=2, seed=0,
                                                   mask_rule=study["rectangle_proposal"], p_values=(60.0, 120.0))
        self.assertEqual(len(document["updates"]), 2)
        self.assertEqual([entry["P_m"] for entry in document["updates"]], [60.0, 120.0])
        self.assertEqual(len(document["eligible_members"]), 8)
        for entry in document["updates"]:
            self.assertEqual(len(entry["slots"]), 8)
            for slot in entry["slots"]:
                self.assertTrue(slot["A_rectangle"][0] < slot["A_rectangle"][2])
                self.assertTrue(slot["B"]["hidden_band_ids"])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder) / "SYNTHETIC_SCHEDULE.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            config = self.config()
            config["formal_step"] = {"chunk_pixels": 7}
            populate(config)
            frozen = dict(formal_config_sha256=frozen_identity(config), source_identities=Dataset.identities,
                          merge_recipe=merge_recipe(curve, 0), AB_schedule_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            loader = FrozenABSchedule(path, config=config, frozen_ids=frozen, dataset=Dataset(), rate_curve=curve, selection_seed=0)
            self.assertEqual(len(loader), 2)

    def test_runner_early_stop_uses_explicit_metric_patience(self):
        progress = RouteProgress("shared", 0, "a" * 64)

        def fake_update(**kwargs):
            return replace(kwargs["progress"], logical_update=kwargs["progress"].logical_update + 1), {"validation": {"metric": 1.0}}

        with patch("encoder_v3.formal_runner.move_views", side_effect=lambda views, device: views), \
             patch("encoder_v3.formal_runner.execute_route_update", side_effect=fake_update):
            with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
                result = run_route(
                    config=self.config(), frozen_ids={}, progress=progress, modules={}, optimizer=None,
                    scheduler=None, schedule=ScheduleFixture(5), directory=Path(folder), device="cpu",
                    control=dict(max_updates=5, max_wall_seconds=10.0, cell_indexed=False, retain_layouts=False,
                                 early_stop=dict(patience=2, min_delta=.1)),
                )
        self.assertEqual(result["updates"], 3)
        self.assertEqual(result["reason"], "early_stop")

    def test_candidate_panel_builder_is_loader_compatible(self):
        study = json.loads((Path(__file__).with_name("reports") / "pilot_study_candidate.json").read_text(encoding="utf-8"))
        members = study["validation_members"]

        class Dataset:
            role = "VAL"
            identities = study["source_identities"]
            rows = members

        row = members[0]
        left, bottom, right, top = bounds(row)
        cases = [
            dict(panel_id="a60", source_id=row["id"], kind="A", P_m=60.0, origin_xy=[left, bottom], rectangle=[left, bottom, right, top]),
            dict(panel_id="b1single", source_id=row["id"], kind="B1", P_m=60.0, origin_xy=[left, bottom], hidden_band_ids=[row["bands"][0]["band_key"]]),
            dict(panel_id="b1multi", source_id=row["id"], kind="B1", P_m=120.0, origin_xy=[left, bottom], hidden_band_ids=[row["bands"][0]["band_key"], row["bands"][1]["band_key"]]),
        ]
        document = build_validation_panel_document(Dataset(), cases)
        self.assertEqual(len(document["panels"]), 3)
        self.assertEqual(len(document["eligible_members"]), 8)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder) / "SYNTHETIC_PANEL.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            config = self.config()
            config["formal_step"] = {"chunk_pixels": 7}
            populate(config)
            curve = AreaEquivalentHillRate(.8092248954506467, 4.010953026221307, 1.219268281571556)
            frozen = dict(formal_config_sha256=frozen_identity(config), source_identities=Dataset.identities,
                          merge_recipe=merge_recipe(curve, 0), validation_panel_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            panel = FrozenValidationPanels(path, config=config, frozen_ids=frozen, dataset=Dataset(), rate_curve=curve, selection_seed=0)
            self.assertEqual(panel.identity, frozen["validation_panel_sha256"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device-state restore check")
    def test_restore_moves_optimizer_moments_to_parameter_device(self):
        model = torch.nn.Linear(1, 1).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(torch.ones(1, 1, device="cuda")).sum().backward()
        optimizer.step()
        scheduler = RouteSchedule(optimizer, peak_lr=1e-3, threshold=0.)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder) / "SYNTHETIC_CUDA_STATE.pt"
            save_checkpoint(path, modules={"encoder": model}, optimizer=optimizer, scheduler=scheduler,
                            logical_update=1, sampler={"next_update": 1, "next_input_signature": "a" * 64},
                            frozen_ids={"kind": "device_fixture"})
            restore_checkpoint(path, modules={"encoder": model}, optimizer=optimizer, scheduler=scheduler,
                               frozen_ids={"kind": "device_fixture"})
        for parameter in model.parameters():
            for key, value in optimizer.state[parameter].items():
                if torch.is_tensor(value) and value.shape == parameter.shape:
                    self.assertEqual(value.device, parameter.device)

    def test_panel_validator_returns_only_a_finite_scalar(self):
        class Panels:
            def __init__(self, metric):
                self.metric = metric

            def collect(self, encoder, decoder, **kwargs):
                return dict(metric=self.metric, metric_status="fixture")

        self.assertEqual(panel_validator(Panels({"S": 2.5}))({"encoder": None, "decoder": None}, 1), 2.5)
        with self.assertRaises(ValidationUnavailable):
            panel_validator(Panels(None))({"encoder": None, "decoder": None}, 1)


if __name__ == "__main__":
    unittest.main()
