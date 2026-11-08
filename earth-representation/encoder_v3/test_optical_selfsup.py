"""E0 (optical_intra_selfsup_v1) protocol tests: pool, sampler, S_intra, gates, panel."""
import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from .data import Observation
from .formal_panels import build_validation_panel_document
from .formal_schedule import _mask_rule, build_optical_selfsup_schedule, optical_train_pool, ranked_members
from .gates import GATE_IDS, OPTICAL_IDS, OPTICAL_SCOPE, optical_identity_only, payload_sha256, require_optical_selfsup_training
from .model import BandObservation, ReflectanceDecoder
from .participation_encoder import ParticipationEncoder, merge_recipe
from .participation_layout import AreaEquivalentHillRate
from .route_control import frozen_identity
from .scheduled_data import OpticalSelfsupSchedule
from .tasks import hidden_band_indices
from .validation_metric import ValidationUnavailable, aggregate_intra_validation, aggregate_validation
from .validation_panels import FrozenValidationPanels

CURVE = AreaEquivalentHillRate(.8092248954506467, 4.010953026221307, 1.219268281571556)
MASK_RULE = dict(count=1, target_patch_fraction=.5, aspect_ratio="log-uniform [0.5,2.0]",
                 position="uniform feasible patch-aligned rectangle",
                 size_rounding="nearest integer patch count, clipped to observation patch layout")
IDENTITIES = {"members": "SYNTHETIC_METADATA_ONLY", "contracts": "SYNTHETIC", "srf": "SYNTHETIC"}


def _row(identifier, sensor, bands, side, version, role="TRAIN"):
    step = 10.0 if sensor == "S2" else 30.0
    return dict(id=identifier, role=role, sensor=sensor, cache_sha256=version,
                bands=[dict(band_key=f"b{index}", shape=[side, side], transform=[step, 0., 0., 0., -step, side * step])
                       for index in range(bands)])


class DatasetFixture:
    """Synthetic native TRAIN release: 10 S2 and 10 Landsat members, no B2 content."""

    role = "TRAIN"
    identities = IDENTITIES

    def __init__(self, *, members=10, bands=4, side=3, role="TRAIN"):
        self.role = role
        self.rows, self.observations, self.reads, self.stats_reads = [], [], [], []
        plan = [("S2", bands)] * members + [("L8", bands)] * members
        for index, (sensor, band_count) in enumerate(plan):
            identifier = f"fixture{index:02d}"
            version = hashlib.sha256(identifier.encode()).hexdigest()
            self.rows.append(_row(identifier, sensor, band_count, side, version, role))
            step = 10.0 if sensor == "S2" else 30.0
            native = [BandObservation(torch.arange(side * side, dtype=torch.float32).reshape(side, side) / (side * side + 1),
                                      torch.ones(side, side, dtype=torch.bool), torch.tensor([450., 500., 550., 600.]),
                                      torch.tensor([0., 0., 1., 0.]), (step, 0., 0., 0., -step, side * step),
                                      "EPSG:32631", f"b{band}") for band in range(band_count)]
            self.observations.append(Observation(native, dict(id=identifier, role=role, sensor=sensor,
                                                              cache_sha256=version, frozen_source_ids=self.identities)))

    def __getitem__(self, index):
        self.reads.append(index)
        return copy.deepcopy(self.observations[index])

    def engineering_train_stds(self, observation):
        self.stats_reads.append(observation.identifier)
        return {band.record_id: float(index + 1) for index, band in enumerate(observation.bands)}


class OpticalFixtures(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp(dir=Path(__file__).resolve().parent))
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.config = dict(formal_step=dict(chunk_pixels=7), fixture="NO_REAL_DATA_OR_OPERATIONAL_AUTHORIZATION")
        self.dataset = DatasetFixture()

    def recipe(self, dataset=None, *, updates=4, seed=11, slots=4):
        dataset = self.dataset if dataset is None else dataset
        document = build_optical_selfsup_schedule(dataset, updates=updates, seed=seed, slots_per_source=slots, mask_rule=MASK_RULE)
        path = self.folder / "sampling_recipe.json"
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return document, path

    def frozen(self, path, dataset=None):
        dataset = self.dataset if dataset is None else dataset
        return dict(formal_config_sha256=frozen_identity(self.config), source_identities=dataset.identities,
                    merge_recipe=merge_recipe(CURVE, 0), sampling_recipe_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def loader(self, path, dataset=None):
        dataset = self.dataset if dataset is None else dataset
        return OpticalSelfsupSchedule(path, config=self.config, frozen_ids=self.frozen(path, dataset), dataset=dataset,
                                      rate_curve=CURVE, selection_seed=0, gate=optical_identity_only)


class PoolAndSamplerChecks(OpticalFixtures):
    def test_pool_is_training_only_and_counts_its_exclusions(self):
        big = DatasetFixture(members=1, bands=10, side=200)
        big.rows[0]["id"] = "fixtureBig"
        big.observations[0].metadata["id"] = "fixtureBig"
        combined = DatasetFixture(members=3, bands=4, side=3)
        combined.rows.extend(big.rows)
        combined.observations.extend(big.observations)
        pool = optical_train_pool(combined, max_native_pixels=16384)
        self.assertEqual(pool["excluded"]["above_native_pixel_cap"], 2)
        self.assertEqual(pool["excluded_by_sensor"], {"L8:10bands": 1, "S2:10bands": 1})
        self.assertNotIn("fixtureBig", pool["families"]["S2"])
        with self.assertRaisesRegex(ValueError, "TRAIN release"):
            optical_train_pool(DatasetFixture(role="VAL"))

    def test_sampler_is_deterministic_distinct_and_source_balanced(self):
        pool = optical_train_pool(self.dataset)
        first = ranked_members(11, 0, "S2", pool["families"]["S2"], 4)
        self.assertEqual(first, ranked_members(11, 0, "S2", pool["families"]["S2"], 4))
        self.assertEqual(len(set(first)), 4)
        self.assertNotEqual(first, ranked_members(11, 1, "S2", pool["families"]["S2"], 4))
        self.assertEqual(len(set(first) & set(ranked_members(11, 0, "Landsat", pool["families"]["Landsat"], 4))), 0)
        with self.assertRaisesRegex(ValueError, "slot count"):
            ranked_members(11, 0, "S2", pool["families"]["S2"], 999)

    def test_recipe_is_compact_and_never_stores_per_update_tensors(self):
        document, _ = self.recipe()
        self.assertNotIn("updates_detail", document)
        self.assertEqual(document["updates"], 4)
        self.assertEqual(document["pool"]["members"], {"S2": 10, "Landsat": 10})
        self.assertFalse(document["B2_used"])
        self.assertEqual(document["tasks"]["A"].split(";")[0], "rectangle spatial hide")

    def test_hidden_band_boundaries_are_reported_not_silently_changed(self):
        with self.assertRaisesRegex(ValueError, "at least two bands"):
            hidden_band_indices(1, seed=0, update=0, observation_id="x")
        self.assertEqual(len(hidden_band_indices(2, seed=0, update=0, observation_id="x")), 1)
        seen, counts = set(), set()
        for update in range(400):
            hidden = hidden_band_indices(8, seed=3, update=update, observation_id="x")
            self.assertTrue(hidden < set(range(8)))
            self.assertGreaterEqual(len(hidden), 1)
            self.assertLessEqual(len(hidden), 7)
            counts.add(len(hidden))
            seen.add("single" if len(hidden) == 1 else "multi")
        self.assertEqual(seen, {"single", "multi"})
        self.assertTrue({2, 3, 4, 5, 6, 7} <= counts)


class ScheduleLoaderChecks(OpticalFixtures):
    def test_prepare_builds_eight_distinct_members_with_A_and_B1(self):
        _, path = self.recipe()
        loader = self.loader(path)
        self.assertEqual(len(loader), 4)
        packet = loader.prepare(0, cell_indexed=False, retain_layouts=False)
        self.assertEqual(len(packet["source_members"]), 8)
        self.assertEqual(len(set(packet["source_members"])), 8)
        self.assertEqual(len(packet["views"]), 16)
        self.assertEqual(sorted({view.task.kind for view in packet["views"]}), ["A", "B1"])
        self.assertEqual(packet["absent_tasks"], [])
        for view in packet["views"]:
            visible = {band.record_id for band in view.task.visible}
            targets = {target.band.record_id for target in view.task.targets}
            self.assertTrue(visible)
            if view.task.kind == "B1":
                self.assertFalse(visible & targets)
                self.assertEqual(len(view.task.metadata["hidden_band_ids"]), len(targets))
        self.assertEqual(packet["AB_input_signature"], loader.prepare(0, cell_indexed=False, retain_layouts=False)["AB_input_signature"])
        self.assertNotEqual(packet["AB_input_signature"], loader.prepare(1, cell_indexed=False, retain_layouts=False)["AB_input_signature"])

    def test_same_seed_replays_and_new_seed_changes_the_pool_order(self):
        _, path = self.recipe(seed=11)
        first = [self.loader(path).members(index) for index in range(3)]
        second = [self.loader(path).members(index) for index in range(3)]
        self.assertEqual(first, second)
        document, other = self.recipe(seed=12)
        self.assertNotEqual(self.loader(other).members(0), first[0])

    def test_tampered_recipe_pool_and_identity_are_rejected_before_band_reads(self):
        dataset = DatasetFixture()
        _, path = self.recipe(dataset)
        frozen = self.frozen(path, dataset)
        path.write_text(path.read_text(encoding="utf-8").replace('"updates": 4', '"updates": 5'), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "sampling recipe differs"):
            OpticalSelfsupSchedule(path, config=self.config, frozen_ids=frozen, dataset=dataset, rate_curve=CURVE,
                                   selection_seed=0, gate=optical_identity_only)
        _, path = self.recipe(dataset)
        dataset.rows[0]["bands"] = dataset.rows[0]["bands"][:3]
        with self.assertRaisesRegex(ValueError, "pool differs"):
            self.loader(path, dataset)
        with self.assertRaisesRegex(ValueError, "Frozen formal configuration differs"):
            OpticalSelfsupSchedule(path, config=dict(self.config, extra=1), frozen_ids=self.frozen(path),
                                   dataset=self.dataset, rate_curve=CURVE, selection_seed=0, gate=optical_identity_only)
        self.assertEqual(dataset.reads, [])

    def test_real_gate_blocks_the_loader_unless_identity_only_is_explicit(self):
        _, path = self.recipe()
        config = json.loads((Path(__file__).parent / "config.json").read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "Training blocked"):
            OpticalSelfsupSchedule(path, config=config, frozen_ids=self.frozen(path), dataset=self.dataset,
                                   rate_curve=CURVE, selection_seed=0)

    def test_pool_cap_cannot_be_silently_lifted_by_a_recipe(self):
        dataset = DatasetFixture(members=10, bands=4, side=3)
        document, path = self.recipe(dataset)
        document["pool"]["max_native_pixels"] = 1
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "pool differs"):
            self.loader(path, dataset)


class IntraMetricChecks(unittest.TestCase):
    def base(self):
        return [
            dict(case_id="a1", kind="A", source="s1", observation="o1", loss=1.0),
            dict(case_id="a2", kind="A", source="s1", observation="o1", loss=3.0),
            dict(case_id="b1s", kind="B1", source="s1", observation="o1", loss=2.0, hidden_count=1, band_count=4),
            dict(case_id="b1m2", kind="B1", source="s1", observation="o1", loss=4.0, hidden_count=2, band_count=4),
            dict(case_id="b1m3", kind="B1", source="s1", observation="o1", loss=8.0, hidden_count=3, band_count=4),
        ]

    def test_hand_computed_scalar_and_weights(self):
        result = aggregate_intra_validation(self.base())
        self.assertAlmostEqual(result["A"], 2.0)
        self.assertAlmostEqual(result["B1"], 4.0)   # 0.5*2 + 0.5*6
        self.assertAlmostEqual(result["S_intra"], 3.0)
        self.assertEqual(result["weights"], {"A": 0.5, "B1": 0.5})
        self.assertEqual(result["status"], "OK")
        self.assertEqual(aggregate_intra_validation(self.base()[::-1]), result)

    def test_b2_cases_and_unequal_sources_are_refused(self):
        cases = self.base() + [dict(case_id="b2", kind="B2", source="s1", observation="o1", loss=1.0)]
        with self.assertRaisesRegex(ValidationUnavailable, "B2 cases must not enter"):
            aggregate_intra_validation(cases)
        cases = self.base() + [dict(case_id="a9", kind="A", source="s2", observation="o2", loss=5.0)]
        with self.assertRaisesRegex(ValidationUnavailable, "not renormalized"):
            aggregate_intra_validation(cases)

    def test_source_weights_are_equal_and_missing_cases_fail(self):
        cases = self.base() + [
            dict(case_id="a3", kind="A", source="s2", observation="o2", loss=20.0),
            dict(case_id="b3s", kind="B1", source="s2", observation="o2", loss=10.0, hidden_count=1, band_count=4),
            dict(case_id="b3m2", kind="B1", source="s2", observation="o2", loss=20.0, hidden_count=2, band_count=4),
            dict(case_id="b3m3", kind="B1", source="s2", observation="o2", loss=30.0, hidden_count=3, band_count=4),
        ]
        result = aggregate_intra_validation(cases)
        self.assertAlmostEqual(result["A"], 11.0)          # s1=2, s2=20, equal sources
        self.assertAlmostEqual(result["B1"], 10.75)        # s1=4, s2=17.5, equal sources
        with self.assertRaisesRegex(ValidationUnavailable, "no evaluation source"):
            aggregate_intra_validation([case for case in self.base() if case["kind"] != "B1"])
        with self.assertRaisesRegex(ValidationUnavailable, "missing hidden counts"):
            aggregate_intra_validation([case for case in self.base() if case["case_id"] != "b1m3"])

    def test_historical_scalar_is_not_renamed(self):
        with self.assertRaisesRegex(ValidationUnavailable, "B2"):
            aggregate_validation(self.base())
        self.assertIsNone(aggregate_validation(self.base(), require_b2=False)["S"])


class OpticalGateChecks(unittest.TestCase):
    def config(self):
        config = dict(B2_training_required=False, JEPA_training_required=False,
                      m9_task_head_training_authorized=False,
                      required_before_training={gate: None for gate in sorted(OPTICAL_IDS)})
        return config

    def approval(self, config, gate):
        return dict(gate_id=gate, status="APPROVED", scope=OPTICAL_SCOPE, approval_reference="synthetic://e0/" + gate,
                    source_sha256="a" * 64, data_sha256=None, config_sha256=payload_sha256(config),
                    tests=["synthetic://e0/" + gate], reviewer="user", approved_at="2026-11-09T00:00:00Z")

    def test_open_gates_block_and_old_ids_do_not_satisfy_e0(self):
        config = self.config()
        with self.assertRaisesRegex(ValueError, "Optical self-supervised training blocked"):
            require_optical_selfsup_training(config)
        config["required_before_training"] = {gate: None for gate in sorted(GATE_IDS)}
        with self.assertRaisesRegex(ValueError, "Optical self-supervised training blocked") as caught:
            require_optical_selfsup_training(config)
        for gate in sorted(OPTICAL_IDS):
            self.assertIn(gate, str(caught.exception))
        for gate in sorted(OPTICAL_IDS):
            config = self.config()
            config["required_before_training"][gate] = self.approval(config, gate)
            with self.assertRaisesRegex(ValueError, "Optical self-supervised training blocked") as caught:
                require_optical_selfsup_training(config)
            blocked = str(caught.exception)
            self.assertNotIn(gate + ",", blocked.replace("blocked: ", ""))
            self.assertEqual(sorted(OPTICAL_IDS - {gate}), sorted(part.strip() for part in blocked.split(": ")[1].split(",")))

    def approved(self, **overrides):
        config = self.config()
        config.update(overrides)
        for gate in sorted(OPTICAL_IDS):
            config["required_before_training"][gate] = self.approval(config, gate)
        return config

    def test_e0_flags_and_scope_are_checked(self):
        config = self.approved()
        self.assertTrue(require_optical_selfsup_training(config))
        with self.assertRaisesRegex(ValueError, "B2_training_required"):
            require_optical_selfsup_training(self.approved(B2_training_required=True))
        with self.assertRaisesRegex(ValueError, "JEPA_training_required"):
            require_optical_selfsup_training(self.approved(JEPA_training_required=True))
        with self.assertRaisesRegex(ValueError, "Task-head"):
            require_optical_selfsup_training(self.approved(m9_task_head_training_authorized=True))
        wrong = json.loads(json.dumps(config))
        wrong["required_before_training"]["optical_data_freeze"]["scope"] = "formal_training"
        with self.assertRaisesRegex(ValueError, "scope"):
            require_optical_selfsup_training(wrong)
        stale = json.loads(json.dumps(config))
        stale["required_before_training"]["optical_data_freeze"]["config_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "different configuration"):
            require_optical_selfsup_training(stale)

    def test_checked_in_e0_config_remains_unauthorized(self):
        config = json.loads((Path(__file__).parent / "config_optical_selfsup_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(set(config["required_before_training"]), OPTICAL_IDS)
        with self.assertRaisesRegex(ValueError, "Optical self-supervised training blocked"):
            require_optical_selfsup_training(config)


class PanelChecks(OpticalFixtures):
    def test_fixed_panel_collects_intra_scalar_end_to_end(self):
        dataset = DatasetFixture(members=2, bands=4, side=3, role="VAL")
        dataset.identities = IDENTITIES
        cases = []
        for row in dataset.rows:
            step = abs(row["bands"][0]["transform"][0])
            top = row["bands"][0]["transform"][5]
            rectangle = [0.0, top - step, step, top]
            for P in (60.0, 120.0):
                cases.append(dict(panel_id=f"{row['id']}-P{int(P)}-A", source_id=row["id"], kind="A", P_m=P,
                                  origin_xy=[0.0, 0.0], rectangle=rectangle))
                for count in (1, 2, 3):
                    hidden = [band["band_key"] for band in row["bands"][:count]]
                    cases.append(dict(panel_id=f"{row['id']}-P{int(P)}-b{count}", source_id=row["id"], kind="B1", P_m=P,
                                      origin_xy=[0.0, 0.0], hidden_band_ids=hidden))
        document = build_validation_panel_document(dataset, cases, status="SYNTHETIC")
        path = self.folder / "panel.json"
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        frozen = dict(formal_config_sha256=frozen_identity(self.config), source_identities=dataset.identities,
                      merge_recipe=merge_recipe(CURVE, 0), validation_panel_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        panels = FrozenValidationPanels(path, config=self.config, frozen_ids=frozen, dataset=dataset, rate_curve=CURVE,
                                        selection_seed=0, gate=optical_identity_only, metric=aggregate_intra_validation)
        torch.manual_seed(7)
        encoder = ParticipationEncoder(rate_curve=CURVE, selection_seed=0, hyper_hidden=8, local_batch_size=4).eval()
        decoder = ReflectanceDecoder(indexed=True).eval()
        result = panels.collect(encoder, decoder, cell_indexed=False, retain_layouts=False)
        self.assertEqual(result["metric_status"], "OK")
        self.assertIn("S_intra", result["metric"])
        self.assertAlmostEqual(result["metric"]["S_intra"], .5 * (result["metric"]["A"] + result["metric"]["B1"]))
        self.assertEqual(result["optimizer_updates"], 0)
        for row in result["rows"]:
            self.assertIsNotNone(row["result"].get("loss"))
        legacy = FrozenValidationPanels(path, config=self.config, frozen_ids=frozen, dataset=dataset, rate_curve=CURVE,
                                        selection_seed=0, gate=optical_identity_only)
        self.assertIsNone(legacy.collect(encoder, decoder, cell_indexed=False, retain_layouts=False)["metric"])


if __name__ == "__main__":
    unittest.main()
