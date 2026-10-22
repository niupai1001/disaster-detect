"""Protect the actual project protocol against plausible execution handoff errors."""
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("research_preflight", ROOT / "scripts/check_research_protocol.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


class ResearchProtocolTests(unittest.TestCase):
    control = "configs/segformer/v4r1/segformer_b0_direct10_detailres_n100_u3600_s1234.yaml"
    utae = "configs/utae/event_cv/F1.yaml"

    def test_original_baseline_keeps_common_core_normalization(self):
        result = preflight.audit_config(self.control)
        self.assertEqual(result["counts"], {"train": 2223, "dev": 669})
        self.assertEqual(result["normalization_fit_n"], 559)

    def test_fourfold_is_not_the_default_protocol(self):
        with self.assertRaisesRegex(ValueError, "original held-out"):
            preflight.audit_config("configs/segformer/event_cv_v5/foldF1_control_s1234.yaml")

    def test_utae_derived_dem_provenance_has_explicit_limit(self):
        result = preflight.audit_config(self.utae, protocol="EVENT_CV_V5")
        self.assertTrue(result["ok"])
        self.assertTrue(result["warnings"])

    def test_official_config_cannot_become_event_protocol_by_renaming_version(self):
        with self.assertRaises(ValueError):
            preflight.audit_config("configs/segformer/v6_main/temporal_single_baseline.yaml",
                                   ["segformer.data.version=event_cv_v5"])

    def test_wrong_fold_statistics_rejected(self):
        with self.assertRaisesRegex(ValueError, "original held-out"):
            preflight.audit_config(self.control, ["segformer.data.stats=manifests/event_cv_v5/F2/stats_train.json"])

    def test_outer_manifest_cannot_enter_utae_training(self):
        with self.assertRaisesRegex(ValueError, "Manifest paths"):
            preflight.audit_config(self.utae, ["utae_cv.protocol.pool=manifests/event_cv_v5/F1/test.jsonl"], protocol="EVENT_CV_V5")

    def test_utae_role_swap_rejected(self):
        with self.assertRaisesRegex(ValueError, "role mapping"):
            preflight.audit_config(self.utae, ["utae_cv.protocol.roles.train=VAL"], protocol="EVENT_CV_V5")

    def test_wrong_selection_and_seed_rejected(self):
        for override in ["segformer.eval.selection_metric=pooled_iou", "segformer.seed=42"]:
            with self.subTest(override=override), self.assertRaises(ValueError):
                preflight.audit_config(self.control, [override])

    def test_missing_config_does_not_fall_back_to_legacy_defaults(self):
        with self.assertRaises(FileNotFoundError):
            preflight.audit_config("configs/does_not_exist.yaml")

    def test_old_official_scheduler_refuses_default_invocation(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/run_v6_remaining.py")],
                                cwd=ROOT, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn(b"--historical-official", result.stderr)

    def test_dem_scheduler_finds_versioned_training_directory(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            import run_dem_arm
            self.assertEqual(run_dem_arm.discovery("demraw", "F1"),
                             ROOT / "outputs/segformer/event_cv_v5_dem/SegFormer-LateDetail-EventCVV5-F1-DEMSlopeRaw/1234")
        finally:
            sys.path.pop(0)

    def test_fourfold_schedulers_refuse_default_invocation(self):
        for script, args in [("run_utae_cv.py", []), ("run_dem_arm.py", ["--arm", "demraw"])]:
            with self.subTest(script=script):
                result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                                        cwd=ROOT, capture_output=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn(b"--historical-event-cv", result.stderr)


if __name__ == "__main__":
    unittest.main()
