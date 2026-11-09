import json
import unittest
from pathlib import Path
from encoder_v3.formal_test_fixtures import populate, populate_adaptive
from encoder_v3.gates import require_adaptive, require_all, require_jepa, require_training


class FormalGateChecks(unittest.TestCase):
    def config(self):
        return json.loads((Path(__file__).resolve().parents[2] / 'src/encoder_v3' / "config.json").read_text(encoding="utf-8"))

    def test_real_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Training blocked"):
            require_training(self.config())

    def test_plain_strings_do_not_approve(self):
        config = self.config()
        config["required_before_training"] = {key: "APPROVED" for key in config["required_before_training"]}
        with self.assertRaisesRegex(ValueError, "structured approval"):
            require_training(config)

    def test_structured_approval_binds_payload_and_scope(self):
        config = self.config()
        config["required_before_training"] = {key: None for key in config["required_before_training"]}
        populate(config)
        self.assertTrue(require_training(config))
        bad = json.loads(json.dumps(config))
        bad["required_before_training"]["merge_configuration"]["scope"] = "preflight_only"
        with self.assertRaisesRegex(ValueError, "scope"):
            require_training(bad)
        bad = json.loads(json.dumps(config))
        bad["required_before_training"]["optimizer_choice"]["config_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "different configuration"):
            require_training(bad)
        bad = json.loads(json.dumps(config))
        bad["candidate"] = dict(bad["candidate"], peak_lr=0.5)
        with self.assertRaisesRegex(ValueError, "different configuration"):
            require_training(bad)

    def test_jepa_and_adaptive_gates_are_not_plain_strings(self):
        config = self.config()
        with self.assertRaisesRegex(ValueError, "JEPA blocked"):
            require_jepa(config)
        config["required_before_jepa"] = {key: "SYNTHETIC_REVIEW_ONLY" for key in config["required_before_jepa"]}
        with self.assertRaisesRegex(ValueError, "structured approval"):
            require_jepa(config)
        adaptive = json.loads(((Path(__file__).resolve().parents[2] / 'src/encoder_v3') / "adaptive_config.json").read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "adaptive gates"):
            require_adaptive(adaptive)
        populate_adaptive(adaptive)
        self.assertTrue(require_adaptive(adaptive))

    def test_all_requires_task_head_stay_closed(self):
        config = self.config()
        config["required_before_training"] = {key: None for key in config["required_before_training"]}
        populate(config)
        config["m9_task_head_training_authorized"] = True
        populate(config)
        with self.assertRaisesRegex(ValueError, "Task-head"):
            require_all(config)
        config["m9_task_head_training_authorized"] = False
        populate(config)
        self.assertTrue(require_all(config))


if __name__ == "__main__":
    unittest.main()
