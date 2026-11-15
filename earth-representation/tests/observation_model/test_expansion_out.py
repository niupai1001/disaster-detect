"""The completion run must be able to leave the frozen expansion release intact."""
import importlib
import os
import unittest
from pathlib import Path

from observation_model import expansion_data


class ExpansionOutputTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("DSH_EXPANSION_OUT", None)
        os.environ.pop("DSH_EXPANSION_CAP", None)
        importlib.reload(expansion_data)

    def test_default_output_is_the_frozen_release(self):
        self.assertEqual(
            expansion_data.OUT,
            expansion_data.ROOT / "observation_model/outputs/encoder_v2/data_expansion",
        )

    def test_output_override_keeps_default_when_unset(self):
        os.environ["DSH_EXPANSION_OUT"] = str(expansion_data.ROOT / "outputs/observation_model/encoder_v2/data_expansion_v2")
        module = importlib.reload(expansion_data)
        self.assertEqual(module.OUT.name, "data_expansion_v2")
        self.assertNotEqual(module.OUT, module._DEFAULT_OUT)
        os.environ.pop("DSH_EXPANSION_OUT", None)
        module = importlib.reload(expansion_data)
        self.assertEqual(module.OUT, module._DEFAULT_OUT)

    def test_network_cap_override_can_only_tighten(self):
        parent, srf = 94_823_570, 7_000_000
        full = 10 * 1024**3 - parent - srf
        self.assertEqual(expansion_data.network_cap(parent, srf), full)
        self.assertEqual(expansion_data.network_cap(parent, srf, "2000000000"), 2_000_000_000)
        # An explicit override replaces the phase cap so a gap-filling run can exceed it.
        self.assertEqual(expansion_data.network_cap(parent, srf, str(full + 1)), full + 1)
        self.assertEqual(expansion_data.network_cap(parent, srf, "30000000000"), 30_000_000_000)


if __name__ == "__main__":
    unittest.main()
