import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation_training.research_route import load_research_route, validate_research_route  # noqa: E402


class SegmentationTrainingResearchRouteTests(unittest.TestCase):
    def test_c2_c5_foundation_route_locks_project_level_choices(self):
        route = load_research_route(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "research_routes"
            / "c2_c5_disaster_aware_foundation.yaml"
        )

        self.assertEqual(route["target_outcome"], "paper_experiment")
        self.assertEqual(route["contribution_type"], "method_plus_experiment")
        self.assertEqual(route["disaster_scope"], ["C2", "C5"])
        self.assertEqual(route["primary_backbone"], "Prithvi")
        self.assertEqual(route["extension_backbone"], "TerraMind")
        self.assertTrue(route["per_disaster_reporting"])
        self.assertTrue(route["category_unknown_required"])
        self.assertEqual(route["sealed_test_policy"], "sealed_until_final")
        self.assertEqual([stage["id"] for stage in route["stages"]], [f"E{index}" for index in range(8)])
        self.assertIn("classical_remote_sensing", route["baseline_ladder"])
        self.assertIn("foundation_prithvi", route["baseline_ladder"])
        self.assertIn("disaster_aware_prithvi", route["baseline_ladder"])
        self.assertTrue(route["acceptance"]["c2_improvement_required"])

    def test_route_requires_both_disasters(self):
        route = load_research_route(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "research_routes"
            / "c2_c5_disaster_aware_foundation.yaml"
        )
        route["disaster_scope"] = ["C5"]

        with self.assertRaisesRegex(ValueError, "C2 and C5"):
            validate_research_route(route)

    def test_route_rejects_unsealed_test_policy(self):
        route = load_research_route(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "research_routes"
            / "c2_c5_disaster_aware_foundation.yaml"
        )
        route["sealed_test_policy"] = "validation_only"

        with self.assertRaisesRegex(ValueError, "sealed"):
            validate_research_route(route)


if __name__ == "__main__":
    unittest.main()
