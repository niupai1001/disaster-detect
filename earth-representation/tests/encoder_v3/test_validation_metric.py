import unittest
from encoder_v3.validation_metric import aggregate_validation, ValidationUnavailable


def case(case_id, kind, source, observation, loss, *, hidden=None, bands=None, P=60.0):
    return dict(case_id=case_id, kind=kind, source=source, observation=observation, loss=loss, hidden_count=hidden, band_count=bands, P_m=P)


class ValidationMetricChecks(unittest.TestCase):
    def base(self):
        return [
            case("a1", "A", "s1", "o1", 1.0), case("a2", "A", "s1", "o1", 3.0),
            case("b1s", "B1", "s1", "o1", 1.0, hidden=1, bands=4),
            case("b1s2", "B1", "s1", "o1", 3.0, hidden=1, bands=4),
            case("b1m2", "B1", "s1", "o1", 4.0, hidden=2, bands=4),
            case("b1m3", "B1", "s1", "o1", 8.0, hidden=3, bands=4),
            case("b2", "B2", "s1", "o1", 6.0),
        ]

    def test_hand_computed_hierarchy(self):
        result = aggregate_validation(self.base())
        self.assertAlmostEqual(result["A"], 2.0)
        self.assertAlmostEqual(result["B1"], 4.0)  # 0.5*2 + 0.5*6
        self.assertAlmostEqual(result["B2"], 6.0)
        self.assertAlmostEqual(result["B"], 5.0)
        self.assertAlmostEqual(result["S"], 3.5)
        self.assertEqual(result["status"], "OK")

    def test_order_independent(self):
        cases = self.base()
        self.assertEqual(aggregate_validation(cases[::-1]), aggregate_validation(cases))

    def test_duplicate_case_rejected(self):
        cases = self.base() + [self.base()[0]]
        with self.assertRaisesRegex(ValidationUnavailable, "duplicate"):
            aggregate_validation(cases)

    def test_missing_multi_hidden_count_rejected(self):
        cases = [item for item in self.base() if item["case_id"] != "b1m3"]
        with self.assertRaisesRegex(ValidationUnavailable, "missing hidden counts"):
            aggregate_validation(cases)

    def test_missing_b2_keeps_formal_scalar_unavailable(self):
        cases = [item for item in self.base() if item["kind"] != "B2"]
        with self.assertRaisesRegex(ValidationUnavailable, "B2"):
            aggregate_validation(cases)
        diagnostic = aggregate_validation(cases, require_b2=False)
        self.assertIsNone(diagnostic["S"])
        self.assertEqual(diagnostic["status"], "B2_UNAVAILABLE")

    def test_nonfinite_and_source_imbalance(self):
        bad = self.base()
        bad[0] = dict(bad[0], loss=float("nan"))
        with self.assertRaisesRegex(ValidationUnavailable, "nonfinite"):
            aggregate_validation(bad)
        cases = self.base()
        cases += [case("a3", "A", "s2", "o2", 10.0), case("a4", "A", "s2", "o2", 30.0)]
        result = aggregate_validation(cases)
        self.assertAlmostEqual(result["A"], 11.0)  # s1=2, s2=20, equal sources -> 11


if __name__ == "__main__":
    unittest.main()
