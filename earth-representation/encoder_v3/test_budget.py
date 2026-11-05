from pathlib import Path
import tempfile
import unittest
from .budget import PilotBudget


class BudgetChecks(unittest.TestCase):
    def test_reservations_survive_crash_and_caps(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(Path(__file__).resolve().parent))
            path=Path(folder)/'budget.json'
            a=PilotBudget(path,study_sha256='a'*64,total_limit=3,per_candidate_limit=2)
            token=a.reserve('AdamW',0)
            recovered=PilotBudget(path,study_sha256='a'*64,total_limit=3,per_candidate_limit=2)
            self.assertEqual(recovered.summary()['incomplete_reservations'],[token])
            second=recovered.reserve('AdamW',0);recovered.complete(second)
            with self.assertRaises(RuntimeError):recovered.reserve('AdamW',1)
            final=recovered.reserve('CAdamW',0);recovered.complete(final)
            with self.assertRaises(RuntimeError):recovered.reserve('CAdamW',1)
            with self.assertRaises(ValueError):recovered.complete(final)
            self.assertEqual(recovered.summary()['remaining_total'],0)
            with self.assertRaises(ValueError):PilotBudget(path,study_sha256='b'*64,total_limit=3,per_candidate_limit=2).summary()

    def test_oversized_budget_rejected(self):
        with self.assertRaises(ValueError):PilotBudget('unused.json',study_sha256='a'*64,total_limit=401)
        with self.assertRaises(ValueError):PilotBudget('unused.json',study_sha256='a'*64,per_candidate_limit=201)
