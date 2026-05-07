from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from project_ops.inventory import collect_inventory, format_markdown


class ProjectInventoryTests(unittest.TestCase):
    def test_collect_inventory_classifies_review_and_runtime_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root / "src" / "pkg" / "module.py", "print('source')\n")
            self._write(root / "configs" / "run.yaml", "experiment_id: test\n")
            self._write(root / "docs" / "note.md", "# Note\n")
            self._write(root / "tests" / "test_module.py", "def test_x(): pass\n")
            self._write(root / ".agent-team" / "tasks" / "task-a.json", "{}\n")
            self._write(root / ".agent-team" / "artifacts" / "task-a" / "run" / "metrics.json", "{}\n")
            self._write(root / "database" / "sample.tif", "data\n")
            self._write(root / "workspace" / "runs" / "run-a" / "metrics.json", "{}\n")
            self._write(root / "misc" / "scratch.tmp", "tmp\n")
            self._write(root / ".git" / "objects" / "ignored.json", "{}\n")

            report = collect_inventory(root)

        categories = {item["path"]: item["category"] for item in report["top_level"]}
        self.assertEqual(categories["src"], "source")
        self.assertEqual(categories["configs"], "configuration")
        self.assertEqual(categories["docs"], "documentation")
        self.assertEqual(categories["tests"], "tests")
        self.assertEqual(categories[".agent-team"], "runtime_ledger")
        self.assertEqual(categories["database"], "local_data")
        self.assertEqual(categories["workspace"], "generated_workspace")
        self.assertEqual(categories["misc"], "other")
        self.assertNotIn(".git", categories)
        self.assertEqual(report["extensions"][".json"], 3)
        self.assertEqual(report["extensions"][".tif"], 1)
        self.assertGreater(report["total_files"], 0)

    def test_format_markdown_exposes_reviewable_and_generated_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root / "src" / "pkg" / "module.py", "print('source')\n")
            self._write(root / "workspace" / "runs" / "run-a" / "metrics.json", "{}\n")
            report = collect_inventory(root)

        markdown = format_markdown(report)

        self.assertIn("# Project Inventory", markdown)
        self.assertIn("## Top-Level Roots", markdown)
        self.assertIn("source", markdown)
        self.assertIn("generated_workspace", markdown)
        self.assertIn("## File Extensions", markdown)

    @staticmethod
    def _write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()

