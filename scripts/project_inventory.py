#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from project_ops.inventory import collect_inventory, format_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize project source, runtime, and generated file roots.")
    parser.add_argument("--root", default=".", help="Project root to scan.")
    parser.add_argument("--max-depth", type=int, help="Only count files whose relative path has at most this many parts.")
    args = parser.parse_args()

    report = collect_inventory(Path(args.root), max_depth=args.max_depth)
    print(format_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

