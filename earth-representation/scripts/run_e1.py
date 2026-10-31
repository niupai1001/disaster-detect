"""Phase 3 driver: run E1-0..E1-4 with one model, one dataset, one budget.

The comparison is only about the augmentation, so everything else is pinned:
same patches, same seed, same step count. Loss levels between different modes
are not directly comparable (the views differ), which is why the summary also
records the drop from the first to the last checkpoint of each run.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ssl import train_dino  # noqa: E402


def read_log(path):
    rows = []
    if Path(path).exists():
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "model_cnn.yaml"))
    ap.add_argument("--experiments", default=str(ROOT / "configs" / "experiments_e1.yaml"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "e1"))
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args(argv)

    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    grid = yaml.safe_load(Path(args.experiments).read_text(encoding="utf-8"))
    budget = grid.get("budget", {})
    steps = args.steps or budget.get("steps", 60)
    seed = budget.get("seed", 0)
    limit = args.limit or budget.get("patches_limit", 0)

    patches = train_dino.load_patches(ROOT / "data" / "interim" / "patches", limit=limit)
    print(f"patches: {len(patches)} | steps per run: {steps}", flush=True)
    if not patches:
        raise SystemExit("no patches materialised yet")

    summary = []
    for name, spec in grid["experiments"].items():
        if args.only and name not in args.only:
            continue
        cfg = json.loads(json.dumps(base))          # deep copy
        cfg["aug"].update(spec.get("aug", {}))
        cfg["train"]["steps"] = steps
        res = train_dino.run(cfg, patches, Path(args.out), seed=seed,
                             max_steps=steps, tag=name)
        rows = read_log(res["log"])
        losses = [r["loss"] for r in rows]
        first = losses[0] if losses else None
        tail = losses[-5:] if losses else []
        summary.append({"id": name, "label": spec.get("label"),
                        "aug": spec.get("aug"), "steps": res["steps"],
                        "first_loss": first,
                        "last_loss": losses[-1] if losses else None,
                        "tail_mean": round(sum(tail) / len(tail), 4) if tail else None,
                        "drop": round(first - losses[-1], 4) if len(losses) > 1 else None,
                        "checkpoint": res["checkpoint"]})

    out = Path(args.out) / "e1_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"experiments": summary, "steps": steps,
                               "seed": seed, "patches": len(patches)},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("summary:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
