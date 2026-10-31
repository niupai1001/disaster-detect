"""Seed sweep for the E1 arms: the plan requires >=3 seeds for core results."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "e1_seeds"

ARMS = [
    ("E1-0", "standard"),
    ("E1-1", "generic"),
    ("E1-2", "physical"),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--n-patches", type=int, default=2048)
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for seed in args.seeds:
        for arm, aug in ARMS:
            tag = f"{arm}_s{seed}"
            cmd = [sys.executable, "-u", "-X", "utf8",
                   "-m", "scripts.train_dino_official",
                   "--tag", tag, "--aug-mode", aug, "--steps", str(args.steps),
                   "--n-patches", str(args.n_patches), "--seed", str(seed),
                   "--out", str(OUT),
                   "--tv-root", str(OUT / "tvlp")]
            print("run", tag, flush=True)
            proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
            log = OUT / f"{tag}.jsonl"
            rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()
                    if l.strip()] if log.exists() else []
            losses = [r["loss"] for r in rows]
            summary.append({"tag": tag, "arm": arm, "seed": seed,
                            "last": losses[-1] if losses else None,
                            "returncode": proc.returncode})
            print(summary[-1], flush=True)

    (OUT / "seed_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
