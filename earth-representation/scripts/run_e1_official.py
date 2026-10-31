"""E1 on the official DINO baseline: does physical augmentation help?

Same model, data, budget and seed; only the augmentation differs.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "e1_official"

EXPERIMENTS = [
    ("E1-0", "standard", "标准 DINO 增强（裁剪/翻转）"),
    ("E1-1", "generic", "普通模糊/缩放（无低通）"),
    ("E1-2", "physical", "低通 + 物理采样"),
    ("E1-3", "physical_band", "E1-2 + 波段子采样"),
    ("E1-4", "physical_band_radiometry", "E1-3 + 有界辐射扰动"),
]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--n-patches", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    global OUT
    if args.out:
        OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, mode, label in EXPERIMENTS:
        if args.only and name not in args.only:
            continue
        cmd = [sys.executable, "-u", "-X", "utf8",
               "-m", "scripts.train_dino_official",
               "--tag", name, "--aug-mode", mode,
               "--steps", str(args.steps),
               "--n-patches", str(args.n_patches),
               "--batch-size", str(args.batch_size),
               "--out", str(OUT)]
        print("run", name, mode, flush=True)
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        log = OUT / f"{name}.jsonl"
        rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()
                if l.strip()] if log.exists() else []
        losses = [r["loss"] for r in rows]
        summary.append({
            "id": name, "mode": mode, "label": label,
            "steps": len(losses),
            "first": losses[0] if losses else None,
            "last": losses[-1] if losses else None,
            "tail_mean": round(sum(losses[-10:]) / len(losses[-10:]), 4) if losses else None,
            "min": min(losses) if losses else None,
            "returncode": proc.returncode,
        })
        print(name, summary[-1], flush=True)

    (OUT / "e1_summary.json").write_text(
        json.dumps({"experiments": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
