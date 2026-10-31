"""E1/E2 experiment suites on the official DINO baseline.

E1 varies the augmentation; E2 varies the scale-consistency rule. Same model,
data, budget and seed within a suite, so only the named factor changes.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SUITES = {
    "e1": {
        "out": ROOT / "outputs" / "e1_official",
        "arms": [
            ("E1-0", "standard", "none", "标准 DINO 增强（裁剪/翻转）"),
            ("E1-1", "generic", "none", "普通模糊/缩放（无低通）"),
            ("E1-2", "physical", "none", "低通 + 物理采样"),
            ("E1-3", "physical_band", "none", "E1-2 + 波段子采样"),
            ("E1-4", "physical_band_radiometry", "none", "E1-3 + 有界辐射扰动"),
        ],
    },
    "e2": {
        "out": ROOT / "outputs" / "e2_official",
        "arms": [
            ("E2-0", "standard", "none", "不加尺度一致性"),
            ("E2-1", "standard", "uniform", "全区域统一权重"),
            ("E2-2", "standard", "fixed", "仅共同尺度 r=2"),
            ("E2-3", "standard", "random", "随机区域，匹配约束比例"),
            ("E2-4", "standard", "selective", "物理退化信息引导的选择性"),
        ],
    },
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=list(SUITES), default="e2")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n-patches", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--scale-lambda", type=float, default=0.1)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args(argv)

    cfg = SUITES[args.suite]
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)

    summary = []
    for name, aug, scale, label in cfg["arms"]:
        if args.only and name not in args.only:
            continue
        cmd = [sys.executable, "-u", "-X", "utf8",
               "-m", "scripts.train_dino_official",
               "--tag", name, "--aug-mode", aug, "--scale-mode", scale,
               "--scale-lambda", str(args.scale_lambda),
               "--steps", str(args.steps), "--n-patches", str(args.n_patches),
               "--batch-size", str(args.batch_size), "--out", str(out)]
        print("run", name, aug, scale, flush=True)
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        log = out / f"{name}.jsonl"
        rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()
                if l.strip()] if log.exists() else []
        losses = [r["loss"] for r in rows]
        rec = {"id": name, "aug": aug, "scale": scale, "label": label,
               "points": len(losses),
               "last": losses[-1] if losses else None,
               "tail_mean": round(sum(losses[-10:]) / len(losses[-10:]), 4) if losses else None,
               "returncode": proc.returncode}
        summary.append(rec)
        print(rec, flush=True)

    (out / f"{args.suite}_summary.json").write_text(
        json.dumps({"suite": args.suite, "steps": args.steps,
                    "experiments": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
