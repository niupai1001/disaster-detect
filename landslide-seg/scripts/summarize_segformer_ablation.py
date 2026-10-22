"""汇总四组 × 三 seed：对每个 best.pt 在 val/test 上评测，再按组聚合均值±标准差。

评测本身调用 ``scripts/evaluate_segformer.py``（独立进程，确保与单次评测完全同一口径），
本脚本只负责批量调度与聚合，输出：
- ``outputs/segformer/<version>/ablation_eval.json``：逐 run + 逐组聚合；
- ``outputs/segformer/<version>/ablation_table.md``：可直接贴进报告的表。

用法::

    python scripts/summarize_segformer_ablation.py
    python scripts/summarize_segformer_ablation.py --splits val test --seeds 1234 2026 3407
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:  # Windows 控制台/管道默认 cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from landslide_seg.config import load_segformer_config, resolve_path  # noqa: E402

METRICS = [
    ("pixel", "iou"),
    ("pixel", "dice"),
    ("pixel", "precision"),
    ("pixel", "recall"),
    ("objects", "object_recall_evaluable"),
    ("objects", "object_recall_strict"),
    ("boundary", "f1"),
]


def _flat(d: dict, group: str, key: str):
    return (d.get(group) or {}).get(key)


def _mean_std(values: list[float]) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "values": []}
    return {
        "n": len(vals),
        "mean": round(float(stats.mean(vals)), 5),
        "std": round(float(stats.pstdev(vals)), 5) if len(vals) > 1 else 0.0,
        "values": [round(float(v), 5) for v in vals],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "segformer" / "base.yaml"))
    ap.add_argument("--experiments", nargs="+", default=["A0", "A1", "B0", "B1"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1234, 2026, 3407])
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    base = load_segformer_config(args.config)["segformer"]
    version = base["data"]["version"]
    root = resolve_path(base["output_dir"]) / version
    evaluator = ROOT / "scripts" / "evaluate_segformer.py"

    rows: list[dict] = []
    for exp in args.experiments:
        for seed in args.seeds:
            run_dir = root / exp / str(seed)
            best = run_dir / "best.pt"
            if not best.is_file():
                print(f"[skip] 缺少 {best}")
                continue
            for split in args.splits:
                out = run_dir / "eval" / f"eval_{split}.json"
                if args.skip_existing and out.is_file():
                    print(f"[cached] {exp} {seed} {split}")
                else:
                    cmd = [
                        sys.executable, str(evaluator),
                        "--checkpoint", str(best),
                        "--split", split,
                    ]
                    print(f"[eval] {exp} seed={seed} split={split}", flush=True)
                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    if proc.returncode != 0:
                        print(proc.stdout[-2000:])
                        print(proc.stderr[-2000:])
                        raise RuntimeError(f"评测失败: {exp} {seed} {split}")
                payload = json.loads(out.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "experiment": exp,
                        "seed": int(seed),
                        "split": split,
                        "checkpoint": str(best).replace("\\", "/"),
                        "report": payload,
                    }
                )

    agg: dict[str, dict] = {}
    for exp in args.experiments:
        agg[exp] = {}
        for split in args.splits:
            sub = [r for r in rows if r["experiment"] == exp and r["split"] == split]
            entry: dict = {"n_runs": len(sub)}
            for group, key in METRICS:
                entry[f"{group}.{key}"] = _mean_std(
                    [_flat(r["report"]["overall"], group, key) for r in sub]
                )
            entry["objects.n_gt"] = _mean_std(
                [_flat(r["report"]["overall"], "objects", "n_gt") for r in sub]
            )
            entry["pixels.tp"] = _mean_std(
                [_flat(r["report"]["overall"], "pixel", "tp") for r in sub]
            )
            entry["pixels.fp"] = _mean_std(
                [_flat(r["report"]["overall"], "pixel", "fp") for r in sub]
            )
            entry["pixels.fn"] = _mean_std(
                [_flat(r["report"]["overall"], "pixel", "fn") for r in sub]
            )
            # 分档对象召回
            bins = {}
            for name in ("small", "medium", "large"):
                bins[name] = _mean_std(
                    [
                        ((r["report"]["overall"].get("objects") or {}).get("bins") or {})
                        .get(name, {})
                        .get("object_recall_evaluable")
                        for r in sub
                    ]
                )
            entry["bins_object_recall_evaluable"] = bins
            entry["negative_tiles"] = _mean_std(
                [
                    ((r["report"]["overall"].get("negative_tiles") or {}).get("n"))
                    for r in sub
                ]
            )
            agg[exp][split] = entry

    out = {
        "version": version,
        "experiments": args.experiments,
        "seeds": args.seeds,
        "splits": args.splits,
        "aggregate": agg,
        "runs": [
            {
                "experiment": r["experiment"],
                "seed": r["seed"],
                "split": r["split"],
                "checkpoint": r["checkpoint"],
                "overall": r["report"]["overall"],
                "by_event": r["report"].get("by_event"),
            }
            for r in rows
        ],
    }
    out_path = root / "ablation_eval.json"
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    # ---- Markdown 表 ----
    lines = ["# 四组结果（均值 ± 标准差，括号内为逐 seed 值）", ""]
    for split in args.splits:
        lines += [f"## {split}", ""]
        header = "| 组 | pixel IoU | Dice | P | R | 对象召回(未截断) | 边界 F1 |"
        lines += [header, "|---|---|---|---|---|---|---|"]
        for exp in args.experiments:
            e = agg[exp][split]
            def cell(k):
                d = e[k]
                if d["n"] == 0:
                    return "N/A"
                vals = ", ".join(f"{v:.3f}" for v in d["values"])
                return f"{d['mean']:.3f} ± {d['std']:.3f} ({vals})"
            lines.append(
                f"| {exp} | {cell('pixel.iou')} | {cell('pixel.dice')} | {cell('pixel.precision')} | "
                f"{cell('pixel.recall')} | {cell('objects.object_recall_evaluable')} | {cell('boundary.f1')} |"
            )
        lines.append("")
    (root / "ablation_table.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"[saved] {out_path}\n[saved] {root / 'ablation_table.md'}")


if __name__ == "__main__":
    main()
