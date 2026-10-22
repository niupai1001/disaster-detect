"""对比两个数据版本的 3-seed 结果（v1 vs v2_clean），逐 seed 配对。

读两份 ``ablation_eval_metric_v2.json``，输出 val/test 的均值±标准差与逐 seed 差值。

用法::

    python scripts/compare_data_versions.py \
        --a outputs/segformer/s2_10band_v1/ablation_eval_metric_v2.json \
        --b outputs/segformer/s2_10band_v2_clean/ablation_eval_metric_v2.json \
        --out outputs/segformer/compare_v1_v2_clean
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:  # Windows 控制台/管道默认 cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from landslide_seg.config import resolve_path  # noqa: E402


def _load_run_map(payload: dict) -> dict[tuple[str, str, int], dict]:
    return {
        (r["experiment"], r["split"], int(r["seed"])): r["overall"]
        for r in payload["runs"]
    }


def _get(overall: dict, path: str):
    node = overall
    for part in path.split("."):
        node = node.get(part) if isinstance(node, dict) else None
        if node is None:
            return None
    return node


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--label-a", default="v1")
    ap.add_argument("--label-b", default="v2_clean")
    args = ap.parse_args()

    pa = json.loads(resolve_path(args.a).read_text(encoding="utf-8"))
    pb = json.loads(resolve_path(args.b).read_text(encoding="utf-8"))
    ra, rb = _load_run_map(pa), _load_run_map(pb)

    keys = [
        ("pixel", "pixel.iou"),
        ("pixel", "pixel.dice"),
        ("objects", "objects.object_recall_evaluable"),
        ("tiny", "objects.tiny.object_recall_evaluable"),
        ("tiny", "objects.tiny.pixel_recall"),
        ("boundary", "boundary.f1"),
    ]
    rows = []
    for exp in pa["experiments"]:
        for split in ("val", "test"):
            seeds = sorted(seed for (e, sp, seed) in ra if e == exp and sp == split)
            seeds_b = sorted(seed for (e, sp, seed) in rb if e == exp and sp == split)
            common = [s for s in seeds if s in seeds_b]
            if not common:
                continue
            entry = {"experiment": exp, "split": split, "seeds": common, "metrics": {}}
            for _name, path in keys:
                va = [_get(ra[(exp, split, s)], path) for s in common]
                vb = [_get(rb[(exp, split, s)], path) for s in common]
                pairs = [(a, b) for a, b in zip(va, vb) if a is not None and b is not None]
                if not pairs:
                    entry["metrics"][path] = {"n": 0}
                    continue
                a_vals = [p[0] for p in pairs]
                b_vals = [p[1] for p in pairs]
                deltas = [b - a for a, b in pairs]
                entry["metrics"][path] = {
                    "n": len(pairs),
                    "a_mean": round(float(stats.mean(a_vals)), 5),
                    "b_mean": round(float(stats.mean(b_vals)), 5),
                    "delta_mean": round(float(stats.mean(deltas)), 5),
                    "delta_per_seed": [round(float(d), 5) for d in deltas],
                }
            rows.append(entry)

    out = {
        "a": {"path": args.a, "label": args.label_a, "metric_version": pa.get("metric_version")},
        "b": {"path": args.b, "label": args.label_b, "metric_version": pb.get("metric_version")},
        "rows": rows,
    }

    lines = [f"# {args.label_a} vs {args.label_b}（metric v2，逐 seed 配对）", ""]
    for split in ("val", "test"):
        lines += [f"## {split}", ""]
        lines += [
            f"| 组 | IoU {args.label_a} | IoU {args.label_b} | ΔIoU | 对象召回 {args.label_a} | 对象召回 {args.label_b} | Δ对象召回 |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            if r["split"] != split:
                continue
            iou = r["metrics"].get("pixel.iou", {})
            obj = r["metrics"].get("objects.object_recall_evaluable", {})
            if not iou.get("n"):
                continue
            lines.append(
                f"| {r['experiment']} | {iou['a_mean']:.4f} | {iou['b_mean']:.4f} | "
                f"{iou['delta_mean']:+.4f} | {obj['a_mean']:.4f} | {obj['b_mean']:.4f} | "
                f"{obj['delta_mean']:+.4f} |"
            )
        lines.append("")

    out_base = (
        resolve_path(args.out)
        if args.out
        else resolve_path("outputs/segformer/compare_data_versions")
    )
    out_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_base.with_suffix(".json")
    md_path = out_base.with_suffix(".md")
    json_path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"[saved] {json_path}\n[saved] {md_path}")


if __name__ == "__main__":
    main()
