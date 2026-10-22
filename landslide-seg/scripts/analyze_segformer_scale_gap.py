"""尺度分层与 val→test 落差分解。

读 `ablation_eval.json`，对每组输出：
- val / test 的按档对象召回（small / medium / large）；
- test 的对象尺度构成；
- **反事实**：如果 test 的对象在每一档上都保持该组 val 的召回率，
  总体对象召回会是多少。用于区分"尺度构成"与"同档内性能下降"。

结果写到 `outputs/segformer/<version>/scale_gap.json`，并打印 markdown 表。

用法::

    python scripts/analyze_segformer_scale_gap.py
"""

from __future__ import annotations

import argparse
import json
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

BINS = ("small", "medium", "large")


def _bin_stats(overall: dict) -> dict:
    out = {}
    for name in BINS:
        b = (overall.get("objects") or {}).get("bins", {}).get(name) or {}
        out[name] = {
            "n_objects": b.get("n_objects", 0),
            "n_evaluable": b.get("n_objects_evaluable", 0),
            "n_matched_evaluable": b.get("n_matched_evaluable", 0),
            "object_recall_evaluable": b.get("object_recall_evaluable"),
            "pixel_recall": b.get("pixel_recall"),
        }
    out["total_evaluable"] = sum(out[n]["n_evaluable"] for n in BINS)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "segformer" / "base.yaml"))
    ap.add_argument("--input-name", default="ablation_eval.json")
    ap.add_argument("--output-suffix", default="")
    args = ap.parse_args()

    base = load_segformer_config(args.config)["segformer"]
    version = base["data"]["version"]
    root = resolve_path(base["output_dir"]) / version
    payload = json.loads((root / args.input_name).read_text(encoding="utf-8"))

    runs = payload["runs"]
    exps = payload["experiments"]
    seeds = payload["seeds"]

    result: dict = {"version": version, "experiments": {}, "counterfactual": {}}
    for exp in exps:
        result["experiments"][exp] = {}
        for split in ("val", "test"):
            sub = [r for r in runs if r["experiment"] == exp and r["split"] == split]
            if not sub:
                continue
            n = len(sub)
            agg = {}
            for name in BINS:
                tot_obj = sum(_bin_stats(r["overall"])[name]["n_objects"] for r in sub)
                tot_eval = sum(_bin_stats(r["overall"])[name]["n_evaluable"] for r in sub)
                tot_match = sum(
                    _bin_stats(r["overall"])[name]["n_matched_evaluable"] for r in sub
                )
                agg[name] = {
                    "n_objects": tot_obj,
                    "n_evaluable": tot_eval,
                    "n_matched_evaluable": tot_match,
                    "object_recall_evaluable": (tot_match / tot_eval) if tot_eval else None,
                }
            overall_eval = sum(agg[b]["n_evaluable"] for b in BINS)
            overall_match = sum(agg[b]["n_matched_evaluable"] for b in BINS)
            agg["overall"] = {
                "n_evaluable": overall_eval,
                "n_matched_evaluable": overall_match,
                "object_recall_evaluable": (overall_match / overall_eval) if overall_eval else None,
            }
            agg["n_runs"] = n
            result["experiments"][exp][split] = agg

    # 反事实：test 的对象构成 × 该组 val 的分档召回
    for exp in exps:
        val = result["experiments"].get(exp, {}).get("val")
        test = result["experiments"].get(exp, {}).get("test")
        if not val or not test:
            continue
        comp = {b: test[b]["n_evaluable"] for b in BINS}
        total = sum(comp.values())
        if total == 0:
            continue
        predicted = 0.0
        for b in BINS:
            rv = val[b]["object_recall_evaluable"]
            if rv is None:
                rv = 0.0
            predicted += comp[b] * rv
        result["counterfactual"][exp] = {
            "test_evaluable_objects": total,
            "test_bin_composition": comp,
            "predicted_overall_recall_under_val_rates": predicted / total,
            "actual_overall_recall": test["overall"]["object_recall_evaluable"],
        }

    out = root / f"scale_gap{args.output_suffix}.json"
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")

    lines = ["# 尺度分层与 val→test 分解", ""]
    lines += [
        "| 组 | split | small 召回 | medium 召回 | large 召回 | 总体对象召回 | 可评估对象 |",
        "|---|---|---|---|---|---|---|",
    ]
    for exp in exps:
        for split in ("val", "test"):
            agg = result["experiments"].get(exp, {}).get(split)
            if not agg:
                continue
            def fmt(b):
                v = agg[b]["object_recall_evaluable"]
                n = agg[b]["n_evaluable"]
                return "N/A" if v is None else f"{v:.3f} (n={n})"
            lines.append(
                f"| {exp} | {split} | {fmt('small')} | {fmt('medium')} | {fmt('large')} | "
                f"{agg['overall']['object_recall_evaluable']:.3f} | {agg['overall']['n_evaluable']} |"
            )
    lines += ["", "## 反事实：test 构成 × val 分档召回", ""]
    lines += [
        "| 组 | test 可评估对象 | 预测总体召回（用 val 分档率） | 实际总体召回 | 同档性能下降造成的差额 |",
        "|---|---:|---:|---:|---:|",
    ]
    for exp, c in result["counterfactual"].items():
        actual = c["actual_overall_recall"]
        pred = c["predicted_overall_recall_under_val_rates"]
        lines.append(
            f"| {exp} | {c['test_evaluable_objects']} | {pred:.3f} | "
            f"{(actual if actual is not None else float('nan')):.3f} | "
            f"{(actual - pred) if actual is not None else float('nan'):+.3f} |"
        )
    (root / f"scale_gap{args.output_suffix}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"[saved] {out}\n[saved] {root / f'scale_gap{args.output_suffix}.md'}")


if __name__ == "__main__":
    main()
