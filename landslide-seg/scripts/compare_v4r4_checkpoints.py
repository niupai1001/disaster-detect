"""同一批瓦片、同一个检验，在 best 与 last 两个检查点上分别做配对 bootstrap。

存在理由：v4r4 主报告 §1.5 的"Self 显著更差"只在 best 口径成立，在 last 口径下反向。
这个脚本就是那个反证的复现入口——它检验的不是架构，而是"换个检查点结论会不会翻"。

用法::

    python scripts/compare_v4r4_checkpoints.py
    python scripts/compare_v4r4_checkpoints.py --tags best_val,last_val --n-boot 5000
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ARMS = {
    "LateDetail": "outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_detailres_n100_u3600_s1234/1234",
    "Self": "outputs/segformer/s2_10band_v4r4/segformer_b0_detail_early_self_residual_n100_u3600_s1234/1234",
    "Cross": "outputs/segformer/s2_10band_v4r4/segformer_b0_detail_early_cross_exchange_n100_u3600_s1234/1234",
}
PAIRS = [("Cross", "Self"), ("Cross", "LateDetail"), ("Self", "LateDetail")]


def load(p: Path) -> dict:
    out = {}
    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["sample_id"]] = {
                "event": r["event_id"],
                "tp": int(float(r["tp"])),
                "fp": int(float(r["fp"])),
                "fn": int(float(r["fn"])),
            }
    return out


def event_mean(rows: dict, ids: list[str]) -> float:
    per: dict[str, list[int]] = {}
    for sid in ids:
        r = rows[sid]
        a = per.setdefault(r["event"], [0, 0, 0])
        a[0] += r["tp"]; a[1] += r["fp"]; a[2] += r["fn"]
    vals = []
    for a in per.values():
        d = a[0] + a[1] + a[2]
        if d:
            vals.append(a[0] / d)
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="best_val,last_val")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261020)
    args = ap.parse_args()

    table: dict[str, dict] = {}
    for tag in [t.strip() for t in args.tags.split(",") if t.strip()]:
        data = {}
        for k, rel in ARMS.items():
            p = ROOT / rel / "eval" / f"per_tile_{tag}.csv"
            if not p.is_file():
                print(f"[skip] {k} 缺少 {p.name}")
                continue
            data[k] = load(p)
        if len(data) < 2:
            continue
        ids = sorted(next(iter(data.values())).keys())
        per_event: dict[str, list[str]] = {}
        for sid in ids:
            per_event.setdefault(data[next(iter(data))][sid]["event"], []).append(sid)
        rng = random.Random(args.seed)
        table[tag] = {"n_tiles": len(ids), "pairs": {}}
        for a, b in PAIRS:
            if a not in data or b not in data:
                continue
            obs = event_mean(data[a], ids) - event_mean(data[b], ids)
            diffs = []
            for _ in range(args.n_boot):
                xs: list[str] = []
                for pool in per_event.values():
                    xs.extend(rng.choices(pool, k=len(pool)))
                diffs.append(event_mean(data[a], xs) - event_mean(data[b], xs))
            diffs.sort()
            lo, hi = diffs[int(0.025 * args.n_boot)], diffs[int(0.975 * args.n_boot) - 1]
            table[tag]["pairs"][f"{a}-{b}"] = {
                "delta": obs, "ci": [lo, hi], "crosses_zero": bool(lo <= 0 <= hi),
            }

    # ---- 输出 ----
    lines = ["# best vs last：同一检验在两个检查点上的结论\n",
             f"配对 bootstrap {args.n_boot} 次，seed {args.seed}；同一批瓦片、同一实现。\n"]
    tags = list(table)
    hdr = "| 比较 | " + " | ".join(f"{t}" for t in tags) + " | 方向是否稳定 |"
    lines.append(hdr)
    lines.append("|---|" + "---|" * len(tags) + "---|")
    keys = sorted({k for t in tags for k in table[t]["pairs"]})
    stable = {}
    for k in keys:
        cells, signs = [], []
        for t in tags:
            v = table[t]["pairs"].get(k)
            if not v:
                cells.append("-")
                continue
            cells.append(f"{v['delta']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]"
                         + ("（跨0）" if v["crosses_zero"] else ""))
            signs.append(v["delta"] > 0)
        same = len(set(signs)) == 1 if signs else False
        stable[k] = same
        lines.append(f"| {k} | " + " | ".join(cells) + f" | {'是' if same else '**否，符号相反**'} |")

    flips = [k for k, v in stable.items() if not v]
    lines.append("\n## 判读\n")
    if flips:
        lines.append(f"- 符号翻转的比较：**{', '.join(flips)}**。这些差异不能归因于架构，只能归因于检查点选择。")
    stable_keys = [k for k, v in stable.items() if v]
    if stable_keys:
        lines.append(f"- 方向稳定的比较：{', '.join(stable_keys)}。只有这些可以作为架构层面的弱证据。")
    lines.append("- 任何一项「区间不跨 0」都应理解为「在该检查点的图块抽样下方向稳定」，不是「架构差异已确立」。")

    out = ROOT / "reports" / "v4r4_results" / "checkpoint_flip.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out.with_suffix(".json")).write_text(
        json.dumps(table, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
