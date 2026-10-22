"""三架构 DEV 差异的配对 bootstrap（v4r4 §7 判据 2/3）。

读各 run 的 ``eval/per_tile_best_val.csv``（逐瓦片 tp/fp/fn），做按事件分层的
配对重采样，给出 Δ 的 95% 区间、逐事件同向数，并与"固定步数"口径交叉核对。

为什么必须做：0.003–0.008 这一级的差异在单种子 + 2 个 DEV 事件下无法凭均值判断。
"""

from __future__ import annotations

import csv
import json
import random
import statistics
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
PAIRS = [("Cross", "Self", "主比较：跨路径交换 vs 容量对照"),
         ("Cross", "LateDetail", "是否超过原主模型"),
         ("Self", "LateDetail", "多一层卷积本身的作用")]
N_BOOT = 5000
SEED = 20261020
FIXED = [500, 1200, 2400, 3600]


def read_rows(p: Path) -> dict:
    out = {}
    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["sample_id"]] = {"event": r["event_id"], "tp": int(float(r["tp"])),
                                   "fp": int(float(r["fp"])), "fn": int(float(r["fn"]))}
    return out


def iou(tp, fp, fn):
    d = tp + fp + fn
    return tp / d if d else float("nan")


def pooled(rows, ids):
    per, tp, fp, fn = {}, 0, 0, 0
    for sid in ids:
        r = rows[sid]
        tp += r["tp"]; fp += r["fp"]; fn += r["fn"]
        a = per.setdefault(r["event"], [0, 0, 0])
        a[0] += r["tp"]; a[1] += r["fp"]; a[2] += r["fn"]
    evs = {k: iou(*v) for k, v in per.items()}
    vals = [v for v in evs.values() if v == v]
    return {"overall": iou(tp, fp, fn), "events": evs,
            "event_mean": sum(vals) / len(vals) if vals else float("nan")}


def main() -> None:
    data, missing = {}, []
    for name, rel in ARMS.items():
        p = ROOT / rel / "eval" / "per_tile_best_val.csv"
        if not p.is_file():
            missing.append(str(p))
            continue
        data[name] = read_rows(p)
    if len(data) < 2:
        raise SystemExit("可用逐瓦片结果不足：" + ", ".join(missing))

    ref = sorted(next(iter(data.values())).keys())
    for k, v in data.items():
        if sorted(v.keys()) != ref:
            raise SystemExit(f"[fatal] {k} 的瓦片集合与参照不一致")

    single = {k: pooled(v, ref) for k, v in data.items()}
    by_event = {}
    for sid in ref:
        by_event.setdefault(data[next(iter(data))][sid]["event"], []).append(sid)
    for ev in by_event:
        by_event[ev].sort()

    rng = random.Random(SEED)
    comps = []
    for a, b, desc in PAIRS:
        if a not in data or b not in data:
            continue
        obs = single[a]["event_mean"] - single[b]["event_mean"]
        ds = []
        for _ in range(N_BOOT):
            ids = []
            for ev, pool in by_event.items():
                ids.extend(rng.choices(pool, k=len(pool)))
            ds.append(pooled(data[a], ids)["event_mean"] - pooled(data[b], ids)["event_mean"])
        ds.sort()
        signs = {ev: single[a]["events"].get(ev, float("nan")) - single[b]["events"].get(ev, float("nan")) for ev in by_event}
        comps.append({
            "a": a, "b": b, "desc": desc, "delta": obs,
            "ci": [ds[int(0.025 * N_BOOT)], ds[int(0.975 * N_BOOT) - 1]],
            "p_positive": sum(1 for d in ds if d > 0) / N_BOOT,
            "per_event": signs,
            "consistent": sum(1 for v in signs.values() if v > 0), "n_events": len(signs),
            "excludes_zero": not (ds[int(0.025 * N_BOOT)] <= 0 <= ds[int(0.975 * N_BOOT) - 1]),
        })

    # 固定步数口径
    fixed = {}
    for name, rel in ARMS.items():
        p = ROOT / rel / "train_summary.json"
        if not p.is_file():
            continue
        h = json.loads(p.read_text(encoding="utf-8"))["history"]
        m = {int(r["global_step"]): r["dev"]["event_mean_iou"] for r in h}
        fixed[name] = {s: m.get(s) for s in FIXED}
        fixed[name]["best"] = max(m.values())
        fixed[name]["best_step"] = max(m, key=m.get)

    out = {"n_tiles": len(ref), "n_boot": N_BOOT, "seed": SEED,
           "single": single, "pairwise": comps, "fixed_steps": fixed, "missing": missing}
    d = ROOT / "reports" / "v4r4_results"
    d.mkdir(parents=True, exist_ok=True)
    (d / "dev_significance.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    L = ["# v4r4 三架构 DEV 显著性\n",
         f"逐瓦片配对、按事件分层 bootstrap {N_BOOT} 次（seed {SEED}），{len(ref)} 个瓦片。\n",
         "## 1. 单点读数\n",
         "| 架构 | DEV 事件均值 | " + " | ".join(sorted(by_event)) + " | 合并 IoU |",
         "|---|---:|" + "---:|" * len(by_event) + "---:|"]
    for k, v in single.items():
        L.append(f"| {k} | {v['event_mean']:.4f} | " +
                 " | ".join(f"{v['events'].get(e, float('nan')):.4f}" for e in sorted(by_event)) +
                 f" | {v['overall']:.4f} |")
    L.append("\n## 2. 配对比较\n")
    L.append("| 比较 | Δ | 95% 区间 | 逐事件同向 | 区间跨 0 |")
    L.append("|---|---:|---|---:|---|")
    for c in comps:
        L.append(f"| {c['desc']} | {c['delta']:+.4f} | [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}] | "
                 f"{c['consistent']}/{c['n_events']} | {'是' if not c['excludes_zero'] else '否'} |")
    L.append("\n## 3. 逐事件差值\n")
    L.append("| 比较 | " + " | ".join(sorted(by_event)) + " |")
    L.append("|---|" + "---:|" * len(by_event))
    for c in comps:
        L.append(f"| {c['desc']} | " +
                 " | ".join(f"{c['per_event'].get(e, float('nan')):+.4f}" for e in sorted(by_event)) + " |")
    L.append("\n## 4. 固定步数 DEV\n")
    L.append("| 架构 | " + " | ".join(str(s) for s in FIXED) + " | best | best@步 |")
    L.append("|---|" + "---:|" * (len(FIXED) + 2))
    for k, v in fixed.items():
        cells = " | ".join(f"{v[s]:.4f}" if v.get(s) is not None else "-" for s in FIXED)
        L.append(f"| {k} | {cells} | {v['best']:.4f} | {v['best_step']} |")
    pair_path = d / "dev_significance.md"
    pair_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {pair_path}")


if __name__ == "__main__":
    main()
