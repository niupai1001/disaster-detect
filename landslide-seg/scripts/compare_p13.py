"""P1-3 两组的配对 bootstrap（best 与 last 两个检查点）。

同批瓦片、同一实现；按事件分层重采样。同时给出两个检查点，
因为上一轮（v4r4）已证明单检查点上的"区间不跨 0"可能是检查点效应。
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

FEW = "outputs/segformer/p13_events/segformer_b0_latedetail_fewevents_n1251_s1234/1234"
ALL = "outputs/segformer/p13_events/segformer_b0_latedetail_allevents_n1251_s1234/1234"
N_BOOT = 5000
SEED = 20261020


def load(p: Path) -> dict:
    out = {}
    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["sample_id"]] = {"event": r["event_id"], "tp": int(float(r["tp"])),
                                   "fp": int(float(r["fp"])), "fn": int(float(r["fn"]))}
    return out


def em(rows: dict, ids: list[str]) -> float:
    per: dict[str, list[int]] = {}
    for sid in ids:
        r = rows[sid]
        a = per.setdefault(r["event"], [0, 0, 0])
        a[0] += r["tp"]; a[1] += r["fp"]; a[2] += r["fn"]
    vals = [v[0] / (v[0] + v[1] + v[2]) for v in per.values() if (v[0] + v[1] + v[2])]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    out = {}
    for tag in ("best_val", "last_val"):
        few, all_ = load(ROOT / FEW / "eval" / f"per_tile_{tag}.csv"), load(ROOT / ALL / "eval" / f"per_tile_{tag}.csv")
        ids = sorted(few)
        assert sorted(all_) == ids
        per_event: dict[str, list[str]] = {}
        for sid in ids:
            per_event.setdefault(few[sid]["event"], []).append(sid)
        rng = random.Random(SEED)
        obs = em(all_, ids) - em(few, ids)
        ds = []
        for _ in range(N_BOOT):
            xs: list[str] = []
            for pool in per_event.values():
                xs.extend(rng.choices(pool, k=len(pool)))
            ds.append(em(all_, xs) - em(few, xs))
        ds.sort()
        lo, hi = ds[int(0.025 * N_BOOT)], ds[int(0.975 * N_BOOT) - 1]
        per_ev = {e: em(all_, v) - em(few, v) for e, v in per_event.items()}
        out[tag] = {"delta": obs, "ci": [lo, hi], "crosses_zero": bool(lo <= 0 <= hi),
                    "p_positive": sum(1 for d in ds if d > 0) / N_BOOT,
                    "per_event": per_ev,
                    "few": em(few, ids), "all": em(all_, ids), "n_tiles": len(ids)}

    L = ["# P1-3 配对比较：AllEvents − FewEvents（同 N=1251）\n",
         f"逐瓦片配对、按事件分层 bootstrap {N_BOOT} 次（seed {SEED}）。\n",
         "| 检查点 | FewEvents | AllEvents | Δ | 95% 区间 | 跨 0 | P(Δ>0) |",
         "|---|---:|---:|---:|---|---:|---:|"]
    for t, v in out.items():
        L.append(f"| {t} | {v['few']:.4f} | {v['all']:.4f} | {v['delta']:+.4f} | "
                 f"[{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] | {'是' if v['crosses_zero'] else '否'} | "
                 f"{v['p_positive']:.3f} |")
    L.append("\n## 逐事件差值\n")
    L.append("| 检查点 | " + " | ".join(sorted(out['best_val']['per_event'])) + " |")
    L.append("|---|" + "---:|" * len(out["best_val"]["per_event"]))
    for t, v in out.items():
        L.append(f"| {t} | " + " | ".join(f"{x:+.4f}" for x in v["per_event"].values()) + " |")
    L.append("\n> 两个检查点方向一致且都不跨 0，才可视为该对比下的稳定差异；"
             "单检查点的「不跨 0」不作为架构/数据结论的依据。\n")

    d = ROOT / "reports" / "p13"
    (d / "significance.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (d / "significance.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
