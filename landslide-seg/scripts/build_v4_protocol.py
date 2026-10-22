"""v4 协议冻结：事件级互斥划分 + 嵌套训练池（25/50/75/100%）。

一次生成，避免再多一个"包"：

    manifests/split_v4.json           # 冻结用的划分（含 sha256，可在确认后重算）
    manifests/allowed_train_pool.csv  # 逐图块，标注它属于哪一档嵌套池

划分规则（写死在 EVENTS 里，改动必须重建并更新哈希）：
- TRAIN   ：9 个事件，覆盖 4 大洲/多种地貌，最大化事件多样性
- DEV     ：hiroshima（历史 val，标 SEEN_DEV，保证与 v1–v3 可比）+ thrissur
- FINAL   ：newzealand（从未参与任何选择，可称全新留出）
- EXCLUDE ：nepal(8) / usa_alaska(15) 样本过少，指标不稳定，只报不选
- DIAGNOSTIC：italy（历史上被反复查看，只能作诊断）

嵌套池：在每个事件内按 `sha256(sample_id)` 排序后取前 ceil(k·n)，
保证 25% ⊂ 50% ⊂ 75% ⊂ 100%，且各档的事件权重一致。

用法::

    python scripts/build_v4_protocol.py --version s2_10band_v4
    python scripts/build_v4_protocol.py --version s2_10band_v4 --freeze
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

TRAIN = [
    "chimanimani",
    "dominicamaria",
    "indonesia",
    "kyrgyzstan1",
    "kyrgyzstan2",
    "hokkaido",
    "itogon",
    "china",
    "usa_puertorico",
]
DEV = ["hiroshima", "thrissur"]
FINAL = ["newzealand"]
EXCLUDE = ["nepal", "usa_alaska"]
DIAGNOSTIC = ["italy"]

FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def stable_key(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="s2_10band_v4")
    ap.add_argument("--freeze", action="store_true", help="写入 sha256 并标记 frozen")
    args = ap.parse_args()
    existing = ROOT / "manifests" / "split_v4.json"
    if existing.exists() and json.loads(existing.read_text(encoding="utf-8")).get("frozen"):
        raise FileExistsError("split_v4.json is frozen; use a new protocol revision (prepare_v4_revision.py)")

    man_path = ROOT / "manifests" / f"{args.version}.jsonl"
    rows = [json.loads(l) for l in man_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    by_event: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_event[r["event_id"]].append(r)

    role = {}
    for ev in TRAIN:
        role[ev] = "train"
    for ev in DEV:
        role[ev] = "dev"
    for ev in FINAL:
        role[ev] = "final"
    for ev in EXCLUDE:
        role[ev] = "exclude_small"
    for ev in DIAGNOSTIC:
        role[ev] = "diagnostic"

    missing = [ev for ev in by_event if ev not in role]
    if missing:
        raise SystemExit(f"以下事件未分配角色，先决定再冻结: {missing}")

    split = {
        "protocol": "event_ood_v4",
        "data_version": args.version,
        "design": {
            "train": TRAIN,
            "dev": DEV,
            "final": FINAL,
            "exclude_small": EXCLUDE,
            "diagnostic": DIAGNOSTIC,
        },
        "notes": {
            "hiroshima": "SEEN_DEV：v1–v3 一直用作 val 做模型选择，不构成盲测，仅用于与历史可比。",
            "italy": "DIAGNOSTIC：多轮查看过，不能作最终留出，也不参与任何筛选。",
            "newzealand": "FINAL：从未参与训练或选择；32.5% 图块用了时相 fallback，结论需单独声明。",
            "nepal/usa_alaska": "样本 <20，指标方差过大，只报告不参与选择。",
        },
        "events": {},
    }

    pool_rows: list[dict] = []
    for ev, items in sorted(by_event.items()):
        r = role[ev]
        split["events"][ev] = {"role": r, "n": len(items)}
        if r == "train":
            ordered = sorted(items, key=lambda x: stable_key(x["sample_id"]))
            n = len(ordered)
            for k in FRACTIONS:
                for item in ordered[: math.ceil(k * n)]:
                    item.setdefault("_pools", set()).add(int(k * 100))

    for ev in TRAIN:
        for item in sorted(by_event.get(ev, []), key=lambda x: stable_key(x["sample_id"])):
            pools = item.get("_pools", set())
            pool_rows.append(
                {
                    "sample_id": item["sample_id"],
                    "event_id": ev,
                    "mask_px": item.get("mask_px", 0),
                    "valid_ratio": item.get("valid_ratio"),
                    "temporal_fallback": item.get("temporal_fallback"),
                    "in_25": int(25 in pools),
                    "in_50": int(50 in pools),
                    "in_75": int(75 in pools),
                    "in_100": int(100 in pools),
                }
            )

    split["pool_summary"] = {
        f"{int(k * 100)}%": {
            "n": sum(1 for p in pool_rows if p[f"in_{int(k * 100)}"]),
            "events": len({p["event_id"] for p in pool_rows if p[f"in_{int(k * 100)}"]}),
            "fg_px": sum(p["mask_px"] for p in pool_rows if p[f"in_{int(k * 100)}"]),
        }
        for k in FRACTIONS
    }
    split["n_train_all"] = len(pool_rows)
    split["n_dev"] = sum(len(v) for e, v in by_event.items() if role[e] == "dev")
    split["n_final"] = sum(len(v) for e, v in by_event.items() if role[e] == "final")
    split["counts_by_role"] = dict(Counter(role.values()))

    out_split = ROOT / "manifests" / "split_v4.json"
    payload = json.dumps(split, ensure_ascii=False, indent=1, sort_keys=True)
    split["sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    split["frozen"] = bool(args.freeze)
    out_split.write_text(json.dumps(split, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")

    out_pool = ROOT / "manifests" / "allowed_train_pool.csv"
    with open(out_pool, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pool_rows[0].keys()))
        w.writeheader()
        w.writerows(pool_rows)

    print(f"[split] train 事件 {len(TRAIN)} | dev {DEV} | final {FINAL}")
    print(f"[size ] train={split['n_train_all']} dev={split['n_dev']} final={split['n_final']}")
    for k, v in split["pool_summary"].items():
        print(f"  {k:>4}: {v['n']:>5} tiles | {v['events']} events | fg {v['fg_px']:,} px")
    print(f"[out  ] {out_split} (frozen={split['frozen']})")
    print(f"[out  ] {out_pool}")


if __name__ == "__main__":
    main()
