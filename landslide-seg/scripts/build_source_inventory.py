"""P0：把某个数据版本的清单整理成可对账的资产表。

产出::

    manifests/source_inventory.csv      # 逐图块，含来源、时相、质量、标签、split
    reports/P0_inventory_summary.md     # 逐事件汇总 + 剔除原因

不读影像，只读 ``manifests/<version>.jsonl`` 与 ``<version>_quarantine.jsonl``，
因此对 6,000+ 图块也是秒级。

用法::

    python scripts/build_source_inventory.py --version s2_10band_v4
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

COLS = [
    "sample_id",
    "event_id",
    "split",
    "label_status",
    "mask_px",
    "mask_coverage",
    "valid_ratio",
    "cloud_fraction",
    "date",
    "event_date",
    "days_after_event",
    "temporal_fallback",
    "temporal_risk",
    "days_delta_vs_post",
    "crs",
    "source_uri",
    "cache_sha256",
]


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def classify_quarantine(rec: dict) -> str:
    cat = rec.get("category")
    if cat:
        return cat
    reasons = " ".join(rec.get("reasons", []))
    if "不可观测" in reasons:
        return "label_unobservable"
    if "valid ratio" in reasons:
        return "valid_ratio_low"
    if "unannotated" in reasons:
        return "unannotated"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="s2_10band_v4")
    ap.add_argument("--out-dir", default=str(ROOT))
    ap.add_argument(
        "--extra-quarantine",
        default="",
        help="额外合并的隔离清单（逗号分隔），用于把未标注统计并进同一张表",
    )
    args = ap.parse_args()

    base = Path(args.out_dir)
    rows = load_jsonl(base / "manifests" / f"{args.version}.jsonl")
    quar = load_jsonl(base / "manifests" / f"{args.version}_quarantine.jsonl")
    for extra in [x for x in args.extra_quarantine.split(",") if x.strip()]:
        quar += load_jsonl(base / "manifests" / extra.strip())
    if not rows:
        raise SystemExit(f"空清单: manifests/{args.version}.jsonl")

    # 1. 逐图块 CSV
    out_csv = base / "manifests" / "source_inventory.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in COLS})

    # 2. 逐事件汇总
    per_event: dict[str, dict] = defaultdict(lambda: {"pos": 0, "px": 0, "split": Counter(), "fb": 0})
    for r in rows:
        e = per_event[r.get("event_id", "?")]
        e["pos"] += 1
        e["px"] += int(r.get("mask_px") or 0)
        e["split"][r.get("split", "?")] += 1
        e["fb"] += 1 if r.get("temporal_fallback") else 0

    quar_event = defaultdict(Counter)
    for r in quar:
        ev = r.get("event_id") or r.get("event") or "?"
        quar_event[ev][classify_quarantine(r)] += 1

    all_events = sorted(set(list(per_event) + list(quar_event)))
    lines = [
        f"# P0 资产汇总：`{args.version}`",
        "",
        f"- 认证图块 **{len(rows):,}**；隔离 **{len(quar):,}**；逐块明细见 `manifests/source_inventory.csv`。",
        "- `可用`=通过准入且通过灾后时间契约的图块；`未标注`=官方 `annotated=False`，本轮不当负样本；",
        "  `标签不可观测`=GT 像素大部分落在 SCL 无效区，监督不可核实。",
        "",
        "| 事件 | 可用 | 其中 train/val/test | 前景像素 | 时相 fallback | 未标注 | 标签不可观测 | 其它剔除 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    tot = Counter()
    for ev in all_events:
        e = per_event.get(ev, {"pos": 0, "px": 0, "split": Counter(), "fb": 0})
        q = quar_event.get(ev, Counter())
        n_un = q.get("unannotated", 0)
        n_lab = q.get("label_unobservable", 0) + q.get("valid_ratio_low", 0)
        n_oth = sum(q.values()) - n_un - n_lab
        split_str = " / ".join(f"{k} {v}" for k, v in sorted(e["split"].items())) or "-"
        lines.append(
            f"| {ev} | {e['pos']:,} | {split_str} | {e['px']:,} | {e['fb']} | {n_un:,} | {n_lab:,} | {n_oth:,} |"
        )
        tot["pos"] += e["pos"]
        tot["px"] += e["px"]
        tot["fb"] += e["fb"]
        tot["un"] += n_un
        tot["lab"] += n_lab
        tot["oth"] += n_oth
    lines.append(
        f"| **合计** | **{tot['pos']:,}** | | **{tot['px']:,}** | **{tot['fb']}** | **{tot['un']:,}** | **{tot['lab']:,}** | **{tot['oth']:,}** |"
    )

    fb_by_ev = sorted(((ev, per_event[ev]["fb"]) for ev in per_event if per_event[ev]["fb"]), key=lambda x: -x[1])
    lines += [
        "",
        "## 时相 fallback 事件排行",
        "",
        "`fallback` = 灾后 120 天窗口内无可用时相，退化为全灾后最低云量。数量大的事件在跨事件比较时要单独声明。",
        "",
        "| 事件 | fallback 图块 | 占该事件可用图块 |",
        "|---|---:|---:|",
    ]
    for ev, n in fb_by_ev:
        lines.append(f"| {ev} | {n} | {n / per_event[ev]['pos']:.1%} |")

    lines += ["", "## 复现", "", "```bash", f"python scripts/build_source_inventory.py --version {args.version}", "```", ""]
    out_md = base / "reports" / "P0_inventory_summary.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] {len(rows):,} 行 → {out_csv}")
    print(f"[done] 事件汇总 → {out_md}")
    print(f"[info] 事件数 {len(all_events)}；train 事件数 {sum(1 for e in per_event if per_event[e]['split'].get('train'))}")


if __name__ == "__main__":
    main()
