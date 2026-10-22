"""P0：v4 数据契约审计（读真实 nc，不只看清单字段）。

检查项
------
1. 波段集合与顺序：必须是 B02..B12 + B8A 共 10 个，且与 ``S2_10_BANDS`` 一致；
2. 数据类型与量级：int16、无异常负值、DN 与 harmonized 量级一致；
3. 时相轴：15 个时相，``time`` 单位可解析，``date = time_origin + time[idx]``；
4. SCL：取值集合与允许清单；``valid_ratio`` 可由 SCL 独立复算并吻合；
5. MASK：取值 ⊆ {0,1,255}；``mask_px`` 可独立复算并吻合；
6. 时相规则：``date >= event_date``、``days_after_event >= 0``；
7. 缓存一致性：``cache_sha256`` 与 npz 实际哈希一致；
8. 泄漏：事件不跨 split；bbox 跨 split 重叠为 0。

输出 ``reports/P0_data_contract_audit.md``，全部 PASS 才允许启动 M2。

用法::

    python scripts/audit_v4_contract.py --version s2_10band_v4 --sample 40
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import h5py  # noqa: E402
import numpy as np  # noqa: E402

EXPECTED_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
SCL_VALID = (4, 5, 6, 7)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _event_dates(raw) -> list[date]:
    """多事件图块的 event_date 是 ``'2018-12-16,2017-07-24,None'`` 形式。

    ``prepare_segformer_data.py`` 取 ``ed[:10]``（第一个日期）做时相基准，
    多事件的安全裁决交给后续 p20 步骤；这里按同样口径复算，避免审计标准与生成标准不一致。
    """
    if isinstance(raw, (list, tuple)):
        cands = [str(x) for x in raw]
    else:
        cands = str(raw).split(",")
    parsed = []
    for c in cands:
        c = c.strip()
        try:
            parsed.append(date.fromisoformat(c))
        except ValueError:
            continue
    return parsed


def audit_tile(rec: dict) -> dict:
    """返回单图块的检查明细；异常不抛出，记进 errors。"""
    out = {"sample_id": rec["sample_id"], "event": rec["event_id"], "checks": {}, "errors": []}
    nc = Path(rec["source_uri"])
    if not nc.is_file():
        out["errors"].append(f"nc 不存在: {nc}")
        return out
    try:
        with h5py.File(nc, "r") as f:
            names = [b for b in EXPECTED_BANDS if b in f]
            extra = [k for k in f.keys() if k not in EXPECTED_BANDS]
            out["checks"]["bands"] = names == EXPECTED_BANDS
            out["checks"]["has_scl"] = "SCL" in f
            out["checks"]["has_mask"] = "MASK" in f
            out["checks"]["has_dem_unused"] = "DEM" in f
            out["extra_vars"] = sorted(extra)

            idx = int(rec["time_index"])
            arr = {}
            for b in EXPECTED_BANDS:
                arr[b] = f[b][idx]
            dtypes = {str(f[b].dtype) for b in EXPECTED_BANDS}
            out["checks"]["dtype_int16"] = dtypes == {"int16"}
            mins = {b: int(arr[b].min()) for b in EXPECTED_BANDS}
            maxs = {b: int(arr[b].max()) for b in EXPECTED_BANDS}
            out["dn_range"] = {"min": min(mins.values()), "max": max(maxs.values())}
            # harmonized 10m S2 L2A 的 DN 落在 0–20000 量级；出现大面积负值说明读错层或未偏移
            out["checks"]["dn_plausible"] = out["dn_range"]["min"] >= 0 and out["dn_range"]["max"] <= 30000

            scl = f["SCL"][idx]
            scl_vals = set(np.unique(scl).tolist())
            out["scl_values"] = sorted(scl_vals)
            out["checks"]["scl_known"] = scl_vals <= set(range(0, 12)) | {255}
            valid_ratio = float(np.isin(scl, SCL_VALID).mean())
            out["valid_ratio_recomputed"] = round(valid_ratio, 4)
            out["valid_ratio_manifest"] = rec.get("valid_ratio")
            out["checks"]["valid_ratio_match"] = (
                rec.get("valid_ratio") is not None
                and abs(valid_ratio - float(rec["valid_ratio"])) <= 0.02
            )

            m = f["MASK"][idx]
            mask_vals = set(np.unique(m).tolist())
            out["mask_values"] = sorted(mask_vals)
            out["checks"]["mask_binary"] = mask_vals <= {0, 1, 255}
            mask_px = int((m == 1).sum())
            out["mask_px_recomputed"] = mask_px
            out["mask_px_manifest"] = rec.get("mask_px")
            out["checks"]["mask_px_match"] = mask_px == int(rec.get("mask_px", -1))

            # 时相：date 必须等于 time_origin + time[idx]
            t = f["time"]
            units = t.attrs.get("units", "")
            origin = None
            m2 = re.search(r"since\s+(\d{4}-\d{2}-\d{2})", str(units))
            if m2:
                origin = date.fromisoformat(m2.group(1))
            if origin is not None:
                d = origin + timedelta(days=int(t[idx]))
                out["date_recomputed"] = d.isoformat()
                out["checks"]["date_match"] = d.isoformat() == rec.get("date")

            n_time = f["B02"].shape[0]
            out["checks"]["n_timesteps_15"] = n_time == 15
            out["checks"]["time_index_in_range"] = 0 <= idx < n_time

            # MASK 在 15 个时相上是否一致（Sen12 的标注是单一事件掩膜）
            mask_cube = f["MASK"][:]
            same = bool((mask_cube == mask_cube[0]).all())
            out["checks"]["mask_static_over_time"] = same
    except Exception as exc:  # pragma: no cover
        out["errors"].append(f"{type(exc).__name__}: {exc}")
        return out

    dates = _event_dates(rec["event_date"])
    chosen = date.fromisoformat(rec["date"])
    if not dates:
        out["errors"].append(f"event_date 不可解析: {rec['event_date']!r}")
        return out
    # p20 已把多事件图块的基准统一到"最晚事件日期"，因此这里按 max(dates) 复算
    ev_dt = max(dates)
    out["n_event_dates"] = len(dates)
    out["chosen_before_latest"] = bool(chosen < ev_dt)
    out["checks"]["date_after_event"] = chosen >= ev_dt
    out["checks"]["days_after_nonneg"] = int(rec.get("days_after_event", -1)) >= 0
    out["checks"]["days_consistent"] = int(rec.get("days_after_event", -999)) == (chosen - ev_dt).days
    return out


def audit_cache(records: list[dict], n: int = 8) -> dict:
    checked = 0
    ok = 0
    for rec in records[:n]:
        p = Path(rec["cache"])
        if not p.is_file():
            continue
        checked += 1
        if sha256_file(p) == rec.get("cache_sha256"):
            ok += 1
    return {"checked": checked, "ok": ok}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="s2_10band_v4")
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in (ROOT / "manifests" / f"{args.version}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_event: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_event[r["event_id"]].append(r)

    rng = random.Random(args.seed)
    per = max(1, args.sample // max(1, len(by_event)))
    sample: list[dict] = []
    for ev, items in sorted(by_event.items()):
        sample += rng.sample(items, min(per, len(items)))

    results = [audit_tile(r) for r in sample]
    checks = sorted({k for r in results for k in r["checks"]})
    summary = {}
    for c in checks:
        vals = [r["checks"][c] for r in results if c in r["checks"]]
        summary[c] = {"n": len(vals), "pass": sum(1 for v in vals if v), "all_pass": bool(vals) and all(vals)}

    cache = audit_cache(rows, n=8)

    # 泄漏
    ev_splits: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        ev_splits[r["event_id"]].add(r["split"])
    ev_leak = {k: sorted(v) for k, v in ev_splits.items() if len(v) > 1}

    boxes = [(r["split"], r["bbox"]) for r in rows if r.get("bbox")]
    overlap = 0
    if len(boxes) <= 4000:
        by_split = defaultdict(list)
        for sp, bb in boxes:
            by_split[sp].append(bb)
        for i, (sp1, b1) in enumerate(boxes):
            for sp2, b2 in boxes[i + 1 :]:
                if sp1 == sp2:
                    continue
                if b1[0] < b2[2] and b2[0] < b1[2] and b1[1] < b2[3] and b2[1] < b1[3]:
                    overlap += 1

    dup = [sid for sid, c in Counter(r["sample_id"] for r in rows).items() if c > 1]

    # 跨 split 邻接：同尺寸 1280 m 图块，格网哈希找 ±1 格内、不同 split 的邻居
    cell = 1280.0
    buckets: dict[tuple[int, int], list[tuple[str, str, tuple]]] = defaultdict(list)
    for r in rows:
        if not r.get("bbox"):
            continue
        bb = r["bbox"]
        cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
        buckets[(int(cx // cell), int(cy // cell))].append((r["split"], r["sample_id"], bb))
    near_pairs = 0
    near_examples: list[str] = []
    for (gx, gy), items in buckets.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for sp1, sid1, b1 in items:
                    for sp2, sid2, b2 in buckets.get((gx + dx, gy + dy), []):
                        if sp1 == sp2:
                            continue
                        gap = max(b2[0] - b1[2], b1[0] - b2[2], b2[1] - b1[3], b1[1] - b2[3])
                        if gap <= 0:
                            near_pairs += 1
                            if len(near_examples) < 5:
                                near_examples.append(f"{sid1}({sp1}) ↔ {sid2}({sp2})")
    near_pairs //= 2

    # 报告
    lines = [
        f"# P0 数据契约审计：`{args.version}`",
        "",
        f"- 抽样 **{len(sample)}** 块（每个事件约 {per} 块，seed={args.seed}），覆盖 {len(by_event)} 个事件。",
        f"- 清单总计 {len(rows):,} 块。",
        "",
        "## 1. 逐项检查",
        "",
        "| 检查 | 抽样数 | 通过 | 结论 |",
        "|---|---:|---:|---|",
    ]
    names = {
        "bands": "10 波段集合与顺序 == S2_10_BANDS",
        "has_scl": "含 SCL",
        "has_mask": "含 MASK",
        "has_dem_unused": "含 DEM（本任务不使用）",
        "dtype_int16": "波段 dtype = int16",
        "dn_plausible": "DN ∈ [0, 30000]",
        "scl_known": "SCL 取值 ⊆ 0–11 ∪ {255}",
        "valid_ratio_match": "valid_ratio 可由 SCL 复算（±0.02）",
        "mask_binary": "MASK 取值 ⊆ {0,1,255}",
        "mask_px_match": "mask_px 可由 MASK 复算",
        "date_match": "date == time_origin + time[idx]",
        "n_timesteps_15": "时相数 = 15",
        "time_index_in_range": "time_index 合法",
        "mask_static_over_time": "15 时相 MASK 一致",
        "date_after_event": "date ≥ event_date",
        "days_after_nonneg": "days_after_event ≥ 0",
        "days_consistent": "days_after_event 与日期差一致",
    }
    for c in checks:
        s = summary[c]
        lines.append(f"| {names.get(c, c)} | {s['n']} | {s['pass']} | {'PASS' if s['all_pass'] else '**FAIL**'} |")

    lines += [
        "",
        "## 2. 缓存与泄漏",
        "",
        f"- npz 缓存哈希核对：{cache['ok']}/{cache['checked']} 一致。",
        f"- 事件跨 split：{('**FAIL** ' + str(ev_leak)) if ev_leak else 'PASS（每个事件只属于一个 split）'}。",
        f"- bbox 跨 split 重叠对：{overlap}。",
        f"- 跨 split 相邻（bbox 间距 ≤ 0 m）：{near_pairs} 对"
        + (f"，例：{'; '.join(near_examples)}" if near_examples else ""),
        f"- 重复 sample_id：{len(dup)}。",
        f"- 多事件图块：抽样中 {sum(1 for r in results if r.get('n_event_dates', 1) > 1)} 块；"
        f"其中所选时相早于最晚事件日期：**{sum(1 for r in results if r.get('chosen_before_latest'))}** 块。"
        "（>0 说明 p20 多事件裁决没有生效，必须重跑该步骤）",
        "",
        "## 3. 抽样明细（前 10 块）",
        "",
        "| sample_id | 事件 | DN 范围 | SCL | MASK | date | 复算 valid | 清单 valid |",
        "|---|---|---|---|---|---|---:|---:|",
    ]
    for r in results[:10]:
        scl = r.get("scl_values", [])
        scl_s = ",".join(str(x) for x in scl[:8]) + ("…" if len(scl) > 8 else "")
        lines.append(
            f"| {r['sample_id']} | {r['event']} | {r.get('dn_range', {}).get('min')}–{r.get('dn_range', {}).get('max')} | "
            f"{scl_s} | {r.get('mask_values')} | {r.get('date_recomputed', '-')} | "
            f"{r.get('valid_ratio_recomputed', '-')} | {r.get('valid_ratio_manifest', '-')} |"
        )

    errs = [(r["sample_id"], e) for r in results for e in r["errors"]]
    lines += ["", "## 4. 异常", ""]
    lines += [f"- {sid}: {e}" for sid, e in errs[:20]] or ["- 无。"]

    lines += [
        "",
        "## 5. 已知限制",
        "",
        "- 本地只有官方 `data_harmonized` 一套产品；`data_raw` 未下载，因此 `raw vs harmonized` 配对消融"
        "（执行计划 P1-4 `Q_SOURCE`）暂不可做，需要时再补约 10 GB。",
        "- DEM 存在于 nc 中但本任务不使用；它属于官方多时相+DEM 基线（P1-7 `B-TEMP`），不进单期主实验。",
        "",
        "## 6. 复现",
        "",
        "```bash",
        f"python scripts/audit_v4_contract.py --version {args.version} --sample {args.sample}",
        "```",
        "",
    ]

    out = ROOT / "reports" / "P0_data_contract_audit.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")

    failed = [c for c, s in summary.items() if not s["all_pass"]]
    print(f"[done] → {out}")
    print(f"[checks] {len(checks)} 项；未全通过: {failed or '无'}")
    print(f"[cache] {cache['ok']}/{cache['checked']}；事件跨 split: {len(ev_leak)}；重叠对: {overlap}")


if __name__ == "__main__":
    main()
