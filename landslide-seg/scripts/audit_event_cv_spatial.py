"""事件交叉验证（EVENT_CV_V5）冻结前的空间关系审计。

计划要求：统一坐标后检查重叠与 10 m 邻接；同 inventory 或相邻事件必须同侧；
若发现计划外的跨组联系，停止冻结并提交修订分组。

方法：
- 同一 CRS 内：直接用 bbox（投影米制）做轴对齐判定，
  「重叠」= x/y 区间均相交；「10 m 邻接」= x/y 区间间隙均 ≤ 10 m。
- 跨 CRS：把中心点投到 EPSG:4326 后算测地距离，报告每个事件对的最小距离
  （不同 UTM 带无法做轴对齐判定，只能给下界）。

产出 ``manifests/event_cv_v5/spatial_audit.json`` 与 ``.md``。
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402

OUT = ROOT / "manifests" / "event_cv_v5"
ADJ_M = 10.0          # 计划规定的「10 米内邻接」
TILE_M = 1280.0       # 128 px × 10 m


def load_tiles() -> list[dict]:
    rows = []
    for name in ("trainval.jsonl", "test.jsonl"):
        p = ROOT / "manifests" / "official_single_geo" / name
        rows += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows


def geodetic_km(lat1, lon1, lat2, lon2):
    """向量化 haversine，单位 km。"""
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def main() -> None:
    rows = load_tiles()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"图块 {len(rows)}，事件 {len({r['event_id'] for r in rows})}")

    # ---- 中心点转经纬度 ----
    from pyproj import Transformer
    by_crs: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_crs[str(r["crs"])].append(r)
    ll = {}
    for crs, items in by_crs.items():
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        xs = np.array([it["center_xy"][0] for it in items], dtype=np.float64)
        ys = np.array([it["center_xy"][1] for it in items], dtype=np.float64)
        lon, lat = tr.transform(xs, ys)
        for it, la, lo in zip(items, lat, lon):
            ll[it["sample_id"]] = (float(la), float(lo))
    print(f"[坐标] 已转换 {len(ll)} 个中心点，{len(by_crs)} 个 CRS 分组")

    # ---- 同 CRS 内：轴对齐重叠 / 10 m 邻接 ----
    same_pairs: dict[tuple[str, str], dict] = defaultdict(lambda: {"overlap": 0, "adjacent": 0, "min_gap_m": None})
    for crs, items in by_crs.items():
        ev = np.array([it["event_id"] for it in items])
        bb = np.array([it["bbox"] for it in items], dtype=np.float64)  # x0,y0,x1,y1
        n = len(items)
        for i in range(n):
            # 向量化比较 i 与 i+1..
            j = np.arange(i + 1, n)
            if j.size == 0:
                continue
            gx = np.maximum(bb[j, 0] - bb[i, 2], bb[i, 0] - bb[j, 2])
            gy = np.maximum(bb[j, 1] - bb[i, 3], bb[i, 1] - bb[j, 3])
            gap = np.maximum(gx, gy)          # <0 表示重叠；>=0 为轴对齐间隙
            ov = (gx < 0) & (gy < 0)
            ad = (~ov) & (gx <= ADJ_M) & (gy <= ADJ_M)
            for idx in range(j.size):
                k = int(j[idx])
                key = tuple(sorted((ev[i], ev[k])))
                rec = same_pairs[key]
                if ov[idx]:
                    rec["overlap"] += 1
                if ad[idx]:
                    rec["adjacent"] += 1
                g = float(gap[idx])
                rec["min_gap_m"] = g if rec["min_gap_m"] is None else min(rec["min_gap_m"], g)

    # ---- 跨事件：测地最小距离 ----
    events = sorted({r["event_id"] for r in rows})
    ev_pts = {e: np.array([ll[r["sample_id"]] for r in rows if r["event_id"] == e]) for e in events}
    cross = {}
    for a_i, a in enumerate(events):
        for b in events[a_i + 1:]:
            A, B = ev_pts[a], ev_pts[b]
            best = np.inf
            for k in range(0, len(A), 256):
                d = geodetic_km(A[k:k + 256, None, 0], A[k:k + 256, None, 1], B[None, :, 0], B[None, :, 1])
                best = min(best, float(d.min()))
            cross[f"{a}|{b}"] = round(best, 3)

    # ---- 汇总成事件对表 ----
    crs_of = {e: {r["crs"] for r in rows if r["event_id"] == e} for e in events}
    pairs = []
    for a_i, a in enumerate(events):
        for b in events[a_i + 1:]:
            key = (a, b)
            s = same_pairs.get(key, {"overlap": 0, "adjacent": 0, "min_gap_m": None})
            pairs.append({
                "event_a": a, "event_b": b,
                "same_crs": crs_of[a] == crs_of[b],
                "overlap_pairs": s["overlap"], "adjacent_pairs": s["adjacent"],
                "min_gap_m_same_crs": s["min_gap_m"],
                "min_center_km": cross[f"{a}|{b}"],
            })
    pairs.sort(key=lambda p: p["min_center_km"])

    risk = [p for p in pairs if p["overlap_pairs"] > 0 or p["adjacent_pairs"] > 0 or p["min_center_km"] < 2.0]

    rep = {
        "n_tiles": len(rows), "n_events": len(events),
        "tiles_per_event": dict(sorted(Counter(r["event_id"] for r in rows).items(), key=lambda x: -x[1])),
        "crs_policy": {crs: sorted({it["event_id"] for it in items}) for crs, items in by_crs.items()},
        "adjacency_threshold_m": ADJ_M, "tile_size_m": TILE_M,
        "pairs": pairs,
        "flagged_pairs": risk,
        "method_note": ("同 CRS 内为 bbox 轴对齐精确判定（重叠/10 m 邻接）；跨 CRS 无法轴对齐，"
                        "给出中心点测地最小距离作为下界，<2 km 视为需人工确认"),
    }
    (OUT / "spatial_audit.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")

    L = ["# EVENT_CV_V5 空间关系审计\n",
         f"- 图块 {len(rows)}，事件 {len(events)}",
         f"- 邻接判据：轴对齐间隙 ≤ {ADJ_M:.0f} m；图块跨度 {TILE_M:.0f} m",
         "- 同 CRS 内为精确 bbox 判定；跨 CRS 只能给中心点测地距离下界（<2 km 需人工确认）\n",
         "## 需要关注的事件对\n",
         "| 事件 A | 事件 B | 同 CRS | 重叠对数 | 10m 邻接对数 | 同CRS最小间隙(m) | 最小中心距(km) |",
         "|---|---|---|---:|---:|---:|---:|"]
    if risk:
        for p in risk:
            gap = "-" if p["min_gap_m_same_crs"] is None else "{:.1f}".format(p["min_gap_m_same_crs"])
            L.append("| {} | {} | {} | {} | {} | {} | {:.2f} |".format(
                p["event_a"], p["event_b"], "是" if p["same_crs"] else "否",
                p["overlap_pairs"], p["adjacent_pairs"], gap, p["min_center_km"]))
    else:
        L.append("| — | — | — | 0 | 0 | — | — |")
    L += ["\n## 全部事件对（按最小中心距升序，前 12）\n",
          "| 事件 A | 事件 B | 重叠 | 邻接 | 最小中心距(km) |", "|---|---|---:|---:|---:|"]
    for p in pairs[:12]:
        L.append(f"| {p['event_a']} | {p['event_b']} | {p['overlap_pairs']} | {p['adjacent_pairs']} | {p['min_center_km']:.2f} |")
    (OUT / "spatial_audit.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print("\n需关注的事件对：")
    for p in risk:
        print(f"  {p['event_a']:14s} {p['event_b']:14s} overlap={p['overlap_pairs']:4d} "
              f"adj={p['adjacent_pairs']:3d} gap={p['min_gap_m_same_crs']} min_km={p['min_center_km']:.2f}")
    print(f"[saved] {OUT}/spatial_audit.md")


if __name__ == "__main__":
    main()
