"""P1-3：固定图块数量、改变训练事件覆盖的两组清单构建。

规则**完全由 TRAIN 属性决定**，训练前写出并冻结（含哈希）。不接触 DEV，
不使用任何模型输出；对象统计直接从 GT 掩膜重算，不读评测产物。

流程：
1. 读 `manifests/v4r1/pool_100.jsonl` 的 TRAIN，按事件统计可用量；
2. 逐瓦片从缓存算 `mask_px / valid_px / 对象数 / 六个面积档`；
3. 枚举 k∈{2,3,4} 个事件的全部组合，`N = 该组合可用量之和`；
4. 对同一 N，把 AllEvents 按可用量比例分摊到九个事件（最大余数法），
   再按 ``sha256(sample_id)`` 升序取前 quota 个（确定性、与 DEV 无关）；
5. 计算两组的「前景比例」与「对象面积档分布」，取
   ``score = TVD(面积档) + 相对前景比差`` 最小者；同分取 N 更大者，再同分取事件名序最小；
6. 拟合**两组交集**（共同 TRAIN 核心）的归一化统计，两组共用；
7. 为每组写 manifest / protocol / 一份组成对照表。

产出：
- ``manifests/p13/{few,all}/pool_100.jsonl``
- ``manifests/p13/stats_common_core.json``（两组共用）
- ``manifests/p13/{few,all}/protocol.json``
- ``manifests/p13/selection_rule.json``（冻结规则与全部候选的分数）
- ``reports/p13/composition.md`` / ``.json``
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402

from landslide_seg.config import file_sha256, resolve_path  # noqa: E402
from landslide_seg.training.protocol import canonical_hash  # noqa: E402
from landslide_seg.training.segformer_metrics import label_objects  # noqa: E402
from prepare_segformer_data import compute_stats  # noqa: E402

SOURCE = "manifests/v4r1/pool_100.jsonl"
OUT = ROOT / "manifests" / "p13"
BINS = [("1_4", 1, 4), ("5_9", 5, 9), ("10_19", 10, 19),
        ("20_49", 20, 49), ("50_199", 50, 199), ("ge200", 200, 10**9)]
KS = (2, 3, 4)


def bin_of(area: int) -> str:
    for name, lo, hi in BINS:
        if lo <= area <= hi:
            return name
    return "ge200"


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(rows: list[dict], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def sha_order(sid: str) -> str:
    return hashlib.sha256(sid.encode()).hexdigest()


def tile_attributes(rec: dict) -> dict:
    z = np.load(resolve_path(rec["cache"]))
    mask = z["mask"].astype(bool)
    valid = z["valid"].astype(bool)
    lab, n = label_objects(mask & valid, 2)
    areas = np.bincount(lab.ravel(), minlength=n + 1)[1:] if n else np.zeros(0, dtype=int)
    bins = Counter(bin_of(int(a)) for a in areas.tolist())
    return {
        "sample_id": rec["sample_id"],
        "event_id": rec["event_id"],
        "mask_px": int(mask.sum()),
        "valid_px": int(valid.sum()),
        "n_objects": int(n),
        "obj_area_sum": int(areas.sum()) if n else 0,
        "bins": {b: int(bins.get(b, 0)) for b, _, _ in BINS},
    }


def aggregate(tiles: list[dict]) -> dict:
    mask = sum(t["mask_px"] for t in tiles)
    valid = sum(t["valid_px"] for t in tiles)
    bins = {b: sum(t["bins"][b] for t in tiles) for b, _, _ in BINS}
    nobj = sum(bins.values())
    return {
        "n": len(tiles),
        "mask_px": mask, "valid_px": valid,
        "fg_ratio": (mask / valid) if valid else 0.0,
        "n_objects": nobj,
        "bin_props": {b: (bins[b] / nobj if nobj else 0.0) for b in bins},
        "bin_counts": bins,
        "mean_obj_area": (sum(t["obj_area_sum"] for t in tiles) / nobj) if nobj else 0.0,
        "events": dict(sorted(Counter(t["event_id"] for t in tiles).items())),
    }


def tvd(a: dict, b: dict) -> float:
    return 0.5 * sum(abs(a[k] - b[k]) for k in a)


def rel_diff(x: float, y: float) -> float:
    s = abs(x) + abs(y)
    return (2 * abs(x - y) / s) if s else 0.0


def quota_alloc(n_total: int, avail: dict[str, int]) -> dict[str, int]:
    """最大余数法按可用量比例分摊，且不超过各事件可用量。"""
    total = sum(avail.values())
    raw = {e: n_total * v / total for e, v in avail.items()}
    base = {e: int(np.floor(raw[e])) for e in avail}
    remain = n_total - sum(base.values())
    for e in sorted(avail, key=lambda k: (-(raw[k] - base[k]), k)):
        if remain <= 0:
            break
        if base[e] < avail[e]:
            base[e] += 1
            remain -= 1
    # 兜底：仍有余量时按可用量降序补
    if remain > 0:
        for e in sorted(avail, key=lambda k: (-avail[k], k)):
            take = min(remain, avail[e] - base[e])
            base[e] += take
            remain -= take
            if remain <= 0:
                break
    assert sum(base.values()) == n_total, (sum(base.values()), n_total)
    assert all(base[e] <= avail[e] for e in avail)
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--force", action="store_true", help="重算逐瓦片属性（默认读缓存）")
    args = ap.parse_args()

    src = resolve_path(args.source)
    rows = read_jsonl(src)
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "val"]
    dev_ids = sorted(r["sample_id"] for r in dev)

    # ---- 1. 逐瓦片属性（缓存到磁盘，避免重复读 npz）----
    attr_path = OUT / "tile_attributes.json"
    if attr_path.is_file() and not args.force:
        tiles = json.loads(attr_path.read_text(encoding="utf-8"))
        print(f"[cache] 复用 {attr_path} ({len(tiles)} 瓦片)")
    else:
        tiles = []
        for i, r in enumerate(train, 1):
            tiles.append(tile_attributes(r))
            if i % 200 == 0:
                print(f"  attrs {i}/{len(train)}", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        attr_path.write_text(json.dumps(tiles, ensure_ascii=False), encoding="utf-8")
        print(f"[done] 逐瓦片属性 -> {attr_path}")

    by_event: dict[str, list[dict]] = defaultdict(list)
    for t in tiles:
        by_event[t["event_id"]].append(t)
    avail = {e: len(v) for e, v in by_event.items()}
    print("[avail]", json.dumps(dict(sorted(avail.items(), key=lambda x: -x[1])), ensure_ascii=False))

    # ---- 2. 枚举组合 ----
    candidates = []
    for k in KS:
        for combo in itertools.combinations(sorted(avail), k):
            n = sum(avail[e] for e in combo)
            if n < 200:
                continue
            few_tiles = [t for e in combo for t in by_event[e]]
            few_agg = aggregate(few_tiles)
            q = quota_alloc(n, avail)
            rng = {e: sorted(by_event[e], key=lambda t: sha_order(t["sample_id"]))[:q[e]] for e in avail}
            all_tiles = [t for e in avail for t in rng[e]]
            assert len(all_tiles) == n, (len(all_tiles), n)
            all_agg = aggregate(all_tiles)
            score = tvd(few_agg["bin_props"], all_agg["bin_props"]) + rel_diff(
                few_agg["fg_ratio"], all_agg["fg_ratio"])
            candidates.append({
                "k": k, "events": list(combo), "n": n, "score": score,
                "tvd_bins": tvd(few_agg["bin_props"], all_agg["bin_props"]),
                "rel_fg": rel_diff(few_agg["fg_ratio"], all_agg["fg_ratio"]),
                "few_agg": few_agg, "all_agg": all_agg, "quota": q,
            })
    candidates.sort(key=lambda c: (c["score"], -c["n"], c["events"]))
    best = candidates[0]
    print(f"[pick] k={best['k']} N={best['n']} events={best['events']} score={best['score']:.4f}")

    # ---- 3. 落地两组清单 ----
    few_ids = sorted(t["sample_id"] for e in best["events"] for t in by_event[e])
    few_set = set(few_ids)
    idx = {t["sample_id"]: t for t in tiles}
    rec_index = {r["sample_id"]: r for r in train}

    all_ids = []
    for e in sorted(avail):
        pool = sorted(by_event[e], key=lambda t: sha_order(t["sample_id"]))[: best["quota"][e]]
        all_ids.extend(t["sample_id"] for t in pool)
    all_ids = sorted(all_ids)
    assert len(all_ids) == len(few_ids)
    assert len(set(few_ids)) == len(few_ids) and len(set(all_ids)) == len(all_ids)

    common = sorted(few_set & set(all_ids))
    print(f"[common core] {len(common)} / N={len(few_ids)}")

    def build_manifest(ids: list[str], dest: Path) -> Path:
        recs = [dict(rec_index[sid], split="train", protocol_role="TRAIN", pool_fraction=100) for sid in ids]
        recs += [dict(r, split="val", protocol_role="DEV") for r in dev]
        write_jsonl(recs, dest)
        return dest

    few_manifest = build_manifest(few_ids, OUT / "few" / "pool_100.jsonl")
    all_manifest = build_manifest(all_ids, OUT / "all" / "pool_100.jsonl")

    # ---- 4. 共用统计：只拟合共同核心 ----
    stats_path = OUT / "stats_common_core.json"
    core_recs = [rec_index[sid] for sid in common]
    stats = compute_stats(core_recs, stats_path)
    stats.update(
        data_version="p13_events",
        fit_split="train_common_core_few∩all",
        fit_sample_ids=common,
        fit_events=dict(Counter(rec_index[sid]["event_id"] for sid in common)),
        source_manifest_sha256=file_sha256(src),
        unit="DN, official harmonized product",
        note=("由 FewEvents 与 AllEvents 两组的交集拟合，两组共用同一份；"
              "避免少事件组经由 '全九事件统计' 额外接触其他事件的辐射分布。DEV/FINAL/DIAG 不参与。"),
    )
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 5. protocol（各自 members / pools，便于 validate_protocol 通过）----
    ident = {r["sample_id"]: {k: r[k] for k in ("event_id", "cache", "cache_sha256", "date", "time_index")}
             for r in rows}

    def build_protocol(manifest: Path, train_ids: list[str], dest: Path) -> Path:
        members = {}
        for sid in list(train_ids) + dev_ids:
            members[sid] = dict(ident[sid], role=("train" if sid in set(train_ids) else "val"))
        payload = {
            "version": "p13_event_coverage",
            "frozen": True,
            "design": {
                "train": sorted({ident[s]["event_id"] for s in train_ids}),
                "dev": ["hiroshima", "thrissur"],
                "diagnostic": ["italy"], "final": ["newzealand"],
                "exclude_small": ["nepal", "usa_alaska"],
            },
            "members": members,
            "dev_ids": dev_ids,
            "pools": {"100": sorted(train_ids)},
            "files": {
                str(manifest).replace("\\", "/"): file_sha256(manifest),
                str(stats_path).replace("\\", "/"): file_sha256(stats_path),
            },
            "notes": {
                "dev": "与 v4r1 完全相同的 669 块，未改动",
                "stats": "只由两组共同核心样本拟合，两组共用",
                "source": str(src).replace("\\", "/"),
            },
        }
        payload["sha256"] = canonical_hash(payload)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        return dest

    few_proto = build_protocol(few_manifest, few_ids, OUT / "few" / "protocol.json")
    all_proto = build_protocol(all_manifest, all_ids, OUT / "all" / "protocol.json")

    # ---- 6. 冻结规则 ----
    rule = {
        "source_manifest": str(src).replace("\\", "/"),
        "source_sha256": file_sha256(src),
        "selection_inputs": "仅 TRAIN 的 GT 掩膜属性（前景像素、对象数、对象面积档）",
        "dev_used": False,
        "k_range": list(KS),
        "score": "TVD(对象面积档比例) + 相对前景比例差",
        "tie_break": ["N 更大", "事件名序最小"],
        "chosen": {
            "few_events": best["events"], "n": best["n"], "score": best["score"],
            "tvd_bins": best["tvd_bins"], "rel_fg": best["rel_fg"],
            "quota": best["quota"],
        },
        "common_core_n": len(common),
        "dev_ids_sha256": hashlib.sha256("".join(dev_ids).encode()).hexdigest(),
        "candidates": [
            {"k": c["k"], "events": c["events"], "n": c["n"], "score": round(c["score"], 6),
             "tvd_bins": round(c["tvd_bins"], 6), "rel_fg": round(c["rel_fg"], 6)}
            for c in candidates
        ],
        "files": {
            "few_manifest": str(few_manifest.relative_to(ROOT)).replace("\\", "/"),
            "few_manifest_sha256": file_sha256(few_manifest),
            "all_manifest": str(all_manifest.relative_to(ROOT)).replace("\\", "/"),
            "all_manifest_sha256": file_sha256(all_manifest),
            "stats": str(stats_path.relative_to(ROOT)).replace("\\", "/"),
            "stats_sha256": file_sha256(stats_path),
            "few_protocol_sha256": json.loads(few_proto.read_text(encoding="utf-8"))["sha256"],
            "all_protocol_sha256": json.loads(all_proto.read_text(encoding="utf-8"))["sha256"],
        },
        "manifest_sha256": {
            "few_manifest": file_sha256(few_manifest),
            "all_manifest": file_sha256(all_manifest),
            "stats": file_sha256(stats_path),
        },
    }
    (OUT / "selection_rule.json").write_text(json.dumps(rule, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 7. 组成对照表 ----
    fa = aggregate([idx[s] for s in few_ids])
    aa = aggregate([idx[s] for s in all_ids])
    ca = aggregate([idx[s] for s in common])
    comp = {"few": fa, "all": aa, "common_core": ca, "chosen": best["events"], "n": best["n"],
            "score": best["score"], "quota": best["quota"]}
    (OUT / "composition.json").write_text(json.dumps(comp, ensure_ascii=False, indent=1), encoding="utf-8")

    L = ["# P1-3 事件覆盖对照：两组训练集的组成\n",
         f"来源 `{SOURCE}`（sha `{file_sha256(src)[:16]}…`）；选择规则只依赖 TRAIN 的 GT 属性，未使用 DEV 或任何模型输出。\n",
         f"N = **{best['n']}**；FewEvents = {best['events']}（{best['k']} 个事件）；"
         f"AllEvents = 九个事件按可用量比例分摊同一 N。\n",
         "## 1. 总体组成\n",
         "| 项 | FewEvents | AllEvents | 共同核心 |", "|---|---:|---:|---:|",
         f"| 图块数 | {fa['n']} | {aa['n']} | {ca['n']} |",
         f"| 事件数 | {len(fa['events'])} | {len(aa['events'])} | {len(ca['events'])} |",
         f"| 前景像素占比 | {fa['fg_ratio']:.5f} | {aa['fg_ratio']:.5f} | {ca['fg_ratio']:.5f} |",
         f"| GT 对象数 | {fa['n_objects']} | {aa['n_objects']} | {ca['n_objects']} |",
         f"| 平均对象面积(px) | {fa['mean_obj_area']:.2f} | {aa['mean_obj_area']:.2f} | {ca['mean_obj_area']:.2f} |",
         "\n## 2. 对象面积档分布（对象数占比）\n",
         "| 面积档 | FewEvents | AllEvents | 差值 |", "|---|---:|---:|---:|"]
    for b, _, _ in BINS:
        L.append(f"| {b} | {fa['bin_props'][b]:.4f} | {aa['bin_props'][b]:.4f} | "
                 f"{fa['bin_props'][b] - aa['bin_props'][b]:+.4f} |")
    L.append(f"\nTVD(面积档) = {best['tvd_bins']:.4f}；相对前景比差 = {best['rel_fg']:.4f}；合计 score = {best['score']:.4f}\n")
    L.append("## 3. 每事件配额\n")
    L.append("| 事件 | 可用 | FewEvents | AllEvents | 全量占比 |")
    L.append("|---|---:|---:|---:|---:|")
    for e in sorted(avail, key=lambda x: -avail[x]):
        L.append(f"| {e} | {avail[e]} | {fa['events'].get(e, 0)} | "
                 f"{best['quota'][e]} | {avail[e]/sum(avail.values()):.4f} |")
    L.append(f"\n共同核心（两组都含）{ca['n']} 块 = FewEvents 的 {ca['n']/fa['n']:.1%}、AllEvents 的 {ca['n']/aa['n']:.1%}。\n")
    L.append("## 4. 全部候选（按 score 升序，前 20）\n")
    L.append("| k | 事件 | N | score | TVD | 相对前景 |")
    L.append("|---:|---|---:|---:|---:|---:|")
    for c in candidates[:20]:
        L.append(f"| {c['k']} | {', '.join(c['events'])} | {c['n']} | {c['score']:.4f} | "
                 f"{c['tvd_bins']:.4f} | {c['rel_fg']:.4f} |")
    (ROOT / "reports" / "p13").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "p13" / "composition.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"[done] few={few_manifest} all={all_manifest} stats={stats_path}")
    print(f"[done] rule={OUT / 'selection_rule.json'}")
    print(f"[done] {ROOT / 'reports' / 'p13' / 'composition.md'}")


if __name__ == "__main__":
    main()
