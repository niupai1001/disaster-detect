"""构建 EVENT_CV_V5：已知历史域上的开发型事件交叉验证。

四折定义（外部留出组 / 内部验证组 / 训练），图块数已与计划核对一致：

| 折 | 外部留出 | 图块 | 内部验证 | 图块 | 训练 |
|----|---------|-----:|---------|-----:|-----:|
| F1 | Italy | 1776 | New Zealand | 453 | 2486 |
| F2 | Kyrgyz联合组 + Hokkaido + China | 979 | New Zealand | 453 | 3283 |
| F3 | New Zealand + Hiroshima + Thrissur + USA Puerto Rico | 985 | Kyrgyz联合组 | 773 | 2957 |
| F4 | Chimanimani + Dominica Maria + Indonesia + Itogon | 975 | New Zealand | 453 | 3287 |

Kyrgyzstan1 与 2 在空间审计中相距 1.28 km、存在 1 对 10 m 邻接，按不可拆分组合处理。

产出（全部在 ``manifests/event_cv_v5/``）：
- ``folds.json``：折定义 + 文件哈希（冻结记录）
- ``fold{F}/pool_100.jsonl``：train / val / test 三类记录
- ``fold{F}/stats_train.json``：只用本折 TRAIN 拟合
- ``fold{F}/protocol.json``
- ``leakage.md`` / ``leakage.json``：分区与事件互斥检查
"""

from __future__ import annotations

import argparse
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

from landslide_seg.config import file_sha256, resolve_path  # noqa: E402
from landslide_seg.training.protocol import canonical_hash  # noqa: E402
from prepare_segformer_data import compute_stats  # noqa: E402

OUT = ROOT / "manifests" / "event_cv_v5"
SRC = ("manifests/official_single_geo/trainval.jsonl", "manifests/official_single_geo/test.jsonl")

KYRGYZ = ("kyrgyzstan1", "kyrgyzstan2")
FOLDS = {
    "F1": {"outer": {"italy"}, "inner": {"newzealand"}, "expect": (1776, 453, 2486)},
    "F2": {"outer": set(KYRGYZ) | {"hokkaido", "china"}, "inner": {"newzealand"}, "expect": (979, 453, 3283)},
    "F3": {"outer": {"newzealand", "hiroshima", "thrissur", "usa_puertorico"},
           "inner": set(KYRGYZ), "expect": (985, 773, 2957)},
    "F4": {"outer": {"chimanimani", "dominicamaria", "indonesia", "itogon"},
           "inner": {"newzealand"}, "expect": (975, 453, 3287)},
}


def load() -> list[dict]:
    rows = []
    for rel in SRC:
        rows += [json.loads(l) for l in resolve_path(rel).read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = load()
    ids = {r["sample_id"] for r in rows}
    if len(ids) != len(rows):
        raise SystemExit("样本 id 重复，停止")
    ev_ok = all(e in {r["event_id"] for r in rows} for f in FOLDS.values() for e in f["outer"] | f["inner"])
    if not ev_ok:
        raise SystemExit("折定义里出现清单不存在的事件")
    print(f"总图块 {len(rows)}，事件 {len({r['event_id'] for r in rows})}")

    folds_meta: dict = {"version": "EVENT_CV_V5",
                        "note": "已知历史域上的开发型事件交叉验证；非最终盲测",
                        "pool": {"n_tiles": len(rows),
                                 "source": [str(s) for s in SRC],
                                 "caveat": "受官方任务筛选影响（4,715 / 6,048），不能冒充全部监督数据"},
                        "kyrgyz_merge_reason": "空间审计：kyrgyzstan1/2 最小中心距 1.28 km，存在 1 对 10 m 邻接",
                        "folds": {}}

    leak_rows = []
    for fold, spec in FOLDS.items():
        d = OUT / fold
        d.mkdir(parents=True, exist_ok=True)
        outer, inner = spec["outer"], spec["inner"]
        if outer & inner:
            raise SystemExit(f"{fold}: 外部留出与内部验证事件重叠")
        recs = []
        for r in rows:
            e = r["event_id"]
            role = "test" if e in outer else ("val" if e in inner else "train")
            recs.append(dict(r, split=role, protocol_role=role.upper(), fold=fold))
        train = [r for r in recs if r["split"] == "train"]
        val = [r for r in recs if r["split"] == "val"]
        test = [r for r in recs if r["split"] == "test"]
        got = (len(test), len(val), len(train))
        if got != spec["expect"]:
            raise SystemExit(f"{fold}: 图块数 {got} 与计划 {spec['expect']} 不符")

        # ---- 分区互斥 ----
        s_tr, s_va, s_te = ({r["sample_id"] for r in x} for x in (train, val, test))
        assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), fold
        assert s_tr | s_va | s_te == ids, fold
        ev_tr = {r["event_id"] for r in train}
        ev_va = {r["event_id"] for r in val}
        ev_te = {r["event_id"] for r in test}
        assert not (ev_tr & ev_te) and not (ev_va & ev_te) and not (ev_tr & ev_va), fold
        # Kyrgyz 组合必须同侧
        for grp in (set(KYRGYZ),):
            sides = [bool(grp & ev_tr), bool(grp & ev_va), bool(grp & ev_te)]
            assert sum(sides) == 1, f"{fold}: 不可拆分组合被拆开 {grp}"

        # 训练用清单只含 train+val（validate_protocol 只认这两类角色），outer 单独一份
        dest = d / "pool_100.jsonl"
        dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train + val),
                        encoding="utf-8")
        test_dest = d / "test.jsonl"
        test_dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in test),
                             encoding="utf-8")

        stats_path = d / "stats_train.json"
        stats = compute_stats(train, stats_path)
        stats.update(data_version="event_cv_v5", fold=fold, fit_split=f"EVENT_CV_V5/{fold}/TRAIN",
                     fit_sample_ids=[r["sample_id"] for r in train],
                     fit_events=dict(Counter(r["event_id"] for r in train)),
                     note="只由本折 TRAIN 的有效像素拟合；内部验证与外部留出均不参与")
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")

        ident = {r["sample_id"]: {k: r[k] for k in ("event_id", "cache", "cache_sha256", "date", "time_index")}
                 for r in recs}
        members = {sid: dict(v, role=("train" if sid in s_tr else ("val" if sid in s_va else "test")))
                   for sid, v in ident.items()}
        payload = {
            "version": f"EVENT_CV_V5/{fold}",
            "frozen": True,
            "design": {"outer": sorted(outer), "inner": sorted(inner),
                       "train_events": sorted(ev_tr)},
            "members": members,
            "dev_ids": sorted(s_va),
            "test_ids": sorted(s_te),
            "pools": {"100": sorted(s_tr)},
            "files": {
                str(dest).replace("\\", "/"): file_sha256(dest),
                str(test_dest).replace("\\", "/"): file_sha256(test_dest),
                str(stats_path).replace("\\", "/"): file_sha256(stats_path),
            },
            "notes": {
                "role": "开发型事件交叉验证：外部留出不参与任何选择",
                "stats": "仅本折 TRAIN",
                "selection": "内部验证事件平均 IoU@0.5，同分取较早 checkpoint",
            },
        }
        payload["sha256"] = canonical_hash(payload)
        (d / "protocol.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

        folds_meta["folds"][fold] = {
            "outer_events": sorted(outer), "inner_events": sorted(inner),
            "n_outer": len(test), "n_inner": len(val), "n_train": len(train),
            "train_events": sorted(ev_tr),
            "manifest_sha256": file_sha256(dest),
            "test_manifest_sha256": file_sha256(test_dest),
            "stats_sha256": file_sha256(stats_path),
            "protocol_sha256": payload["sha256"],
        }
        leak_rows.append({"fold": fold, "n_train": len(train), "n_val": len(val), "n_test": len(test),
                          "event_overlap_train_test": sorted(ev_tr & ev_te),
                          "event_overlap_train_val": sorted(ev_tr & ev_va),
                          "event_overlap_val_test": sorted(ev_va & ev_te),
                          "sample_id_overlap": 0})
        print(f"[{fold}] train {len(train)} / inner {len(val)} / outer {len(test)}  "
              f"outer={sorted(outer)}  inner={sorted(inner)}")

    (OUT / "folds.json").write_text(json.dumps(folds_meta, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT / "leakage.json").write_text(json.dumps(leak_rows, indent=1, ensure_ascii=False), encoding="utf-8")

    L = ["# EVENT_CV_V5 泄漏检查\n",
         f"- 总图块 {len(rows)}，四折共用同一池",
         "- 每折内 train / inner-val / outer-test 三者在样本 id 与事件上均互斥（构建时断言）",
         "- kyrgyzstan1+2 作为不可拆分组，四折中均未被拆开\n",
         "| 折 | 训练 | 内部验证 | 外部留出 | 事件重叠（train∩test） | 事件重叠（train∩val） |",
         "|---|---:|---:|---:|---|---|"]
    for r in leak_rows:
        L.append(f"| {r['fold']} | {r['n_train']} | {r['n_val']} | {r['n_test']} | "
                 f"{'无' if not r['event_overlap_train_test'] else r['event_overlap_train_test']} | "
                 f"{'无' if not r['event_overlap_train_val'] else r['event_overlap_train_val']} |")
    L.append("\n空间层面：`spatial_audit.md` 显示 78 个事件对中仅 kyrgyzstan1–2（1.28 km、1 对 10 m 邻接）"
             "存在接触，其余全部 >500 km，跨组无重叠也无邻接。")
    (OUT / "leakage.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n[saved] {OUT}/folds.json 与 leakage.md")


if __name__ == "__main__":
    main()
