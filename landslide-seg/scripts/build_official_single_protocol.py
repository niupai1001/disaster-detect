"""构建 BENCH_OFFICIAL 单期协议：官方 split 锁定、成员映射、manifest、TRAIN-only 统计。

对应执行计划 `docs/next_execution_plan_official_single_2026-10-20.md` 的步骤 1–3。

只做数据与协议，不训练。产物：

- ``manifests/official_single/protocol_lock.json``：commit、文件哈希、成员数、规则
- ``manifests/official_single/mapping.csv``：官方 ID → 本地 nc，逐 split
- ``manifests/official_single/overlap.json`` / ``.md``：与现有 DEV/DIAG/FINAL 的交集
- ``manifests/official_single/trainval.jsonl``、``test.jsonl``
- ``manifests/official_single/stats_train.json``（只用官方 TRAIN 拟合）
- ``manifests/official_single/protocol.json``
- ``manifests/official_single/exclusions.jsonl``（每块被排除的原因）

关键约束：
- 不猜 split；全部来自固定 commit 下的 ``tasks/S12LS-LD/harmonized/s2/splits.json``。
- 单期选择沿用现有灾后规则（最早可用，窗口内无解则最低云 fallback），不按任何分数挑日期。
- 统计只用官方 TRAIN 的图块，不复用含旧 VAL/TEST 的统计。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
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

from landslide_seg.config import file_sha256, load_segformer_config, resolve_path  # noqa: E402
from landslide_seg.training.protocol import canonical_hash  # noqa: E402
from prepare_segformer_data import compute_stats, process_one  # noqa: E402

REPO = ROOT / "vendor" / "official" / "Sen12Landslides"
TASK = "tasks/S12LS-LD/harmonized/s2"
OUT = ROOT / "manifests" / "official_single"
CACHE = ROOT / "data" / "s2_official_single"
REF_CFG = "configs/segformer/p2m1/segformer_b0_latedetail_alltrain_s1234_ctl.yaml"
SPLITS = ("train", "val", "test")


def git_commit() -> dict:
    def run(*args):
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()
    return {"commit": run("rev-parse", "HEAD"), "date": run("log", "-1", "--format=%cI"),
            "subject": run("log", "-1", "--format=%s"),
            "url": "https://github.com/PaulH97/Sen12Landslides"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=REF_CFG, help="只取数据参数（窗口/阈值），不改模型")
    ap.add_argument("--force", action="store_true", help="重算已存在的缓存")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_segformer_config(args.config)["segformer"]
    dcfg = cfg["data"]
    splits_path = REPO / TASK / "splits.json"
    norm_path = REPO / TASK / "norm.json"
    official = json.loads(splits_path.read_text(encoding="utf-8"))

    # ---------- 1. 协议锁定 ----------
    lock = {
        "protocol": "BENCH_OFFICIAL/S12LS-LD/harmonized/s2",
        "source": git_commit(),
        "files": {
            "splits.json": {"path": str(splits_path.relative_to(ROOT)).replace("\\", "/"),
                            "sha256": file_sha256(splits_path)},
            "norm.json": {"path": str(norm_path.relative_to(ROOT)).replace("\\", "/"),
                          "sha256": file_sha256(norm_path)},
        },
        "counts_official": {sp: len(official[sp]) for sp in SPLITS},
        "total_official": sum(len(official[sp]) for sp in SPLITS),
        "input_definition": {
            "bands": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
            "grid": [128, 128],
            "official_modalities": "S2 10 band + SCL + DEM, 15 timesteps, harmonized",
            "this_stage": "单期（十波段，无 DEM，无多时相）",
        },
        "single_date_rule": {
            "window_days": int(dcfg["post_event_window_days"]),
            "max_cloud": float(dcfg["max_cloud"]),
            "min_valid_ratio": float(dcfg["min_valid_ratio"]),
            "max_label_invalid_ratio": float(dcfg["max_label_invalid_ratio"]),
            "rule": "灾后窗口内最早可用（cloud<=max_cloud 且 scl_valid>=min_valid_ratio）；"
                    "窗口内无解则取全灾后最低云并标 temporal_fallback",
            "note": "不依据 VAL/TEST 分数挑日期",
        },
        "metric_definition": {
            "threshold": float(cfg["eval"]["threshold"]),
            "pixel": "pooled foreground IoU/Dice/Precision/Recall + AP（阈值无关）",
            "object": "8 邻接对象，IoU>=0.25 一对一匹配，六档面积",
            "boundary": f"1px 容差边界 F1（tol={cfg['eval']['boundary_tol_px']}）",
            "selection": "官方 VAL 前景 pooled IoU@0.5 选 best；TEST 只评测不选模",
            "official_metric_note": "官方表值为 binary foreground AP/F1/IoU（75 epochs，3 seeds，多时相+DEM），"
                                    "与本项目 pooled 单期指标不是同一口径，保留双口径命名",
        },
        "training_budget": {"seed": int(cfg["seed"]), "batch": int(cfg["train"]["batch_size"]),
                            "max_steps": int(cfg["train"]["max_steps"]), "amp": bool(cfg["train"]["amp"]),
                            "note": "固定更新预算的单期变体，不声称复现官方 75 epochs"},
    }

    # ---------- 2. 逐成员映射 ----------
    rows, manifest, excluded = [], [], []
    for sp in SPLITS:
        for rec in official[sp]:
            sid = rec["id"]
            local = ROOT / "data" / "sen12" / Path(rec["s2"]).name
            entry = {"official_id": sid, "official_split": sp, "inventory": rec["inventory"],
                     "official_s2": rec["s2"], "local_nc": str(local).replace("\\", "/"),
                     "official_pixel_annotated": rec.get("pixel_annotated"),
                     "exists": local.is_file()}
            if not local.is_file():
                entry["status"] = "missing_local_nc"
                rows.append(entry)
                excluded.append({"sample_id": sid, "split": sp, "reasons": ["local nc 不存在"]})
                continue
            out, rej = process_one(local, dcfg, CACHE)
            if out is None:
                entry["status"] = "excluded"
                entry["reason"] = "; ".join(rej["reasons"]) if rej else "unknown"
                rows.append(entry)
                excluded.append({"sample_id": sid, "split": sp, "reason": entry["reason"]})
                continue
            entry["status"] = "ok"
            entry["chosen_date"] = out["date"]
            entry["mask_px"] = out.get("mask_px")
            rows.append(entry)
            manifest.append(dict(out, split=sp, official_id=sid, official_split=sp,
                                 protocol_role={"train": "TRAIN", "val": "VAL", "test": "TEST"}[sp],
                                 inventory=rec["inventory"], pool_fraction=100))

    fields = ["official_id", "official_split", "inventory", "official_s2", "local_nc",
              "official_pixel_annotated", "exists", "status", "reason", "chosen_date", "mask_px"]
    with (OUT / "mapping.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (OUT / "exclusions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in excluded), encoding="utf-8")

    per_split = {sp: {"official": len(official[sp]),
                      "usable": sum(1 for m in manifest if m["split"] == sp),
                      "excluded": sum(1 for e in excluded if e["split"] == sp)}
                 for sp in SPLITS}
    lock["per_split"] = per_split

    # ---------- 3. 与现有角色的交集 ----------
    split_v4 = json.loads((ROOT / "manifests" / "split_v4.json").read_text(encoding="utf-8"))
    role_of = {e: role for role, evs in split_v4["design"].items() for e in evs}
    overlap = {"per_split_role": {}, "official_train_vs_current": {}}
    for sp in SPLITS:
        cnt = defaultdict(int)
        for m in manifest:
            if m["split"] == sp:
                cnt[role_of.get(m["event_id"], "unmapped")] += 1
        overlap["per_split_role"][sp] = dict(sorted(cnt.items()))
    for role in ("dev", "diagnostic", "final", "exclude_small", "train"):
        overlap["official_train_vs_current"][role] = sum(
            1 for m in manifest if m["split"] == "train" and role_of.get(m["event_id"]) == role)
    # 事件级清单
    ev_rows = []
    inv_counts = defaultdict(lambda: defaultdict(int))
    for m in manifest:
        inv_counts[m["event_id"]][m["split"]] += 1
    for ev in sorted(inv_counts, key=lambda e: -sum(inv_counts[e].values())):
        v = inv_counts[ev]
        ev_rows.append({"event": ev, "role": role_of.get(ev, "unmapped"),
                        **{sp: v.get(sp, 0) for sp in SPLITS},
                        "total": sum(v.values())})
    overlap["events"] = ev_rows
    (OUT / "overlap.json").write_text(json.dumps(overlap, indent=1, ensure_ascii=False), encoding="utf-8")

    L = ["# 官方 split 与现有事件角色的交集\n",
         f"来源 `{lock['source']['url']}` @ `{lock['source']['commit']}`\n",
         "## 1. 每个 split 的成员按现有角色分布\n",
         "| split | " + " | ".join(sorted({r for sp in overlap["per_split_role"] for r in overlap["per_split_role"][sp]})) + " |",
         "|---|" + "---:|" * len({r for sp in overlap["per_split_role"] for r in overlap["per_split_role"][sp]})]
    roles = sorted({r for sp in overlap["per_split_role"] for r in overlap["per_split_role"][sp]})
    for sp in SPLITS:
        d = overlap["per_split_role"][sp]
        L.append(f"| {sp} | " + " | ".join(str(d.get(r, 0)) for r in roles) + " |")
    L.append("\n## 2. 官方 TRAIN 落在现有角色上的图块数\n")
    for k, v in overlap["official_train_vs_current"].items():
        L.append(f"- 当前 **{k}**：{v}")
    L.append("\n## 3. 逐事件\n")
    L.append("| 事件 | 现有角色 | train | val | test | 合计 |")
    L.append("|---|---|---:|---:|---:|---:|")
    for r in ev_rows:
        L.append(f"| {r['event']} | {r['role']} | {r['train']} | {r['val']} | {r['test']} | {r['total']} |")
    (OUT / "overlap.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---------- 4. manifest 与统计 ----------
    train = [m for m in manifest if m["split"] == "train"]
    val = [m for m in manifest if m["split"] == "val"]
    test = [m for m in manifest if m["split"] == "test"]

    def write(recs, path):
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")

    write(train + val, OUT / "trainval.jsonl")
    write(test, OUT / "test.jsonl")

    stats_path = OUT / "stats_train.json"
    stats = compute_stats(train, stats_path)
    stats.update(data_version="official_single",
                 fit_split="official_TRAIN",
                 fit_sample_ids=[m["sample_id"] for m in train],
                 fit_events=dict(Counter(m["event_id"] for m in train)),
                 unit="DN, official harmonized product",
                 note="仅由官方 S12LS-LD TRAIN 的可用单期图块拟合；未使用官方 VAL/TEST 或任何旧统计")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------- 5. protocol ----------
    ident = {m["sample_id"]: {k: m[k] for k in ("event_id", "cache", "cache_sha256", "date", "time_index")}
             for m in manifest}
    members = {}
    for m in manifest:
        members[m["sample_id"]] = dict(ident[m["sample_id"]],
                                       role={"train": "train", "val": "val", "test": "test"}[m["split"]])
    payload = {
        "version": "BENCH_OFFICIAL_S12LS-LD_s2_single_date",
        "frozen": True,
        "design": {"train": sorted({m["event_id"] for m in train}),
                   "val": sorted({m["event_id"] for m in val}),
                   "test": sorted({m["event_id"] for m in test})},
        "members": members,
        "dev_ids": sorted(m["sample_id"] for m in val),
        "test_ids": sorted(m["sample_id"] for m in test),
        "pools": {"100": sorted(m["sample_id"] for m in train)},
        "files": {
            str((OUT / "trainval.jsonl")).replace("\\", "/"): file_sha256(OUT / "trainval.jsonl"),
            str((OUT / "test.jsonl")).replace("\\", "/"): file_sha256(OUT / "test.jsonl"),
            str(stats_path).replace("\\", "/"): file_sha256(stats_path),
        },
        "notes": {
            "official_source": lock["source"],
            "splits_sha256": lock["files"]["splits.json"]["sha256"],
            "single_date": "沿用项目固定灾后规则；官方是多时相+DEM，故本协议是单期变体",
            "stats": "仅官方 TRAIN",
            "reserved_events_in_official_train": overlap["official_train_vs_current"],
        },
    }
    payload["sha256"] = canonical_hash(payload)
    (OUT / "protocol.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    lock["protocol_sha256"] = payload["sha256"]
    lock["manifest_sha256"] = {
        "trainval": file_sha256(OUT / "trainval.jsonl"),
        "test": file_sha256(OUT / "test.jsonl"),
        "stats_train": file_sha256(stats_path),
    }
    lock["n_samples"] = {"train": len(train), "val": len(val), "test": len(test)}
    lock["temporal_fallback"] = {
        sp: sum(1 for m in manifest if m["split"] == sp and m.get("temporal_fallback"))
        for sp in SPLITS
    }
    lock["exclusion_reasons"] = dict(Counter(
        e.get("reason", "unknown").split(":")[0] for e in excluded))
    (OUT / "protocol_lock.json").write_text(json.dumps(lock, indent=1, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({k: lock[k] for k in ("per_split", "n_samples", "temporal_fallback",
                                          "exclusion_reasons", "protocol_sha256")},
                     indent=1, ensure_ascii=False))
    print(f"[saved] {OUT}/protocol_lock.json")


if __name__ == "__main__":
    main()
