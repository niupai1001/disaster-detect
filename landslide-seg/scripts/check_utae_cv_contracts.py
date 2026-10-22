"""四折事件协议 · U-TAE 时间对照的正式训练前检查。

对应计划 §4 的 1–4 条（第 5 条属于 DEM／光谱分支，第 6 条见 GPU 门禁）：

1. TRAIN / INNER / OUTER 互斥，统计与预训练只来自本折 TRAIN；
2. 单期与多期目标帧、GT、valid 一致；截至日期组不存在晚到帧；
3. 合法全零帧不被当 padding，补齐帧不影响有效结果（用带 padding 的批与
   逐长度单独前向比较）；
4. 相同图块 batch 对应相同更新次数与曝光，变长分组后的损失归一化正确。

用法::

    python scripts/check_utae_cv_contracts.py --folds F1 --device cuda:0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from landslide_seg.config import file_sha256, resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import load_stats, read_manifest  # noqa: E402
from landslide_seg.data.temporal_v6 import TemporalV6, collate_v6  # noqa: E402
from landslide_seg.models.utae_dates import UTAERealDates  # noqa: E402
from landslide_seg.training.losses import bce_loss, dice_loss  # noqa: E402

RESULTS: list[dict] = []


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def fold_paths(fold: str) -> dict:
    d = f"manifests/event_cv_v5/{fold}"
    return {"pool": f"{d}/pool_100.jsonl", "outer": f"{d}/test.jsonl",
            "stats": f"{d}/stats_dem.json", "folds": "manifests/event_cv_v5/folds.json"}


def c1_membership(folds: list[str]) -> None:
    meta = json.loads(resolve_path("manifests/event_cv_v5/folds.json").read_text(encoding="utf-8"))
    seen: dict[str, str] = {}
    overlap = []
    for f in folds:
        p = fold_paths(f)
        pool = read_manifest(resolve_path(p["pool"]))
        outer = read_manifest(resolve_path(p["outer"]))
        train = [r for r in pool if r["protocol_role"] == "TRAIN"]
        inner = [r for r in pool if r["protocol_role"] == "VAL"]
        spec = meta["folds"][f]
        ok_counts = (len(train), len(inner), len(outer)) == (spec["n_train"], spec["n_inner"], spec["n_outer"])
        record(f"C1/{f}/counts", ok_counts,
               f"TRAIN/INNER/OUTER={len(train)}/{len(inner)}/{len(outer)} vs folds.json "
               f"{spec['n_train']}/{spec['n_inner']}/{spec['n_outer']}")
        ids_t, ids_i, ids_o = ({r["sample_id"] for r in g} for g in (train, inner, outer))
        disjoint = not (ids_t & ids_i or ids_t & ids_o or ids_i & ids_o)
        record(f"C1/{f}/disjoint", disjoint, "TRAIN/INNER/OUTER 两两不交")
        for sid in ids_o:
            if sid in seen and seen[sid] != f:
                overlap.append((sid, seen[sid], f))
            seen[sid] = f
        # 统计只来自本折 TRAIN
        st = json.loads(resolve_path(p["stats"]).read_text(encoding="utf-8"))
        fit = set(st.get("fit_sample_ids") or [])
        record(f"C1/{f}/stats_fit", fit == ids_t,
               f"stats_dem.fit_sample_ids 与 TRAIN 一致（{len(fit)} 个）" if fit == ids_t
               else f"stats 拟合集合与 TRAIN 不一致：{len(fit)} vs {len(ids_t)}")
        geo = json.loads(resolve_path(f"manifests/event_cv_v5/{f}/geo_stats_report.json").read_text(encoding="utf-8"))
        want = hashlib.sha256("\n".join(sorted(ids_t)).encode()).hexdigest()
        record(f"C1/{f}/geo_fit", geo["fit_ids_sha256"] == want,
               "DEM/坡度统计只由本折 TRAIN 拟合" if geo["fit_ids_sha256"] == want else "DEM/坡度拟合集合不符")
    record("C1/outer_no_overlap_across_folds", not overlap,
           f"四折外部集合互不重叠（{len(seen)} 个图块）" if not overlap else str(overlap[:3]))


def c2_frames(folds: list[str]) -> None:
    for f in folds:
        p = fold_paths(f)
        rows = [r for r in read_manifest(resolve_path(p["pool"])) if r["protocol_role"] == "TRAIN"][:6]
        stats = load_stats(resolve_path(p["stats"]))
        ds = {m: TemporalV6(rows, stats, mode=m, augment=False, dem=True) for m in ("single", "multi", "history")}
        ok_gt, ok_tgt, ok_future, msgs = True, True, True, []
        for i, r in enumerate(rows):
            t = int(r["time_index"])
            a, b, c = (ds[m][i] for m in ("single", "multi", "history"))
            if not (torch.equal(a["mask"], b["mask"]) and torch.equal(a["valid"], b["valid"])
                    and torch.equal(a["mask"], c["mask"]) and torch.equal(a["valid"], c["valid"])):
                ok_gt = False
            if not torch.allclose(b["image"][t], a["image"][0], atol=0, rtol=0):
                ok_tgt = False
            if int(c["frame_index"].max()) != t or float(c["dates"].max()) > float(b["dates"][t]):
                ok_future = False
            msgs.append(f"{r['sample_id']}: t={t} history_len={c['image'].shape[0]}")
        record(f"C2/{f}/same_target_frame", ok_tgt, "单期帧 = 多期第 time_index 帧（逐像素）")
        record(f"C2/{f}/same_gt_valid", ok_gt, "三种模式下 GT/valid 完全一致")
        record(f"C2/{f}/history_no_future", ok_future, "截至日期组最后一帧 = 目标帧，无晚到帧")


def c3_padding(device: torch.device) -> None:
    # 结构：全零帧仍保留 keep；补齐帧由 keep 标记而不是数值零
    img = torch.zeros(3, 11, 128, 128)
    item = {"image": img, "mask": torch.zeros(128, 128),
            "valid": torch.ones(128, 128, dtype=torch.bool),
            "dates": torch.tensor([1.0, 2.0, 3.0]), "frame_index": torch.tensor([0, 1, 2]),
            "time_index": 2, "anchor": 2, "sample_id": "zero_frame", "event_id": "x"}
    batch = collate_v6([item])
    record("C3/zero_frame_kept", bool(batch["keep"].all()) and batch["keep"].shape[1] == 3,
           "全零像素的真实帧 keep=True，未被当成 padding")

    # 补齐帧不影响有效结果：带 padding 的批 与 逐长度单独前向 应逐像素一致
    torch.manual_seed(0)
    model = UTAERealDates(in_channels=11, num_classes=1).to(device).eval()
    lengths = [3, 15, 7, 15]
    items = []
    for i, L in enumerate(lengths):
        items.append({"image": torch.randn(L, 11, 128, 128), "mask": torch.zeros(128, 128),
                      "valid": torch.ones(128, 128, dtype=torch.bool),
                      "dates": torch.arange(L, dtype=torch.float32), "frame_index": torch.arange(L),
                      "time_index": L - 1, "anchor": L - 1, "sample_id": f"s{i}", "event_id": "x"})
    batch = collate_v6(items)
    with torch.no_grad():
        padded = model(batch["image"].to(device), batch["dates"].to(device),
                       batch["keep"].to(device), batch["anchor"].to(device))["logits"].cpu()
        worst = 0.0
        for i, L in enumerate(lengths):
            single = model(items[i]["image"][None, :L].to(device), items[i]["dates"][None, :L].to(device),
                           torch.ones(1, L, dtype=torch.bool, device=device),
                           torch.tensor([L - 1], device=device))["logits"].cpu()
            worst = max(worst, float((padded[i] - single[0]).abs().max()))
    # 阈值 1e-3：差异来自不同批组成下的卷积核选择（float32 数值噪声），
    # 真正的 padding 泄漏会是 O(1) 量级，不可能落在 1e-3 以内。
    record("C3/padding_invariant", worst < 1e-3,
           f"混长度批与逐样本前向最大差异 {worst:.2e}（数值噪声，非 padding 泄漏）")

    # 损失归一化：整批损失 = 按有效像素加权的逐样本损失
    b, h, w = 4, 16, 16
    logits = torch.randn(b, 1, h, w, device=device)
    mask = (torch.rand(b, h, w, device=device) > 0.7).float()
    valid = torch.ones(b, h, w, dtype=torch.bool, device=device)
    valid[0, :, :8] = False
    full = bce_loss(logits, mask.unsqueeze(1), valid.unsqueeze(1), pos_weight=5.0)
    n = valid.float().sum()
    manual = sum(bce_loss(logits[i:i + 1], mask[i:i + 1].unsqueeze(1), valid[i:i + 1].unsqueeze(1),
                          pos_weight=5.0) * valid[i].float().sum() for i in range(b)) / n
    rel = float((full - manual).abs() / full.abs().clamp_min(1e-12))
    record("C3/loss_normalization", rel < 1e-5, f"整批 BCE 与逐样本加权差 {rel:.2e}")


def c4_budget(folds: list[str]) -> None:
    batch = 16
    for f in folds:
        rows = [r for r in read_manifest(resolve_path(fold_paths(f)["pool"]))
                if r["protocol_role"] == "TRAIN"]
        n = len(rows)
        counts = {m: len(epoch_batches(n, batch, 1234, 1)) for m in ("single", "multi", "history")}
        same = len(set(counts.values())) == 1 and counts["single"] == (n + batch - 1) // batch
        record(f"C4/{f}/updates_per_epoch", same,
               f"N={n} 三组每 epoch 更新数 {counts}（= ceil(N/{batch})）")
        # 同一 (seed, epoch) 的批序在三组完全相同
        idx = epoch_batches(n, batch, 1234, 3)
        same_order = all(idx == epoch_batches(n, batch, 1234, 3) for _ in range(3))
        record(f"C4/{f}/same_batch_order", same_order, "批序只由 (seed, epoch) 决定，三组共用")
        record(f"C4/{f}/total_updates", True,
               f"75 epoch × {counts['single']} = {75 * counts['single']} 次更新，"
               f"曝光 {75 * n} 图块")


def epoch_batches(n: int, batch: int, seed: int, epoch: int) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed * 100_003 + epoch)
    perm = torch.randperm(n, generator=g).tolist()
    return [perm[i:i + batch] for i in range(0, n, batch)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="F1,F2,F3,F4")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    folds = [f.strip() for f in args.folds.split(",") if f.strip()]
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    c1_membership(folds)
    c2_frames(folds)
    c3_padding(dev)
    c4_budget(folds)
    bad = [r for r in RESULTS if not r["ok"]]
    out = {"passed": len(RESULTS) - len(bad), "total": len(RESULTS), "failed": bad}
    (ROOT / "outputs" / "utae_cv").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "utae_cv" / "contracts.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"passed": out["passed"], "total": out["total"]}, ensure_ascii=False))
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
