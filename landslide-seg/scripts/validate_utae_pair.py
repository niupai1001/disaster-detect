"""U-TAE 时序配对的验收门槛（执行计划 §4 第 5、6 条）。

检查项：
1. 两组初始参数逐位一致（同 seed 构建的 state_dict 哈希相同）；
2. 单期组的输入帧 == 15 期组中 ``time_index`` 那一帧（逐像素）；
3. memmap 里那帧 == 原单期缓存 npz 的 x10（确认缓存没有错位）；
4. 两组 mask / valid 逐像素一致（同一评价范围）；
5. 真实前反向无 NaN/Inf；少量样本能拟合；checkpoint 重载输出一致；
6. 有效像素 pooled IoU 与手算一致。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402
import torch  # noqa: E402

from landslide_seg.config import resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import load_stats, read_manifest  # noqa: E402
from landslide_seg.data.temporal_dataset import TemporalPairDataset, collate  # noqa: E402
from landslide_seg.models.utae_seg import UTAESeg  # noqa: E402
from landslide_seg.training.metrics import confusion, summarize  # noqa: E402

STATS = "manifests/official_single_geo/stats_dem.json"
TRAINVAL = "manifests/official_single/trainval.jsonl"
OUT = ROOT / "reports" / "event_cv_v5" / "utae_validation.json"


def main() -> None:
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows = [r for r in read_manifest(resolve_path(TRAINVAL)) if r["split"] == "train"][:8]
    stats = load_stats(resolve_path(STATS))
    gates = []

    def gate(name, ok, detail, hard=True):
        gates.append({"check": name, "ok": bool(ok), "hard": hard, "detail": detail})
        print(f"[{'PASS' if ok else ('FAIL' if hard else 'WARN')}] {name}: {detail}")

    ds_s = TemporalPairDataset(rows, stats, mode="single", augment=False)
    ds_m = TemporalPairDataset(rows, stats, mode="multi", augment=False)

    # 2/3/4：帧一致性与评价范围
    max_frame, max_npz, mask_ok, valid_ok = 0.0, 0.0, True, True
    for i, r in enumerate(rows):
        a, b = ds_s[i], ds_m[i]
        t = int(r["time_index"])
        max_frame = max(max_frame, float((a["image"][0] - b["image"][t]).abs().max()))
        mask_ok &= bool(torch.equal(a["mask"], b["mask"]))
        valid_ok &= bool(torch.equal(a["valid"], b["valid"]))
        # memmap 原始帧 vs 原 npz（两者都是未归一化的 DN）
        row = ds_s.pos[r["sample_id"]]
        raw_mm = np.asarray(ds_s.X[row, t], dtype=np.int32)
        raw_npz = np.load(resolve_path(r["cache"]))["x10"].astype(np.int32)
        max_npz = max(max_npz, float(np.abs(raw_mm - raw_npz).max()))
    gate("单期帧 == 15 期中对应帧", max_frame == 0.0, {"max_abs_diff": max_frame})
    gate("memmap 帧 == 原单期缓存 x10", max_npz == 0.0, {"max_abs_diff": max_npz})
    gate("两组 mask 逐像素一致", mask_ok, {"n": len(rows)})
    gate("两组 valid 逐像素一致", valid_ok, {"n": len(rows)})
    gate("15 期组确有 15 个时相", ds_m[0]["image"].shape[0] == 15, list(ds_m[0]["image"].shape))

    # 1：两组初始参数逐位一致
    torch.manual_seed(1234)
    m1 = UTAESeg(in_channels=11, num_classes=1)
    torch.manual_seed(1234)
    m2 = UTAESeg(in_channels=11, num_classes=1)
    same_init = m1.digest() == m2.digest()
    gate("两组初始参数逐位一致", same_init, {"digest": m1.digest()[:16]})

    # 5：前反向 + 过拟合小批 + 重载
    model = m1.to(dev)
    batch = collate([ds_m[i] for i in range(4)])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    losses = []
    model.train()
    for _ in range(12):
        opt.zero_grad(set_to_none=True)
        logits = model(batch["image"].to(dev), batch["dates"].to(dev))["logits"]
        from landslide_seg.training.losses import bce_loss, dice_loss
        mk = batch["mask"].to(dev).unsqueeze(1)
        va = batch["valid"].to(dev).unsqueeze(1)
        loss = 0.5 * bce_loss(logits, mk, va, pos_weight=5.0) + 0.5 * dice_loss(logits, mk, va)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    finite = all(np.isfinite(losses))
    gate("前反向无 NaN/Inf", finite, {"loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4)})
    gate("小批能拟合（loss 下降）", losses[-1] < losses[0], {"first": round(losses[0], 4), "last": round(losses[-1], 4)})

    p = ROOT / "outputs" / "utae_pair" / "_validation.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, p)
    m3 = UTAESeg(in_channels=11, num_classes=1).to(dev)
    m3.load_state_dict(torch.load(p, map_location=dev)["model"])
    with torch.no_grad():
        model.eval(); m3.eval()
        a = model(batch["image"].to(dev), batch["dates"].to(dev))["logits"]
        b = m3(batch["image"].to(dev), batch["dates"].to(dev))["logits"]
    d = float((a - b).abs().max())
    gate("checkpoint 重载输出一致", d == 0.0, {"max_abs_diff": d})

    # 6：有效像素指标手算
    pred = np.array([[True, True], [False, False]])
    gt = np.array([[True, False], [True, False]])
    va = np.ones((2, 2), dtype=bool)
    iou = summarize([confusion(pred, gt, va)])["iou"]
    gate("pooled IoU 与手算一致（1/3）", abs(iou - 1 / 3) < 1e-12, {"iou": iou})

    rep = {"gates": gates, "all_pass": all(g["ok"] for g in gates if g["hard"]),
           "params": model.count_params()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[done] all_pass={rep['all_pass']} -> {OUT}")
    if not rep["all_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
