"""M1 验收：在认证后的真实小批量上跑完整闭环并逐项断言。

检查项：
  A1 三个官方权重真实加载（记录身份哈希）
  A2 真实认证批次前向：12 波段 / 10 波段 / 输出形状
  A3 反向：允许训练的参数确实被更新，冻结参数不变
  A4 保存 → **全新实例**加载 → 推理结果一致（eval + 固定种子）
  A5 无 GT 泄漏：改变 region_mask 或用 None，logits 必须完全一致
  A6 空图输出：presence 低于阈值时必须输出空 mask
  A7 可视化：RGB / SCL / GT / 粗概率 / 细概率 / 二值
  A8 指标：无效像素被剔除（构造一个全无效样本验证）

用法：
  python scripts/accept_m1.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from landslide_seg.config import file_identity, load_config, manifest_identity, set_seed  # noqa: E402
from landslide_seg.data.certified import filter_records, make_batches, read_manifest  # noqa: E402
from landslide_seg.inference.visualize import save_panel  # noqa: E402
from landslide_seg.models.build import DEFAULT_WEIGHTS, build_from_config  # noqa: E402
from landslide_seg.training.losses import CombinedLoss  # noqa: E402
from landslide_seg.training.metrics import confusion, summarize  # noqa: E402
from landslide_seg.training.trainer import Trainer  # noqa: E402


def fixed_forward(model, batch, seed=1234):
    model.eval()
    torch.manual_seed(seed)
    random.seed(seed)
    with torch.no_grad():
        return model(batch.x12, batch.x10, batch.s2_dates, return_features=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config(args.config)
    info = set_seed(int(cfg["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = ROOT / "outputs"
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "visualizations").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    report: dict = {"device": device, "seed": info, "checks": {}}
    records = filter_records(read_manifest(cfg["data"]["manifest"]), split=args.split)
    if not records:
        print("[BLOCKED] 认证清单里没有可用的训练样本")
        return 2
    batch = make_batches(records[:4], batch_size=2, shuffle=False,
                         boa_offset=float(cfg["data"].get("boa_offset_dn", 0.0)))[0]

    # A1 权重身份
    report["weights"] = {
        "terramind": file_identity(ROOT / "weights" / "TerraMind_v1_small.pt"),
        "anysat": file_identity(DEFAULT_WEIGHTS["anysat"]),
        "remoteclip": file_identity(DEFAULT_WEIGHTS["remoteclip"]),
    }
    model = build_from_config(cfg, device=device)
    report["params"] = model.parameter_report()
    print("[A1] weights ok:", {k: v.get("size_bytes") for k, v in report["weights"].items()})

    # A2 前向
    model.eval()
    with torch.no_grad():
        o = model(
            batch.x12.to(device), batch.x10.to(device), batch.s2_dates.to(device),
            region_mask=batch.mask.unsqueeze(1).float().to(device),
            valid=batch.valid.to(device),
        )
    shapes = {
        "x12": list(batch.x12.shape), "x10": list(batch.x10.shape),
        "fine_logits": list(o.fine_logits.shape),
        "coarse_logits": list(o.coarse_logits.shape),
        "presence": list(o.presence_logits.shape),
        "sem_map": list(o.sem_map.shape),
        "region_embedding": list(o.region_embedding.shape) if o.region_embedding is not None else None,
        "pyramid": [list(p.shape) for p in o.features["pyramid"]],
    }
    report["checks"]["A2_shapes"] = shapes
    print("[A2] shapes", shapes["fine_logits"], shapes["region_embedding"])

    # A3 反向 + 参数更新
    trainer = Trainer(model, CombinedLoss(**{k: float(v) for k, v in cfg["loss"].items()}),
                      lr=float(cfg["optim"]["lr"]), device=device, out_dir=out)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    before = {n: p.detach().clone() for n, p in named[:10]}
    frozen_before = {n: p.detach().clone() for n, p in model.named_parameters()
                     if not p.requires_grad}
    frozen_sample = dict(list(frozen_before.items())[:10])
    losses = []
    for _ in range(args.steps):
        log = trainer.train_step(batch)
        losses.append(log["loss"])
    moved = sum(1 for n, p in named[:10] if not torch.allclose(before[n], p.detach(), atol=1e-12))
    frozen_moved = 0
    for n, v in frozen_sample.items():
        cur = dict(model.named_parameters())[n].detach()
        if not torch.allclose(v, cur, atol=1e-12):
            frozen_moved += 1
    report["checks"]["A3_train"] = {
        "losses": losses,
        "trainable_moved": f"{moved}/{len(before)}",
        "frozen_moved": f"{frozen_moved}/{len(frozen_sample)}",
    }
    print(f"[A3] trainable moved {moved}/{len(before)}, frozen moved {frozen_moved}/{len(frozen_sample)}")

    # A4 保存 → 新实例加载
    ckpt = trainer.save(out / "checkpoints" / "m1_accept.pt", extra={"accept": True})
    fresh = build_from_config(cfg, device=device)
    fresh_trainer = Trainer(fresh, CombinedLoss(**{k: float(v) for k, v in cfg["loss"].items()}),
                            device=device, out_dir=out)
    fresh_trainer.load(ckpt)
    a = fixed_forward(model, batch.to(device)).fine_logits
    b = fixed_forward(fresh, batch.to(device)).fine_logits
    diff = float((a - b).abs().max())
    report["checks"]["A4_reload"] = {"checkpoint": str(ckpt),
                                     "size_mb": round(ckpt.stat().st_size / 1e6, 1),
                                     "max_abs_diff": diff}
    print(f"[A4] reload max|Δ| = {diff:.3e}")

    # A5 无 GT 泄漏
    m1 = batch.mask.unsqueeze(1).float().to(device)
    v1 = batch.valid.to(device)
    with torch.no_grad():
        torch.manual_seed(7); random.seed(7)
        o_a = model(batch.x12.to(device), batch.x10.to(device), batch.s2_dates.to(device),
                    region_mask=m1, valid=v1, return_features=False)
        torch.manual_seed(7); random.seed(7)
        o_b = model(batch.x12.to(device), batch.x10.to(device), batch.s2_dates.to(device),
                    region_mask=1.0 - m1, valid=v1, return_features=False)
        torch.manual_seed(7); random.seed(7)
        o_c = model(batch.x12.to(device), batch.x10.to(device), batch.s2_dates.to(device),
                    region_mask=None, valid=None, return_features=False)
    leak_1 = float((o_a.fine_logits - o_b.fine_logits).abs().max())
    leak_2 = float((o_a.fine_logits - o_c.fine_logits).abs().max())
    report["checks"]["A5_no_gt_leak"] = {
        "flip_region_mask": leak_1, "drop_region_mask": leak_2,
        "pass": max(leak_1, leak_2) < 1e-6,
    }
    print(f"[A5] leak check flip={leak_1:.3e} drop={leak_2:.3e}")

    # A6 空图输出（功能验证 + 真实负样本说明）
    # 本数据集没有"已验证无滑坡"的图块（annotated=False 的图块既无掩膜也无事件日期，
    # 属未标注），因此只能做**功能级**验证：把阈值抬高使 presence 必然低于门限，
    # 输出必须为空。真实负样本上的拒检能力标注为 N/A，不能声称已验证。
    negs = [r for r in records if r.get("label_status") in ("verified-empty", "negative")]
    nb = batch.to(device)
    forced = fresh.predict(
        nb.x12, nb.x10, nb.s2_dates, threshold=0.5,
        empty_threshold=0.99, use_presence_gate=True,
    )
    gated_empty = float(forced["mask"].sum().item()) == 0.0
    ungated = fresh.predict(
        nb.x12, nb.x10, nb.s2_dates, threshold=0.5, use_presence_gate=False
    )
    report["checks"]["A6_empty"] = {
        "verified_negative_tiles": len(negs),
        "real_negative_validation": "N/A（数据集无已验证负样本）",
        "gate_forces_empty": gated_empty,
        "ungated_nonempty_ratio": float((ungated["mask"].sum(dim=(1, 2, 3)) > 0).float().mean()),
        "pass": gated_empty,
    }
    print(f"[A6] gate→empty={gated_empty}（真实负样本 {len(negs)} 个）")

    # A7 可视化
    vis = []
    with torch.no_grad():
        pr = fresh.predict(batch.x12.to(device), batch.x10.to(device), batch.s2_dates.to(device))
        oc = fresh(batch.x12.to(device), batch.x10.to(device), batch.s2_dates.to(device),
                   return_features=False)
    for i in range(min(2, batch.x12.shape[0])):
        x12 = batch.x12[i].cpu().numpy()
        rgb = np.stack([x12[3], x12[2], x12[1]], -1)
        lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
        rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
        p = save_panel(
            out / "visualizations" / f"m1_accept_{i}.png", rgb,
            batch.mask[i].cpu().numpy(),
            torch.sigmoid(oc.coarse_logits)[i, 0].cpu().numpy(),
            pr["prob"][i, 0].cpu().numpy(),
            pr["mask"][i, 0].cpu().numpy(),
            valid=batch.valid[i].cpu().numpy(),
            title=f"M1 ACCEPT {batch.sample_ids[i]}",
            presence=float(pr["presence"][i, 0]),
        )
        vis.append(str(p))
    report["checks"]["A7_vis"] = vis

    # A8 指标：全无效样本不应贡献任何计数
    gt = np.zeros((8, 8), dtype=np.int64)
    gt[0, 0] = 1
    pred = np.zeros((8, 8), dtype=np.uint8)
    valid = np.zeros((8, 8), dtype=bool)          # 全无效
    c_all_invalid = confusion(pred, gt, valid)
    c_normal = confusion(pred, gt, np.ones_like(valid))
    report["checks"]["A8_metrics"] = {
        "all_invalid": c_all_invalid, "normal": c_normal,
        "pass": sum(c_all_invalid.values()) == 0,
    }
    print(f"[A8] all-invalid counters {c_all_invalid}")

    report["all_passed"] = bool(
        report["checks"]["A3_train"]["frozen_moved"].startswith("0/")
        and diff < 1e-6
        and report["checks"]["A5_no_gt_leak"]["pass"]
        and report["checks"]["A6_empty"]["pass"]
        and report["checks"]["A8_metrics"]["pass"]
    )
    path = out / "logs" / "accept_m1.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[M1] all_passed = {report['all_passed']}")
    print(f"[saved] {path}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
