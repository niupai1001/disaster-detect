"""P2 补充门槛：空标注、全无效 batch、优化器/调度器状态恢复、旧 A1 漂移。

- 空标注（GT 全 0、valid 正常）：损失有限、可反传 → Dice 定义不产生 NaN；
- 全无效 batch：`_train_epoch` 必须跳过且不计 optimizer 更新；
- 恢复：保存/加载 optimizer+scheduler state 后，LR 与 last_epoch 连续；
- 旧 A1：用当前代码重跑 val，IoU 必须等于历史 0.3068（无意外漂移）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch  # noqa: E402

from landslide_seg.config import load_segformer_config, resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import (  # noqa: E402
    S2SegDataset,
    collate,
    filter_split,
    load_stats,
    read_manifest,
)
from landslide_seg.training.segformer_metrics import evaluate_tile_v2, summarize_tiles_v2  # noqa: E402
from landslide_seg.training.segformer_trainer import (  # noqa: E402
    _loss,
    _make_optimizer,
    _train_epoch,
    _make_scheduler,
    load_checkpoint,
)


def main() -> None:
    cfg_path = ROOT / "configs" / "segformer" / "coarse_to_fine" / "cf1.yaml"
    full = load_segformer_config(cfg_path)
    cfg = full["segformer"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, meta = load_checkpoint(
        resolve_path(cfg["output_dir"]) / cfg["data"]["version"] / "CF1" / "1234" / "best.pt",
        device=str(device),
    )
    model.train()
    stats = load_stats(resolve_path(cfg["data"]["stats"]))
    records = filter_split(read_manifest(resolve_path(cfg["data"]["manifest"])), "train")[:8]
    ds = S2SegDataset(records, stats, preload=True)

    report = {}

    # 1) 空标注 batch
    batch = collate([ds[i] for i in range(4)])
    image, mask, valid = batch["image"].to(device), batch["mask"].to(device), batch["valid"].to(device)
    empty = torch.zeros_like(mask)
    out = model(image)
    loss, parts = _loss(full, out["logits"], empty, valid, out["base_logits"])
    loss.backward()
    report["empty_gt"] = {
        "loss": round(float(loss.detach()), 5),
        "fine": round(float(parts["fine"].detach()), 5),
        "coarse": None if parts["coarse"] is None else round(float(parts["coarse"].detach()), 5),
        "finite": bool(torch.isfinite(loss)),
        "grad_finite": bool(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)),
    }
    model.zero_grad(set_to_none=True)

    # 2) 全无效 batch + 正常 batch → 只计 1 步
    bad = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    bad["valid"] = torch.zeros_like(valid)
    opt = _make_optimizer(model, full)
    sched = _make_scheduler(opt, full, total_steps=3600)
    mean_loss, skipped, steps, coarse = _train_epoch(
        model, [bad, batch], opt, sched, full, device
    )
    report["all_invalid_batch"] = {
        "skipped": skipped,
        "optimizer_steps": steps,
        "loss_finite": bool(torch.isfinite(torch.tensor(mean_loss))),
        "coarse_loss": None if coarse == 0 else round(float(coarse), 5),
    }

    # 3) 优化器/调度器状态恢复
    for _ in range(5):
        opt.step()
        sched.step()
    lr_before = opt.param_groups[0]["lr"]
    epoch_before = sched.last_epoch
    state = {"opt": opt.state_dict(), "sched": sched.state_dict()}
    opt2 = _make_optimizer(model, full)
    sched2 = _make_scheduler(opt2, full, total_steps=3600)
    opt2.load_state_dict(state["opt"])
    sched2.load_state_dict(state["sched"])
    report["resume"] = {
        "lr_before": lr_before,
        "lr_after": opt2.param_groups[0]["lr"],
        "epoch_before": epoch_before,
        "epoch_after": sched2.last_epoch,
        "ok": abs(lr_before - opt2.param_groups[0]["lr"]) < 1e-12
        and epoch_before == sched2.last_epoch,
    }

    # 4) 旧 A1 漂移：当前代码重跑 val
    a1, _ = load_checkpoint(
        resolve_path(cfg["output_dir"]) / cfg["data"]["version"] / "A1" / "1234" / "best.pt",
        device=str(device),
    )
    a1.eval()
    val_records = filter_split(read_manifest(resolve_path(cfg["data"]["manifest"])), "val")
    ds_val = S2SegDataset(val_records, stats, preload=True)
    tiles = []
    with torch.no_grad():
        for i in range(0, len(ds_val), 32):
            b = collate([ds_val[j] for j in range(i, min(i + 32, len(ds_val)))])
            o = a1(b["image"].to(device))
            prob = torch.sigmoid(o["logits"].float()[:, 0]).cpu().numpy()
            for k, sid in enumerate(b["sample_id"]):
                tiles.append(
                    evaluate_tile_v2(
                        prob[k], b["mask"][k].numpy().astype(bool), b["valid"][k].numpy().astype(bool),
                        threshold=0.5, iou_threshold=0.25, iou_strict=0.5, boundary_tol=1,
                        sample_id=sid, event_id=b["event_id"][k],
                    )
                )
    iou = summarize_tiles_v2(tiles)["pixel"]["iou"]
    report["a1_drift"] = {
        "iou_now": round(float(iou), 4),
        "iou_history": 0.3068,
        "ok": abs(float(iou) - 0.3068) < 5e-4,
    }

    out_dir = resolve_path("outputs/segformer/coarse_to_fine")
    (out_dir / "extra_gates.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"[saved] {out_dir / 'extra_gates.json'}")


if __name__ == "__main__":
    main()
