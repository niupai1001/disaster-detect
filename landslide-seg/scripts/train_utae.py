"""U-TAE 时序配对训练（单期 vs 15 期），按官方设置。

固定条件（执行计划 §3 C）：
- 同一 3012/758/945 可用子集、同源标签、同一套 valid 评价范围；
- 两组共用单期 TRAIN + DEM 拟合的通道统计（同一 normalizer）；
- 官方 U-TAE 结构 + AdamW lr 1e-3、weight decay 0.01、正类权重 5、BCE/Dice 各 0.5；
- 75 个完整 epoch、batch 16、FP32、梯度裁剪 1.0、不提前结束；
- 每 epoch 按 VAL 有效像素 pooled IoU@0.5 选 best（同分取较早），best/last 都评测。

用法::

    python scripts/train_utae.py --mode single --device cuda:0
    python scripts/train_utae.py --mode multi  --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from landslide_seg.config import resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import load_stats, read_manifest  # noqa: E402
from landslide_seg.data.temporal_dataset import TemporalPairDataset, collate  # noqa: E402
from landslide_seg.models.utae_seg import UTAESeg  # noqa: E402
from landslide_seg.training.losses import bce_loss, dice_loss  # noqa: E402
from landslide_seg.training.metrics import confusion, summarize  # noqa: E402

EPOCHS = 75
BATCH = 16
LR, WD, POS_W = 1e-3, 0.01, 5.0
BCE_W, DICE_W = 0.5, 0.5
CLIP = 1.0
THRESH = 0.5
SEED = 1234
STATS = "manifests/official_single_geo/stats_dem.json"
TRAINVAL = "manifests/official_single/trainval.jsonl"
TEST = "manifests/official_single/test.jsonl"
OUT_ROOT = "outputs/utae_pair"


def set_seed(seed: int = SEED) -> None:
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def loss_fn(logits: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if mask.ndim == logits.ndim - 1:
        mask, valid = mask.unsqueeze(1), valid.unsqueeze(1)
    b = bce_loss(logits, mask, valid, pos_weight=POS_W)
    d = dice_loss(logits, mask, valid)
    return BCE_W * b + DICE_W * d


@torch.no_grad()
def evaluate(model: UTAESeg, loader: DataLoader, device) -> dict:
    model.eval()
    counts, by_event = [], {}
    for batch in loader:
        logits = model(batch["image"].to(device), batch["dates"].to(device))["logits"].float()
        prob = logits[:, 0].sigmoid().cpu().numpy()
        gt = batch["mask"].numpy().astype(bool)
        va = batch["valid"].numpy().astype(bool)
        for i, ev in enumerate(batch["event_id"]):
            c = confusion(prob[i] >= THRESH, gt[i], va[i])
            counts.append(c)
            by_event.setdefault(ev, []).append(c)
    pooled = summarize(counts)
    per_event = {ev: summarize(cs)["iou"] for ev, cs in sorted(by_event.items())}
    vals = [v for v in per_event.values() if v is not None]
    return {"pooled_iou": pooled["iou"], "dice": pooled["dice"], "precision": pooled["precision"],
            "recall": pooled["recall"], "event_mean_iou": (float(np.mean(vals)) if vals else None),
            "per_event": per_event}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("single", "multi"), required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    rows = read_manifest(resolve_path(TRAINVAL))
    test_rows = read_manifest(resolve_path(TEST))
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    stats = load_stats(resolve_path(STATS))
    run_dir = ROOT / OUT_ROOT / f"utae_{args.mode}" / "1234"
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed()
    model = UTAESeg(in_channels=11, num_classes=1).to(dev)
    # 两组必须从完全相同的初始状态出发：digest 写入 run_meta 供验收比对
    init_digest = model.digest()
    meta = {"mode": args.mode, "seed": SEED, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
            "weight_decay": WD, "pos_weight": POS_W, "bce": BCE_W, "dice": DICE_W,
            "grad_clip": CLIP, "threshold": THRESH, "stats": STATS,
            "n_train": len(train), "n_val": len(val), "n_test": len(test_rows),
            "params": model.count_params(), "init_digest": init_digest,
            "input": "single" if args.mode == "single" else "15 timesteps",
            "selection": "VAL 有效像素 pooled IoU@0.5，同分取较早 epoch"}
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")

    tr_ds = TemporalPairDataset(train, stats, mode=args.mode, augment=True, seed=SEED)
    va_ds = TemporalPairDataset(val, stats, mode=args.mode, augment=False, seed=SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    history, best_iou, best_epoch = [], -1.0, -1
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        tr_ds.set_epoch(epoch)
        loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=0,
                            collate_fn=collate, drop_last=False)
        model.train()
        tot, cnt, skipped = 0.0, 0, 0
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            logits = model(batch["image"].to(dev), batch["dates"].to(dev))["logits"]
            loss = loss_fn(logits, batch["mask"].to(dev), batch["valid"].to(dev))
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP, error_if_nonfinite=False)
            opt.step()
            tot += float(loss.detach()); cnt += 1

        va_loader = DataLoader(va_ds, batch_size=BATCH, shuffle=False, num_workers=0, collate_fn=collate)
        ev = evaluate(model, va_loader, dev)
        improved = (ev["pooled_iou"] or -1) > best_iou
        if improved:
            best_iou, best_epoch = ev["pooled_iou"], epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": ev, "meta": meta},
                       run_dir / "best.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch, "val": ev, "meta": meta},
                   run_dir / "last.pt")
        row = {"epoch": epoch, "train_loss": tot / max(cnt, 1), "skipped": skipped,
               "val": ev, "seconds": round(time.time() - t0, 1)}
        history.append(row)
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{args.mode}] epoch {epoch:3d} loss {tot/max(cnt,1):.4f} "
              f"VAL pooled {ev['pooled_iou']:.4f} event_mean {ev['event_mean_iou']} best {best_epoch}", flush=True)

    summary = {"mode": args.mode, "best_epoch": best_epoch, "best_val_pooled_iou": best_iou,
               "epochs": EPOCHS, "seconds": round(time.time() - t0, 1), "meta": meta,
               "history": history}
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {run_dir}  best epoch {best_epoch} pooled {best_iou:.4f}")


if __name__ == "__main__":
    main()
