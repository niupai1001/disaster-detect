"""U-TAE 真实日期配对训练（官方组件日期编码启用版）。

与历史 ``train_utae.py`` 的区别只有计划要求的三点：

1. 输入日期来自 ``data/s2_temporal_v5/index.json`` 的真实观测日期（绝对天数，
   基准 2015-12-03），不再用 ``arange(T)`` 帧序号；
2. 启用 vendored LTAE 已有的 ``PositionalEncoder``（原文件保持不变，适配器
   ``UTAERealDates`` 按其 forward 语义补回被注释掉的位置编码相加）；
3. 缺失按 v6 侧表的显式规则填 TRAIN 均值，云/阴影/雪/未知 SCL 只记录。

其余条件与历史配对一致：3012/758/945、十波段 + DEM、75 epoch、batch16、
FP32、AdamW lr 1e-3 / wd 0.01、pos_weight 5、0.5 BCE + 0.5 Dice、seed1234。
两组（单期 / 15 期）从同一初始化摘要出发，样本顺序与增强一致。

用法::

    python scripts/train_utae_v6.py --mode single --device cuda:0
    python scripts/train_utae_v6.py --mode multi  --device cuda:1
"""

from __future__ import annotations

import argparse
import json
import random
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

from landslide_seg.config import file_sha256, resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import filter_split, load_stats, read_manifest  # noqa: E402
from landslide_seg.data.temporal_v6 import TemporalV6, collate_v6  # noqa: E402
from landslide_seg.models.utae_dates import UTAERealDates  # noqa: E402
from landslide_seg.training.losses import bce_loss, dice_loss  # noqa: E402
from landslide_seg.training.metrics import confusion, summarize  # noqa: E402

EPOCHS, BATCH = 75, 16
LR, WD, POS_W, CLIP, THRESH, SEED = 1e-3, 0.01, 5.0, 1.0, 0.5, 1234
STATS = "manifests/official_single_geo/stats_dem.json"
TRAINVAL = "manifests/official_single/trainval.jsonl"
TEST = "manifests/official_single/test.jsonl"
OUT_ROOT = "outputs/utae_v6"


def set_seed(seed: int = SEED) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {"seed": seed, "deterministic": True}


def loss_fn(logits, mask, valid):
    if mask.ndim == logits.ndim - 1:
        mask, valid = mask.unsqueeze(1), valid.unsqueeze(1)
    return 0.5 * bce_loss(logits, mask, valid, pos_weight=POS_W) + 0.5 * dice_loss(logits, mask, valid)


class LengthBatchSampler:
    """按真实序列长度分桶，避免 padding 影响 GroupNorm；每 epoch 组内/组间重排。"""

    def __init__(self, rows, batch_size: int, seed: int, epoch: int, shuffle: bool):
        groups: dict[int, list[int]] = {}
        for i, r in enumerate(rows):
            groups.setdefault(int(r["time_index"]) + 1, []).append(i)
        gen = torch.Generator().manual_seed(seed * 100_003 + epoch)
        order = sorted(groups)
        if shuffle:
            order = [order[i] for i in torch.randperm(len(order), generator=gen).tolist()]
        batches = []
        for length in order:
            idx = groups[length]
            if shuffle:
                idx = [idx[i] for i in torch.randperm(len(idx), generator=gen).tolist()]
            for start in range(0, len(idx), batch_size):
                batches.append(idx[start:start + batch_size])
        if shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches), generator=gen).tolist()]
        self.batches = batches
        self.epoch = epoch

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


@torch.no_grad()
def evaluate(model, ds, device, mode: str) -> dict:
    model.eval()
    if mode == "history":
        loader = DataLoader(ds, batch_sampler=LengthBatchSampler(ds.rows, BATCH, SEED, 0, False),
                            num_workers=0, collate_fn=collate_v6)
    else:
        loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0, collate_fn=collate_v6)
    counts, by_event = [], {}
    for batch in loader:
        logits = model(batch["image"].to(device), batch["dates"].to(device),
                       batch["keep"].to(device), batch["anchor"].to(device))["logits"].float()
        prob = logits[:, 0].sigmoid().cpu().numpy()
        gt = batch["mask"].numpy().astype(bool)
        va = batch["valid"].numpy().astype(bool)
        for i, ev in enumerate(batch["event_id"]):
            c = confusion(prob[i] >= THRESH, gt[i], va[i])
            counts.append(c)
            by_event.setdefault(ev, []).append(c)
    pooled = summarize(counts)
    per_event = {ev: summarize(cs)["iou"] for ev, cs in sorted(by_event.items())}
    vals = [float(v) for v in per_event.values() if v is not None and np.isfinite(v)]
    return {"pooled_iou": float(pooled["iou"]), "dice": float(pooled["dice"]),
            "precision": float(pooled["precision"]), "recall": float(pooled["recall"]),
            "event_mean_iou": float(np.mean(vals)) if vals else None,
            "per_event": per_event}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("single", "multi", "history"), required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=None)
    args = ap.parse_args()
    dev = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    rows = read_manifest(resolve_path(TRAINVAL))
    train = filter_split(rows, "train")
    val = filter_split(rows, "val")
    test_rows = read_manifest(resolve_path(TEST))
    stats = load_stats(resolve_path(STATS))
    run_dir = ROOT / OUT_ROOT / f"utae_{args.mode}_realdates" / str(SEED)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_info = set_seed(SEED)

    model = UTAERealDates(in_channels=11, num_classes=1).to(dev)
    init_digest = model.digest()
    meta = {
        "mode": args.mode, "seed": SEED, "seed_info": seed_info, "epochs": EPOCHS,
        "batch": BATCH, "lr": LR, "weight_decay": WD, "pos_weight": POS_W,
        "bce": 0.5, "dice": 0.5, "grad_clip": CLIP, "threshold": THRESH,
        "stats": STATS, "stats_sha256": file_sha256(resolve_path(STATS)),
        "trainval_sha256": file_sha256(resolve_path(TRAINVAL)),
        "test_sha256": file_sha256(resolve_path(TEST)),
        "temporal_index_sha256": file_sha256(resolve_path("data/s2_temporal_v5/index.json")),
        "quality_complete_sha256": file_sha256(resolve_path("data/s2_temporal_v6_quality/complete.json")),
        "n_train": len(train), "n_val": len(val), "n_test": len(test_rows),
        "params": model.count_params(), "init_digest": init_digest,
        "dates": {"source": "data/s2_temporal_v5/index.json", "reference": "2015-12-03",
                  "unit": "days", "single": "time_index 的真实日期",
                  "multi": "15 帧真实日期", "encoder": "LTAE PositionalEncoder 启用"},
        "missing": "SCL=0 / 非有限 / 显式 FillValue → TRAIN 均值；云与 SCL255 仅记录",
        "selection": "VAL 有效像素 pooled IoU@0.5，同分取较早 epoch",
        "implementation": "官方 vendored UTAE + 独立适配器启用日期位置编码（非未经修改的官方复现）",
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False),
                                           encoding="utf-8")
    tr_ds = TemporalV6(train, stats, mode=args.mode, augment=True, seed=SEED, dem=True, preload=True)
    va_ds = TemporalV6(val, stats, mode=args.mode, augment=False, seed=SEED, dem=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    start_epoch, best_iou, best_epoch, history = 1, -1.0, -1, []
    if args.resume and (run_dir / "last.pt").is_file():
        ckpt = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        if ckpt.get("init_digest") != init_digest:
            raise ValueError("恢复时初始化摘要不一致")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_iou, best_epoch = float(ckpt["best_iou"]), int(ckpt["best_epoch"])
        history = ckpt["history"]
        print(f"[resume] epoch {start_epoch} best {best_iou:.4f}@{best_epoch}", flush=True)

    t0 = time.time()
    last_epoch = args.max_epochs or EPOCHS
    for epoch in range(start_epoch, last_epoch + 1):
        tr_ds.set_epoch(epoch)
        gen = torch.Generator().manual_seed(SEED * 100_003 + epoch)
        if args.mode == "history":
            loader = DataLoader(tr_ds, batch_sampler=LengthBatchSampler(train, BATCH, SEED, epoch, True),
                                num_workers=0, collate_fn=collate_v6)
        else:
            loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=0,
                                collate_fn=collate_v6, generator=gen, drop_last=False)
        model.train()
        tot, cnt, skipped = 0.0, 0, 0
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            logits = model(batch["image"].to(dev), batch["dates"].to(dev),
                           batch["keep"].to(dev), batch["anchor"].to(dev))["logits"]
            loss = loss_fn(logits, batch["mask"].to(dev), batch["valid"].to(dev))
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP, error_if_nonfinite=False)
            opt.step()
            tot += float(loss.detach()); cnt += 1
        ev = evaluate(model, va_ds, dev, args.mode)
        improved = (ev["pooled_iou"] or -1) > best_iou
        if improved:
            best_iou, best_epoch = ev["pooled_iou"], epoch
        payload = {"model": model.state_dict(), "optimizer": opt.state_dict(),
                   "epoch": epoch, "val": ev, "best_iou": best_iou, "best_epoch": best_epoch,
                   "history": history, "meta": meta, "init_digest": init_digest}
        torch.save(payload, run_dir / "last.pt")
        if improved:
            torch.save(payload, run_dir / "best.pt")
        row = {"epoch": epoch, "train_loss": tot / max(cnt, 1), "skipped": skipped,
               "val": ev, "seconds": round(time.time() - t0, 1)}
        history.append(row)
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{args.mode}] epoch {epoch:3d} loss {tot/max(cnt,1):.4f} "
              f"VAL pooled {ev['pooled_iou']:.4f} event_mean {ev['event_mean_iou']} "
              f"best {best_epoch}", flush=True)

    summary = {"mode": args.mode, "best_epoch": best_epoch, "best_val_pooled_iou": best_iou,
               "epochs": last_epoch, "seconds": round(time.time() - t0, 1), "meta": meta,
               "history": history, "complete": last_epoch == EPOCHS,
               "peak_alloc_mib": torch.cuda.max_memory_allocated(dev) / 2**20 if dev.type == "cuda" else None}
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                                                encoding="utf-8")
    print(f"[done] {run_dir} best epoch {best_epoch} pooled {best_iou:.4f}", flush=True)


if __name__ == "__main__":
    main()
