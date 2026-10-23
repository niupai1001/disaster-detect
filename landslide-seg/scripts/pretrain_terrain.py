"""地形编码器的自监督预训练与同源监督热身（DEM／坡度）。

任务定义（计划 §3）：

- ``ssl``／``ssl_extra``：每折仅用本折 TRAIN footprint；同一位置同时遮挡 DEM 与
  坡度（50% 块状遮挡，16×16 块），只重建 DEM；3000 次成功更新、batch 64、
  AdamW lr 6e-4。``ssl_extra`` 从 ``ssl`` 状态再续 3000 步，用作"预算控制"。
- ``scoring``：从 ``ssl`` 状态出发，用本折 TRAIN 的既有 GT 生成局部前景比例
  （16×16 块均值），训练同一个读出头 3000 步；不引入 ``annotated=False`` 负图块。
  该评分称为"同源监督的相对地形关联评分"，不是历史先验或校准概率。

产物写到 ``weights/terrain/``：``encoder`` 与（视任务）``head``／``decoder``
的 state_dict、步数、统计哈希与训练行哈希。
"""

from __future__ import annotations

import argparse
import hashlib
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
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from landslide_seg.config import file_sha256, resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import S2SegDataset, load_stats, read_manifest  # noqa: E402
from landslide_seg.models.terrain import TerrainEncoder, TerrainScoreHead, valid_block_fraction  # noqa: E402

OUT = ROOT / "weights" / "terrain"
BLOCK = 16
GRID = 128 // BLOCK


class ReconstructHead(nn.Module):
    def __init__(self, in_chans: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_chans, 64, 3, padding=1, bias=False), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1, bias=False), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, padding=1, bias=False), nn.GroupNorm(8, 16), nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.body(feat)


def block_mask(batch: int, device, rng: torch.Generator, ratio: float = 0.5) -> torch.Tensor:
    keep = torch.rand(batch, 1, GRID, GRID, generator=rng) < ratio
    keep = keep.repeat_interleave(BLOCK, dim=2).repeat_interleave(BLOCK, dim=3)
    return keep.to(device)


def batches(rows, stats, seed, batch, augment=True):
    ds = S2SegDataset(rows, stats, augment=augment, seed=seed, extra="both",
                      preload=False, sidecar="data/v4r1_geo")
    return DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0, drop_last=True,
                      generator=torch.Generator().manual_seed(seed))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=("ssl", "ssl_extra", "scoring"))
    ap.add_argument("--fold", default="v4r1", help="协议标识，默认 event_ood_v4r1 的 v4r1 侧表")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--init", default=None, help="scoring/ssl_extra 的起点权重")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.fold != "v4r1":
        raise ValueError("此入口只读取 v4r1，不能用 --fold 改名冒充其他协议")
    if args.steps <= 0 or args.batch <= 0:
        raise ValueError("steps 与 batch 必须为正数")
    out_path = Path(args.out) if args.out else OUT / f"v4r1raw_{args.task}_valid_v2.pt"
    if out_path.exists():
        raise FileExistsError(f"不覆盖已有地形权重: {out_path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)

    man = ROOT / "manifests" / "v4r1"
    pool = read_manifest(man / "pool_100.jsonl")
    rows = [r for r in pool if r["protocol_role"] == "TRAIN"]
    stats = load_stats(man / "stats_geoboth_raw.json")
    loader = batches(rows, stats, args.seed, args.batch, augment=args.task != "scoring")
    if len(loader) == 0:
        raise ValueError("batch 大于训练样本数，drop_last 会产生空训练循环")

    encoder = TerrainEncoder(2, pretrained=False).to(dev)
    head = TerrainScoreHead(encoder.out_channels).to(dev) if args.task == "scoring" else None
    decoder = ReconstructHead(encoder.out_channels).to(dev) if args.task != "scoring" else None

    init = args.init
    if init is None and args.task in ("ssl_extra", "scoring"):
        init = str(OUT / "v4r1raw_ssl.pt")
    if init:
        ckpt = torch.load(Path(init), map_location="cpu", weights_only=False)
        for key, path in (("stats_sha256", man / "stats_geoboth_raw.json"),
                          ("pool_sha256", man / "pool_100.jsonl")):
            if ckpt.get(key) != file_sha256(path):
                raise ValueError(f"初始化 {key} 与当前训练协议不符")
        encoder.load_state_dict(ckpt["encoder"])
        if args.task == "ssl_extra":
            decoder.load_state_dict(ckpt["decoder"])
        if args.task == "scoring" and ckpt.get("head"):
            head.load_state_dict(ckpt["head"])
        print(f"[init] {init}", flush=True)

    params = list(encoder.parameters())
    params += list(head.parameters()) if head is not None else []
    params += list(decoder.parameters()) if decoder is not None else []
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator().manual_seed(args.seed)
    t0, step, running = time.time(), 0, 0.0
    encoder.train()
    (head or decoder).train()
    while step < args.steps:
        for batch in loader:
            image = batch["image"].to(dev)                 # [B,12,128,128] 已按本折统计 z-score
            terrain = image[:, 10:]                        # DEM, SLOPE
            dem = image[:, 10:11]
            opt.zero_grad(set_to_none=True)
            if args.task == "scoring":
                target, weights = valid_block_fraction(batch["mask"].to(dev).unsqueeze(1),
                                                       batch["valid"].to(dev).unsqueeze(1), BLOCK)
                logits = F.avg_pool2d(head(encoder(terrain), (128, 128)), BLOCK)
                loss = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weights).sum() \
                    / weights.sum().clamp_min(1e-12)
            else:
                mask = block_mask(terrain.shape[0], dev, rng)
                masked = terrain * (~mask).float()
                recon = decoder(encoder(masked))
                weights = mask & batch["valid"].to(dev).unsqueeze(1).bool()
                loss = torch.where(weights, (recon - dem).abs(), 0.0).sum() / weights.sum().clamp_min(1)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{args.task} loss 非有限")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            running += float(loss.detach())
            if step % 500 == 0 or step == args.steps:
                print(f"[{args.fold}/{args.task}] step {step}/{args.steps} loss {running/500:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                running = 0.0
            if step >= args.steps:
                break

    payload = {
        "encoder": encoder.state_dict(),
        "steps": args.steps,
        "task": args.task,
        "target_version": 2,
        "target_definition": "valid 像素的16×16区块滑坡比例；空区块不参与损失" if head is not None else "仅在遮挡且valid的像素重建DEM",
        "fold": args.fold,
        "seed": args.seed,
        "lr": args.lr,
        "batch": args.batch,
        "mask": {"block": BLOCK, "ratio": 0.5, "same_position_dem_slope": True,
                 "reconstruct": "DEM"},
        "stats_sha256": file_sha256(man / "stats_geoboth_raw.json"),
        "pool_sha256": file_sha256(man / "pool_100.jsonl"),
        "fit_ids_sha256": hashlib.sha256(
            "\n".join(sorted(r["sample_id"] for r in rows)).encode()).hexdigest(),
        "init": init,
        "seconds": round(time.time() - t0, 1),
    }
    if head is not None:
        payload["head"] = head.state_dict()
    if decoder is not None:
        payload["decoder"] = decoder.state_dict()
    torch.save(payload, out_path)
    print(json.dumps({"out": str(out_path).replace(chr(92), "/"), "steps": args.steps,
                      "seconds": payload["seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
