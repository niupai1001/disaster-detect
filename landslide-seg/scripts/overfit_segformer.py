"""P2 过拟合诊断：2–8 个可靠训练瓦片、关闭增强、最多 500 次更新。

判据（不是"跑了一次 backward 就算学会"）：
- loss 相对首步下降 ≥50%；
- 每个瓦片都出现预测正像素（不退化为常数）；
- 训练瓦片 IoU/Dice 有明确提升（记录实际值，不硬编必须达到的 Dice）。

用法::

    python scripts/overfit_segformer.py --config configs/segformer/a0.yaml --tiles 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:  # Windows 控制台/管道默认 cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402
import torch  # noqa: E402

from landslide_seg.config import load_segformer_config, resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import (  # noqa: E402
    S2SegDataset,
    collate,
    filter_split,
    load_stats,
    read_manifest,
)
from landslide_seg.models.factory import build_model, model_type  # noqa: E402
from landslide_seg.training.losses import bce_loss, dice_loss  # noqa: E402
from landslide_seg.training.metrics import confusion  # noqa: E402


def pick_tiles(records: list[dict], n: int) -> list[dict]:
    """按正像素数在分布上取分位点，覆盖小/中/大目标。"""
    pos = [r for r in records if r["mask_px"] > 0]
    pos.sort(key=lambda r: r["mask_px"])
    if len(pos) <= n:
        return pos
    idx = np.linspace(0, len(pos) - 1, n).round().astype(int).tolist()
    picked, seen = [], set()
    for i in idx:
        if i in seen:
            continue
        seen.add(i)
        picked.append(pos[i])
    return picked


def evaluate_tiles(prob: np.ndarray, gt: np.ndarray, valid: np.ndarray, thr: float = 0.5) -> dict:
    pred = prob >= thr
    c = confusion(pred, gt, valid)
    denom = c["tp"] + c["fp"] + c["fn"]
    iou = c["tp"] / denom if denom else float("nan")
    dice = 2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"]) if (2 * c["tp"] + c["fp"] + c["fn"]) else float("nan")
    return {"iou": iou, "dice": dice, "pred_px": int(pred.sum()), "gt_px": int(gt.sum()), **c}


def _rgb(x10: np.ndarray) -> np.ndarray:
    rgb = np.stack([x10[2], x10[1], x10[0]], axis=-1).astype(np.float32)
    lo, hi = np.percentile(rgb, [2, 98], axis=(0, 1))
    return np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "segformer" / "a0.yaml"))
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--tiles", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_segformer_config(args.config, args.set)["segformer"]
    dev = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    manifest = resolve_path(cfg["data"]["manifest"])
    stats_path = resolve_path(cfg["data"]["stats"])
    pretrained = resolve_path(cfg["model"]["pretrained_dir"])

    records = filter_split(read_manifest(manifest), "train")
    stats = load_stats(stats_path)
    chosen = pick_tiles(records, args.tiles)
    if not chosen:
        raise RuntimeError("train split 里没有正瓦片")

    ds = S2SegDataset(chosen, stats, augment=False, preload=True)
    batch = collate([ds[i] for i in range(len(ds))])
    image = batch["image"].to(dev)
    mask = batch["mask"].to(dev)
    valid = batch["valid"].to(dev)

    model = build_model(
        cfg,
        pretrained_dir=pretrained if model_type(cfg) == "segformer" else None,
    ).to(dev)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    bce_w = float(cfg["loss"]["bce_weight"])
    dice_w = float(cfg["loss"]["dice_weight"])
    history: list[dict] = []
    t0 = time.time()
    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        out = model(image)
        logits = out["logits"].float()[:, 0]
        loss = bce_w * bce_loss(logits, mask, valid) + dice_w * dice_loss(logits, mask, valid)
        coarse_w = float(cfg["loss"].get("coarse_weight", 0.0) or 0.0)
        if coarse_w > 0:
            coarse = out["base_logits"].float()[:, 0]
            loss = loss + coarse_w * (
                bce_w * bce_loss(coarse, mask, valid) + dice_w * dice_loss(coarse, mask, valid)
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            with torch.no_grad():
                prob = torch.sigmoid(model(image)["logits"].float()[:, 0])
                per_tile = evaluate_tiles(
                    prob.cpu().numpy(), mask.cpu().numpy(), valid.cpu().numpy()
                )
            history.append({"step": step, "loss": float(loss.detach()), "iou": per_tile["iou"]})
            print(f"[step {step:3d}] loss={float(loss.detach()):.4f} mean_iou={per_tile['iou']:.4f}")

    with torch.no_grad():
        prob = torch.sigmoid(model(image)["logits"].float()[:, 0]).cpu().numpy()
    gt = mask.cpu().numpy().astype(bool)
    v = valid.cpu().numpy().astype(bool)

    tiles = []
    all_positive = True
    for i, rec in enumerate(chosen):
        m = evaluate_tiles(prob[i], gt[i], v[i])
        tiles.append(
            {
                "sample_id": rec["sample_id"],
                "event_id": rec["event_id"],
                "gt_px": m["gt_px"],
                "pred_px": m["pred_px"],
                "iou": round(float(m["iou"]), 4),
                "dice": round(float(m["dice"]), 4),
            }
        )
        if m["pred_px"] == 0:
            all_positive = False

    first = history[0]["loss"] if history else float("nan")
    last = history[-1]["loss"] if history else float("nan")
    drop = (first - last) / first if first and first > 0 else 0.0
    mean_iou = float(np.nanmean([t["iou"] for t in tiles]))
    mean_dice = float(np.nanmean([t["dice"] for t in tiles]))
    report = {
        "experiment": cfg["experiment"],
        "seed": cfg["seed"],
        "adapter": cfg["model"]["adapter"],
        "backbone": cfg["model"].get("backbone", "b0"),
        "stem": bool(cfg["model"]["stem"]),
        "device": str(dev),
        "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
        "n_tiles": len(chosen),
        "max_steps": args.max_steps,
        "lr": args.lr,
        "first_loss": first,
        "last_loss": last,
        "loss_drop_frac": round(drop, 4),
        "mean_train_iou": round(mean_iou, 4),
        "mean_train_dice": round(mean_dice, 4),
        "all_tiles_have_pred_positives": all_positive,
        "history": history,
        "tiles": tiles,
        "seconds": round(time.time() - t0, 1),
        "checks": {
            "loss_drop_ge_50pct": bool(drop >= 0.5),
            "all_tiles_predicted_positive": bool(all_positive),
            "mean_iou_positive": bool(mean_iou > 0),
            "foreground_dice_ge_0.90": bool(mean_dice >= 0.90),
        },
    }
    report["all_passed"] = all(report["checks"].values())

    out_dir = resolve_path("outputs/segformer/_overfit") / cfg["experiment"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    # ---- 图：最多 4 个瓦片的 RGB / GT / 概率 / 预测 ----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_show = min(4, len(chosen))
        fig, axes = plt.subplots(n_show, 4, figsize=(12, 3 * n_show), squeeze=False)
        for i in range(n_show):
            rec = chosen[i]
            raw = np.load(rec["cache"])["x10"].astype(np.float32)
            axes[i][0].imshow(_rgb(raw))
            axes[i][0].set_title(f"{rec['sample_id']} RGB")
            axes[i][1].imshow(gt[i], cmap="gray")
            axes[i][1].set_title("GT")
            axes[i][2].imshow(prob[i], cmap="magma", vmin=0, vmax=1)
            axes[i][2].set_title("prob")
            axes[i][3].imshow(prob[i] >= 0.5, cmap="gray")
            axes[i][3].set_title("pred")
            for j in range(4):
                axes[i][j].axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / "overfit.png", dpi=110)
        plt.close(fig)
    except Exception as exc:  # 可视化失败不影响诊断结论
        report["figure_error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps({k: v for k, v in report.items() if k not in ("history", "tiles")},
                     indent=1, ensure_ascii=False))
    print(f"[saved] {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
