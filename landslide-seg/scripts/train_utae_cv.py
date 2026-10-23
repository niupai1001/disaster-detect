"""四折事件协议 · U-TAE 时间对照训练（配置驱动、统一更新预算）。

与 ``train_utae_v6.py`` 的差别只在"协议与预算"三处，模型与日期接口完全复用：

1. **配置驱动**：折、TRAIN/INNER/OUTER 清单、统计、时间模式、输出目录、恢复状态
   全部来自 YAML，不再硬编码官方 manifest；
2. **统一批序**：三组（single / multi / history）按 ``(seed, epoch)`` 生成同一条
   图块置换，逐 step 拿到同一批图块，每 epoch 更新次数都是 ``ceil(N/batch)``；
   变长序列只在模型内部按 ``keep`` 分组（``UTAERealDates`` 已有该路径），
   不再用长度分桶改变优化器更新次数；
3. **单一损失/单一更新**：每个 batch 一次共同损失、一次 ``step``；补齐帧由
   ``keep`` 显式标记，不靠数值零识别 padding。

选模：每 epoch 用 INNER 事件平均 IoU（同分取较早状态），OUTER 不参与选择。

用法::

    python scripts/train_utae_cv.py --config configs/utae/event_cv/F1.yaml --mode single
    python scripts/train_utae_cv.py --config configs/utae/event_cv/F1.yaml --mode history --resume
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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

import yaml  # noqa: E402

from landslide_seg.config import file_sha256, resolve_path, set_seed  # noqa: E402
from landslide_seg.data.segformer_dataset import load_stats, read_manifest  # noqa: E402
from landslide_seg.data.temporal_v6 import MODES, TemporalV6, collate_v6  # noqa: E402
from landslide_seg.models.utae_dates import UTAERealDates  # noqa: E402
from landslide_seg.training.losses import bce_loss, dice_loss  # noqa: E402
from landslide_seg.training.metrics import confusion, summarize  # noqa: E402


# --------------------------------------------------------------------------- 配置
def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_cv_config(path: Path, overrides: list[str] | None = None) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base_ref = raw.pop("_base", None)
    cfg: dict = {}
    if base_ref:
        bp = (path.parent / base_ref).resolve()
        cfg = yaml.safe_load(bp.read_text(encoding="utf-8")) or {}
    cfg = _deep_merge(cfg, raw)
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        try:
            node[parts[-1]] = json.loads(value)
        except Exception:
            node[parts[-1]] = value
    if "utae_cv" not in cfg:
        raise ValueError("配置缺少 utae_cv 段")
    return cfg["utae_cv"]


def config_sha(cfg: dict) -> str:
    body = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


# --------------------------------------------------------------------------- 批序
def epoch_batches(n: int, batch: int, seed: int, epoch: int) -> list[list[int]]:
    """由 ``(seed, epoch)`` 唯一决定的图块批序；三组共用，尾批不丢弃。"""
    g = torch.Generator().manual_seed(seed * 100_003 + epoch)
    perm = torch.randperm(n, generator=g).tolist()
    return [perm[i:i + batch] for i in range(0, n, batch)]


def fixed_batches(n: int, batch: int) -> list[list[int]]:
    return [list(range(i, min(i + batch, n))) for i in range(0, n, batch)]


# --------------------------------------------------------------------------- 评测
@torch.no_grad()
def evaluate(model, ds, device, threshold: float, batch: int) -> dict:
    model.eval()
    loader = DataLoader(ds, batch_sampler=fixed_batches(len(ds), batch),
                        num_workers=0, collate_fn=collate_v6)
    counts, by_event = [], {}
    for batch_items in loader:
        logits = model(batch_items["image"].to(device), batch_items["dates"].to(device),
                       batch_items["keep"].to(device), batch_items["anchor"].to(device))["logits"].float()
        prob = logits[:, 0].sigmoid().cpu().numpy()
        gt = batch_items["mask"].numpy().astype(bool)
        va = batch_items["valid"].numpy().astype(bool)
        cells = [(prob[i] >= threshold, gt[i], va[i]) for i in range(len(prob))]
        for i, ev in enumerate(batch_items["event_id"]):
            c = confusion(*cells[i])
            counts.append(c)
            by_event.setdefault(ev, []).append(c)
    pooled = summarize(counts)
    per_event = {ev: summarize(cs) for ev, cs in sorted(by_event.items())}
    ious = [v["iou"] for v in per_event.values() if np.isfinite(v["iou"])]
    return {
        "pooled": {k: _r(pooled[k]) for k in ("iou", "dice", "precision", "recall")},
        "event_mean_iou": float(np.mean(ious)) if ious else None,
        "per_event": {ev: {"iou": _r(v["iou"]), "dice": _r(v["dice"]),
                           "precision": _r(v["precision"]), "recall": _r(v["recall"]),
                           "n_pixels_valid": int(v["tp"] + v["fp"] + v["fn"] + v["tn"])}
                      for ev, v in per_event.items()},
    }


def _r(v) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(v), 6)


# --------------------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--limit-batches", type=int, default=None, help="冒烟用，正式训练不要设")
    ap.add_argument("--tag", default="", help="冒烟/调试输出后缀，避免污染正式 run")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    cfg = load_cv_config(resolve_path(args.config), args.set)
    seed = int(cfg["seed"])
    dev = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    prot = cfg["protocol"]
    pool_rows = read_manifest(resolve_path(prot["pool"]))
    outer_rows = read_manifest(resolve_path(prot["outer"])) if prot.get("outer") else []
    roles = prot["roles"]
    train_rows = [r for r in pool_rows if r.get("protocol_role") == roles["train"]]
    inner_rows = [r for r in pool_rows if r.get("protocol_role") == roles["inner"]]
    if not train_rows or not inner_rows:
        raise RuntimeError("TRAIN/INNER 至少有一组为空")
    ids = lambda rows: {r["sample_id"] for r in rows}  # noqa: E731
    if ids(train_rows) & ids(inner_rows) or ids(train_rows) & ids(outer_rows) or ids(inner_rows) & ids(outer_rows):
        raise RuntimeError("TRAIN/INNER/OUTER 成员重叠")

    stats_path = resolve_path(prot["stats"])
    stats = load_stats(stats_path)
    dem = bool(cfg["data"].get("dem", True))
    in_ch = int(cfg["data"].get("in_channels", 11))
    if len(np.asarray(stats["mean"]).ravel()) != in_ch:
        raise ValueError(f"统计通道数与 in_channels={in_ch} 不一致")

    tcfg, ecfg = cfg["train"], cfg["eval"]
    epochs, batch = int(tcfg["epochs"]), int(tcfg["batch_size"])
    run_dir = ROOT / cfg["output_dir"] / str(cfg["fold"]) / args.mode / str(seed)
    if args.tag:
        run_dir = run_dir.parent / f"{args.mode}_{args.tag}" / str(seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_info = set_seed(seed, bool(cfg.get("deterministic", True)))
    model = UTAERealDates(in_channels=in_ch, num_classes=1).to(dev)
    init_digest = model.digest()
    cfg_sha = config_sha(cfg)
    n_updates = (len(train_rows) + batch - 1) // batch
    meta = {
        "fold": cfg["fold"], "mode": args.mode, "seed": seed, "seed_info": seed_info,
        "epochs": epochs, "batch_size": batch, "updates_per_epoch": n_updates,
        "expected_total_updates": epochs * n_updates,
        "lr": tcfg["lr"], "weight_decay": tcfg["weight_decay"], "grad_clip": tcfg["grad_clip"],
        "pos_weight": tcfg["pos_weight"], "bce": tcfg["bce_weight"], "dice": tcfg["dice_weight"],
        "amp": bool(tcfg.get("amp", False)),
        "threshold": ecfg["threshold"], "selection": ecfg["selection"],
        "stats": str(prot["stats"]), "stats_sha256": file_sha256(stats_path),
        "pool": str(prot["pool"]), "pool_sha256": file_sha256(resolve_path(prot["pool"])),
        "outer": str(prot.get("outer")), "outer_sha256": file_sha256(resolve_path(prot["outer"])) if prot.get("outer") else None,
        "temporal_index_sha256": file_sha256(resolve_path(prot.get("temporal_root", "data/s2_temporal_v5")) / "index.json"),
        "quality_complete_sha256": (file_sha256(resolve_path(prot["quality"]) / "complete.json")
                                    if prot.get("quality") else None),
        "n_train": len(train_rows), "n_inner": len(inner_rows), "n_outer": len(outer_rows),
        "train_events": sorted({r["event_id"] for r in train_rows}),
        "inner_events": sorted({r["event_id"] for r in inner_rows}),
        "outer_events": sorted({r["event_id"] for r in outer_rows}),
        "params": model.count_params(), "init_digest": init_digest, "config_sha256": cfg_sha,
        "batch_order": "same permutation for all modes: torch.randperm(seed*100003+epoch)",
        "padding": "explicit keep mask; variable lengths grouped inside UTAERealDates; no length bucketing",
        "dates": {"source": str(Path(prot.get("temporal_root", "data/s2_temporal_v5")) / "index.json"), "mode": args.mode},
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")

    quality = prot.get("quality")
    tr_ds = TemporalV6(train_rows, stats, mode=args.mode, augment=True, seed=seed,
                       dem=dem, preload=bool(cfg["data"].get("preload", False)),
                       root=prot.get("temporal_root", "data/s2_temporal_v5"),
                       quality=quality or "data/s2_temporal_v4r1_none")
    in_ds = TemporalV6(inner_rows, stats, mode=args.mode, augment=False, seed=seed, dem=dem,
                       root=prot.get("temporal_root", "data/s2_temporal_v5"),
                       quality=quality or "data/s2_temporal_v4r1_none")
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]),
                            weight_decay=float(tcfg["weight_decay"]))

    start_epoch, best, best_epoch, history = 1, -1.0, -1, []
    if args.resume and (run_dir / "last.pt").is_file():
        ck = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        if ck.get("config_sha256") != cfg_sha:
            raise ValueError("恢复时配置哈希不一致")
        if ck.get("init_digest") != init_digest:
            raise ValueError("恢复时初始化摘要不一致")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_epoch = int(ck["epoch"]) + 1
        best, best_epoch = float(ck["best"]), int(ck["best_epoch"])
        history = ck["history"]
        print(f"[resume] epoch {start_epoch} best {best:.4f}@{best_epoch}", flush=True)

    last_epoch = args.max_epochs or epochs
    t0 = time.time()
    for epoch in range(start_epoch, last_epoch + 1):
        tr_ds.set_epoch(epoch)
        batches = epoch_batches(len(tr_ds), batch, seed, epoch)
        if args.limit_batches:
            batches = batches[:args.limit_batches]
        loader = DataLoader(tr_ds, batch_sampler=batches, num_workers=0, collate_fn=collate_v6)
        model.train()
        tot, cnt, skipped = 0.0, 0, 0
        for batch_items in loader:
            opt.zero_grad(set_to_none=True)
            logits = model(batch_items["image"].to(dev), batch_items["dates"].to(dev),
                           batch_items["keep"].to(dev), batch_items["anchor"].to(dev))["logits"]
            mask = batch_items["mask"].to(dev).unsqueeze(1)
            valid = batch_items["valid"].to(dev).unsqueeze(1)
            loss = (float(tcfg["bce_weight"]) * bce_loss(logits, mask, valid,
                                                         pos_weight=float(tcfg["pos_weight"]))
                    + float(tcfg["dice_weight"]) * dice_loss(logits, mask, valid))
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]),
                                           error_if_nonfinite=False)
            opt.step()
            tot += float(loss.detach())
            cnt += 1
        ev = evaluate(model, in_ds, dev, float(ecfg["threshold"]), batch)
        score = ev["event_mean_iou"] if ecfg["selection"] == "inner_event_mean_iou" else ev["pooled"]["iou"]
        improved = score is not None and score > best
        if improved:
            best, best_epoch = float(score), epoch
        payload = {"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch,
                   "inner": ev, "best": best, "best_epoch": best_epoch, "history": history,
                   "meta": meta, "init_digest": init_digest, "config_sha256": cfg_sha}
        torch.save(payload, run_dir / "last.pt")
        if improved:
            torch.save(payload, run_dir / "best.pt")
        row = {"epoch": epoch, "updates": cnt, "skipped": skipped,
               "train_loss": round(tot / max(cnt, 1), 6), "inner": ev,
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if epoch % int(tcfg.get("log_every", 1)) == 0 or epoch == last_epoch:
            print(f"[{cfg['fold']}/{args.mode}] epoch {epoch:3d} loss {row['train_loss']:.4f} "
                  f"inner pooled {ev['pooled']['iou']} event_mean {ev['event_mean_iou']} "
                  f"best {best_epoch}", flush=True)

    summary = {
        "fold": cfg["fold"], "mode": args.mode, "best_epoch": best_epoch, "best_inner_selection": best,
        "selection": ecfg["selection"], "epochs": last_epoch, "updates_per_epoch": n_updates,
        "seconds": round(time.time() - t0, 1), "meta": meta, "history": history,
        "complete": last_epoch == epochs and not args.limit_batches,
        "peak_alloc_mib": round(torch.cuda.max_memory_allocated(dev) / 2**20, 1) if dev.type == "cuda" else None,
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                                                encoding="utf-8")
    print(f"[done] {run_dir} best epoch {best_epoch} score {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
