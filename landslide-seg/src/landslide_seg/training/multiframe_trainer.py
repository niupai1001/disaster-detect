"""多帧/光谱主模型的训练循环：微批量累积 + 精确更新预算 + 单一评价范围。

计划要求（``docs/planned_temporal_spectral_experiments_2026-10-20.md`` §1）：

- 四组统一用**同一微批量与同一累积方式**（默认 microbatch 8 × accumulation 8 = 有效 batch 64）；
- 每 100 次**成功更新**只在 VAL 上按 pooled IoU@0.5 选 best；
- 梯度累积时 BCE 按**全有效 batch 的总有效像素**归一化，Dice 按样本数加权，
  否则累积结果不等同原目标函数；
- 记录输入帧曝光、训练时长与峰值显存；多帧不是同计算预算。

实现上先把一个有效 batch 的 64 个样本物化（数据集项很轻：每样本最多 15×10×128×128，
约 12 MB），据此得到全局分母，再逐微批前反向。这样累积梯度与「一次性 64 样本」的目标
函数在数学上等价，而不需要把激活全留在显存里。
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.temporal_v6 import collate_v6
from .metrics import confusion, summarize


# --------------------------------------------------------------------------
# 有效 batch 流（与 FullBatchStream 相同的洗牌语义，但输出「样本列表」）
# --------------------------------------------------------------------------
class EffectiveBatchStream:
    def __init__(self, dataset, batch_size: int, seed: int):
        if not len(dataset) or batch_size < 1:
            raise ValueError("需要非空数据集与正的有效 batch")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch, self.offset = 0, 0
        self.exposures = np.zeros(len(dataset), dtype=np.int64)
        self.order = self._order()

    def _order(self):
        gen = torch.Generator().manual_seed(self.seed * 100_003 + self.epoch)
        return torch.randperm(len(self.dataset), generator=gen).tolist()

    def next_batch(self) -> list[dict]:
        items = []
        for _ in range(self.batch_size):
            if self.offset == len(self.order):
                self.epoch += 1
                self.offset = 0
                self.order = self._order()
            idx = self.order[self.offset]
            self.dataset.set_epoch(self.epoch)
            items.append(self.dataset[idx])
            self.offset += 1
            self.exposures[idx] += 1
        return items

    def state_dict(self) -> dict:
        return {"n": len(self.dataset), "batch_size": self.batch_size, "seed": self.seed,
                "epoch": self.epoch, "offset": self.offset, "exposures": self.exposures.tolist()}

    def load_state_dict(self, state: dict) -> None:
        if (state["n"] != len(self.dataset) or state["batch_size"] != self.batch_size
                or state["seed"] != self.seed):
            raise ValueError("采样器身份与 checkpoint 不一致")
        self.epoch = int(state["epoch"])
        self.offset = int(state["offset"])
        self.exposures = np.asarray(state["exposures"], dtype=np.int64)
        self.order = self._order()


# --------------------------------------------------------------------------
# 损失：返回分子，由调用方用全 batch 分母归一化
# --------------------------------------------------------------------------
def _pos_weight(cfg: dict, logits: torch.Tensor) -> torch.Tensor | None:
    pw = float(cfg["segformer"]["loss"].get("pos_weight", 1.0) or 1.0)
    if pw == 1.0:
        return None
    return torch.tensor([pw], device=logits.device, dtype=logits.dtype)


def dice_per_sample(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                    eps: float = 1e-6) -> torch.Tensor:
    """逐样本 Dice（只在 valid 像素上统计），返回 ``[B]``。"""
    logits = logits.float().flatten(1)
    target = target.float().flatten(1)
    v = valid.float().flatten(1)
    prob = torch.sigmoid(logits)
    inter = (prob * target * v).sum(1)
    denom = ((prob + target) * v).sum(1)
    return (2 * inter + eps) / (denom + eps)


def loss_numerators(cfg: dict, logits: torch.Tensor, mask: torch.Tensor,
                    valid: torch.Tensor) -> dict:
    """返回 ``bce_sum``（按 valid 求和）与 ``dice_sum``（逐样本求和）。"""
    logits = logits.float()[:, 0]
    raw = F.binary_cross_entropy_with_logits(
        logits, mask.float(), reduction="none", pos_weight=_pos_weight(cfg, logits)
    )
    bce_sum = (raw * valid.float()).sum()
    dice_sum = dice_per_sample(logits, mask, valid).sum()
    return {"bce_sum": bce_sum, "dice_sum": dice_sum,
            "valid_px": int(valid.sum()), "n": int(logits.shape[0])}


def _combine(cfg: dict, parts: dict, valid_total: int, n_total: int) -> torch.Tensor:
    lc = cfg["segformer"]["loss"]
    return (float(lc["bce_weight"]) * parts["bce_sum"] / max(valid_total, 1)
            + float(lc["dice_weight"]) * parts["dice_sum"] / max(n_total, 1))


# --------------------------------------------------------------------------
# 评测
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate_multiframe(model, loader, cfg: dict, device: torch.device) -> dict:
    model.eval()
    threshold = float(cfg["segformer"]["eval"]["threshold"])
    counts, by_event = [], {}
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        prob = model(image, batch["dates"].to(device), batch["keep"].to(device),
                     batch["anchor"].to(device))["logits"].float()[:, 0].sigmoid().cpu().numpy()
        gt = batch["mask"].numpy().astype(bool)
        v = batch["valid"].numpy().astype(bool)
        for i, event in enumerate(batch["event_id"]):
            c = confusion(prob[i] >= threshold, gt[i], v[i])
            counts.append(c)
            by_event.setdefault(event, []).append(c)
    events = {ev: summarize(cs) for ev, cs in sorted(by_event.items())}
    values = [m["iou"] for m in events.values() if m["iou"] is not None and np.isfinite(m["iou"])]
    return {"pooled": summarize(counts), "by_event": events,
            "event_mean_iou": float(np.mean(values)) if values else None}


def make_loader(ds, batch_size: int, shuffle: bool = False, seed: int = 0) -> DataLoader:
    gen = None
    if shuffle:
        gen = torch.Generator().manual_seed(int(seed))
    return DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=0,
                      collate_fn=collate_v6, generator=gen, drop_last=False)


# --------------------------------------------------------------------------
# 主循环
# --------------------------------------------------------------------------
def run_multiframe(cfg: dict, model, optimizer, scheduler, train_ds, val_ds, train_recs,
                   val_recs, stats, meta, seed_info, run_dir, device, resume: bool,
                   log) -> dict:
    from .segformer_trainer import _save_checkpoint

    s = cfg["segformer"]
    tc = s["train"]
    target = int(tc["max_steps"])
    interval = int(tc.get("eval_every_updates", 100))
    eff = int(tc.get("effective_batch", tc["batch_size"]))
    micro = int(tc.get("micro_batch", tc["batch_size"]))
    if eff <= 0 or micro <= 0 or eff % micro:
        raise ValueError("effective_batch 必须为正且能被 micro_batch 整除")
    accum = eff // micro
    clip = float(s["optim"]["grad_clip"])
    sel = str(s["eval"].get("selection_metric", "dev_pooled_iou"))
    if sel != "dev_pooled_iou":
        raise ValueError("多帧/光谱主模型统一按 VAL pooled IoU 选模")

    last, best = run_dir / "last.pt", run_dir / "best.pt"
    stream = EffectiveBatchStream(train_ds, eff, int(s["seed"]))
    history: list[dict] = []
    step, best_step, score, elapsed_prev = 0, -1, -1.0, 0.0
    frames_seen = 0
    if resume:
        if not last.is_file():
            raise FileNotFoundError(f"没有可恢复的 checkpoint: {last}")
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        if ckpt["config"] != cfg:
            raise ValueError("恢复配置与 checkpoint 不一致")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        step, best_step, score = ckpt["global_step"], ckpt["best_step"], ckpt["best_val_iou"]
        history = ckpt["history"]
        elapsed_prev = float(ckpt.get("elapsed_seconds", 0.0))
        frames_seen = int(ckpt.get("frames_seen", 0))
        stream.load_state_dict(ckpt["stream_state"])
        rng = ckpt["rng"]
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        del ckpt

    if not resume:
        (run_dir / "resolved_config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
        (run_dir / "run_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in history),
                            encoding="utf-8")

    val_loader = make_loader(val_ds, micro)
    val_loader.generator = torch.Generator().manual_seed(0)
    log(f"[multiframe] {s['experiment']} device={device} target={target} eff={eff} "
        f"micro={micro} accum={accum} train={len(train_recs)} dev={len(val_recs)}")

    started = time.time()
    model.train()
    while step < target:
        stop = min(target, ((step // interval) + 1) * interval)
        total_loss, total_count, stalled = 0.0, 0, 0
        while step < stop:
            items = stream.next_batch()
            micro_batches = [items[i:i + micro] for i in range(0, len(items), micro)]
            batches = [collate_v6(mb) for mb in micro_batches]
            valid_total = int(sum(int(b["valid"].sum()) for b in batches))
            n_total = int(sum(b["image"].shape[0] for b in batches))
            if valid_total == 0:
                stalled += 1
                if stalled >= 3:
                    raise RuntimeError("连续三个有效 batch 无有效像素")
                continue
            optimizer.zero_grad(set_to_none=True)
            update_loss = 0.0
            for b in batches:
                image = b["image"].to(device, non_blocking=True)
                out = model(image, b["dates"].to(device), b["keep"].to(device),
                            b["anchor"].to(device))
                parts = loss_numerators(cfg, out["logits"], b["mask"].to(device),
                                        b["valid"].to(device))
                loss = _combine(cfg, parts, valid_total, n_total)
                if not torch.isfinite(loss):
                    raise FloatingPointError("训练损失出现非有限值")
                loss.backward()
                update_loss += float(loss.detach())
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            frames_seen += n_total * int(batches[0]["image"].shape[1])
            total_loss += update_loss
            total_count += 1

        dev = evaluate_multiframe(model, val_loader, cfg, device)
        model.train()
        value = (dev.get("pooled") or {}).get("iou")
        if value is None:
            raise RuntimeError("VAL pooled IoU 未定义")
        improved = value > score
        if improved:
            score, best_step = float(value), step
        elapsed = elapsed_prev + time.time() - started
        row = {"global_step": step, "epoch": stream.epoch,
               "train_loss": total_loss / max(total_count, 1), "dev": dev,
               "val_iou": float(value), "selection_metric": sel,
               "lr": [float(g["lr"]) for g in optimizer.param_groups],
               "samples_exposed": int(stream.exposures.sum()), "frames_exposed": int(frames_seen),
               "seconds": round(elapsed, 2)}
        history.append(row)
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        extra = {"global_step": step, "best_step": best_step, "stream_state": stream.state_dict(),
                 "history": history, "stats": stats, "skipped_total": 0,
                 "elapsed_seconds": elapsed, "frames_seen": frames_seen}
        _save_checkpoint(last, model, optimizer, scheduler, stream.epoch, score, best_step,
                         cfg, meta, seed_info, extra=extra)
        if improved:
            _save_checkpoint(best, model, optimizer, scheduler, stream.epoch, score, best_step,
                             cfg, meta, seed_info, extra=extra)
        log(f"[step {step:4d}] loss={row['train_loss']:.4f} dev={value:.4f} "
            f"best={score:.4f}@{best_step} frames={frames_seen} time={elapsed/60:.1f}m")

    elapsed = elapsed_prev + time.time() - started
    summary = {"experiment": s["experiment"], "seed": s["seed"], "run_dir": str(run_dir),
               "n_train": len(train_recs), "n_val": len(val_recs),
               "params": model.count_params(), "global_step": step, "target_updates": target,
               "complete": step == target, "effective_batch": eff, "micro_batch": micro,
               "accumulation": accum, "selection_metric": sel, "best_val_iou": score,
               "best_step": best_step, "seconds": elapsed, "device": str(device),
               "frames_exposed": int(frames_seen),
               "samples_exposed": int(stream.exposures.sum()),
               "peak_alloc_mib": torch.cuda.max_memory_allocated(device) / 2**20
               if device.type == "cuda" else None,
               "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20
               if device.type == "cuda" else None,
               "history": history, "best_checkpoint": str(best), "last_checkpoint": str(last)}
    (run_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
    log(f"[done] updates={step}/{target} dev={score:.5f}@{best_step} {elapsed/60:.1f}min "
        f"frames={frames_seen}")
    return summary
