"""主模型（单期/多期 + 可选光谱关系编码）的精确更新训练器。

对应计划 §1、§3、§4：

- 固定 3600 次**成功更新**、有效 batch 64、230400 次目标图块曝光；
- 为适配多帧显存，统一使用 ``microbatch × accumulation = 64``；四组使用同一
  微批量与累积方式（本批实测后锁定为 8 × 8），不因单期更快而改成 batch64；
- 梯度累积时 BCE 按**全有效 batch 的有效像素总数**归一化，Dice 按**样本数**加权，
  否则累积结果不等同原目标函数；
- 每 100 次更新只按 VAL pooled IoU@0.5 选 best；TEST 不参与任何选择；
- 单期和多期共用同一时间聚合接口；T=1 时退化为原空间特征。

BatchNorm 说明：SegFormer 解码器含 BatchNorm，microbatch=8 的批统计与 batch64
不同。四组共用同一 microbatch，因此组间可比，但不宣称与历史 batch64 结果逐位等价；
该差异写入 run_meta 并在报告中标注。
"""

from __future__ import annotations

import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..config import file_sha256, manifest_identity, set_seed
from ..data.temporal_v6 import collate_v6
from .metrics import confusion, summarize


# --------------------------------------------------------------------------
# 确定性样本流：与 update_training.FullBatchStream 同一抽样规则
# --------------------------------------------------------------------------
class TemporalStream:
    def __init__(self, n: int, batch_size: int, seed: int, state: dict | None = None):
        if n < 1 or batch_size < 1:
            raise ValueError("需要非空数据集与正 batch_size")
        self.n, self.batch_size, self.seed = int(n), int(batch_size), int(seed)
        self.epoch, self.offset = 0, 0
        self.exposures = np.zeros(self.n, dtype=np.int64)
        if state:
            if (state["n"], state["batch_size"], state["seed"]) != (self.n, self.batch_size, self.seed):
                raise ValueError("采样器身份与 checkpoint 不一致")
            self.epoch, self.offset = int(state["epoch"]), int(state["offset"])
            self.exposures = np.asarray(state["exposures"], dtype=np.int64)
        self.order = self._order()

    def _order(self) -> list[int]:
        gen = torch.Generator().manual_seed(self.seed * 100_003 + self.epoch)
        return torch.randperm(self.n, generator=gen).tolist()

    def next_indices(self, k: int) -> list[int]:
        out = []
        for _ in range(int(k)):
            if self.offset >= len(self.order):
                self.epoch += 1
                self.offset = 0
                self.order = self._order()
            idx = self.order[self.offset]
            out.append(idx)
            self.exposures[idx] += 1
            self.offset += 1
        return out

    def state_dict(self) -> dict:
        return {"n": self.n, "batch_size": self.batch_size, "seed": self.seed,
                "epoch": self.epoch, "offset": self.offset,
                "exposures": self.exposures.tolist()}


# --------------------------------------------------------------------------
# 评测（只算 pooled / 逐事件像素指标，用于选模）
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate_pooled(model, ds, indices, device, *, batch_size: int, threshold: float = 0.5) -> dict:
    model.eval()
    counts, by_event = [], {}
    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        batch = collate_v6([ds[i] for i in chunk])
        logits = model(
            batch["image"].to(device), batch["dates"].to(device),
            batch["keep"].to(device), batch["anchor"].to(device),
        )["logits"].float()
        prob = logits[:, 0].sigmoid().cpu().numpy()
        gt = batch["mask"].numpy().astype(bool)
        valid = batch["valid"].numpy().astype(bool)
        for i, ev in enumerate(batch["event_id"]):
            c = confusion(prob[i] >= threshold, gt[i], valid[i])
            counts.append(c)
            by_event.setdefault(ev, []).append(c)
    pooled = summarize(counts)
    events = {ev: summarize(cs)["iou"] for ev, cs in sorted(by_event.items())}
    values = [float(v) for v in events.values() if v is not None and np.isfinite(v)]
    return {"pooled_iou": float(pooled["iou"]), "dice": float(pooled["dice"]),
            "precision": float(pooled["precision"]), "recall": float(pooled["recall"]),
            "event_mean_iou": float(np.mean(values)) if values else None,
            "by_event": events}


# --------------------------------------------------------------------------
# 损失项：返回全 batch 可精确累加的 bce 总和 / dice 总和
# --------------------------------------------------------------------------
def _seg_terms(logits: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor):
    logits = logits.float()
    target = mask.float()
    v = valid.float()
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce_sum = (raw * v).sum()
    prob = torch.sigmoid(logits) * v
    tgt = target * v
    dims = tuple(range(1, prob.ndim))
    inter = (prob * tgt).sum(dim=dims)
    denom = prob.sum(dim=dims) + tgt.sum(dim=dims)
    dice = (2 * inter + 1e-6) / (denom + 1e-6)
    dice_sum = (1.0 - dice).sum()
    return bce_sum, dice_sum


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict) -> None:
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _save(path: Path, model, optimizer, scheduler, *, step, best_step, best_score,
          cfg, meta, stream, history, elapsed, module_activity=None) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": int(step),
        "best_step": int(best_step),
        "best_val_iou": float(best_score),
        "config": cfg,
        "meta": meta,
        "stream_state": stream.state_dict(),
        "history": history,
        "rng": _rng_state(),
        "params": model.count_params(),
        "adapter_report": model.adapter_report,
        "module_activity": module_activity or model.module_activity(),
        "elapsed_seconds": float(elapsed),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_temporal_training(cfg: dict, model, train_ds, val_ds, train_indices, val_indices,
                          meta: dict, run_dir: Path, device: torch.device,
                          *, resume: bool = False, log=print, stop_after: int | None = None) -> dict:
    s = cfg["segformer"]
    tc = s["train"]
    target = int(tc["max_steps"])
    interval = int(tc.get("eval_every_updates", 100))
    report_steps = sorted({int(x) for x in tc.get("report_steps", [target])})
    micro = int(tc.get("microbatch", 8))
    batch = int(tc["batch_size"])
    if batch % micro:
        raise ValueError("effective batch 必须能被 microbatch 整除")
    accum = batch // micro
    if int(tc.get("accumulation", accum)) != accum:
        raise ValueError("accumulation 与 batch_size/microbatch 不一致")
    if target % interval:
        raise ValueError("max_steps 必须是 eval_every_updates 的整数倍")

    optimizer = torch.optim.AdamW([
        {"params": model.core.param_groups()["encoder"], "lr": float(s["optim"]["encoder_lr"])},
        {"params": model.core.param_groups()["decoder"] + model.core.param_groups()["new"]
         + (list(model.pools.parameters()) if model.pools is not None else [])
         + (list(model.spectral.parameters()) if model.spectral is not None else []),
         "lr": float(s["optim"]["head_lr"])},
    ], weight_decay=float(s["optim"]["weight_decay"]))
    warmup = max(1, int(target * float(s["optim"]["warmup_frac"])))
    power = float(s["optim"].get("poly_power", 1.0))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, target - warmup)
        return max(0.0, (1.0 - progress) ** power)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    history: list[dict] = []
    step, best_step, best_score = 0, -1, -1.0
    best_dev: dict | None = None
    stream_state = None
    elapsed_before = 0.0

    if resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"没有可恢复的 checkpoint: {last_path}")
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        if ckpt["config"] != cfg:
            raise ValueError("恢复时配置与 checkpoint 不一致")
        if ckpt["meta"]["manifest"] != meta["manifest"] or ckpt["meta"]["stats_sha256"] != meta["stats_sha256"]:
            raise ValueError("恢复时数据清单/统计与 checkpoint 不一致")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        step, best_step, best_score = int(ckpt["global_step"]), int(ckpt["best_step"]), float(ckpt["best_val_iou"])
        stream_state, history = ckpt["stream_state"], ckpt["history"]
        elapsed_before = float(ckpt.get("elapsed_seconds", 0.0))
        _restore_rng(ckpt["rng"])
        log(f"[resume] step={step} best={best_score:.4f}@{best_step}")
        del ckpt

    stream = TemporalStream(len(train_ds), batch, int(s["seed"]), stream_state)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in history),
                            encoding="utf-8")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    log(f"[updates] {s['experiment']} device={device} start={step} target={target} "
        f"train={len(train_indices)} val={len(val_indices)} batch={batch} micro={micro} accum={accum}")
    started = time.time()
    eval_batch = int(s["eval"].get("batch_size", 8))

    while step < target:
        stop = min(target, ((step // interval) + 1) * interval)
        if stop_after is not None:
            stop = min(stop, int(stop_after))
        bce_total, dice_total, samples = 0.0, 0.0, 0
        epoch_bce, epoch_dice, epoch_valid, epoch_samples = 0.0, 0.0, 0.0, 0
        while step < stop:
            indices = stream.next_indices(batch)
            train_ds.set_epoch(stream.epoch)
            batch_cpu = collate_v6([train_ds[i] for i in indices])
            image = batch_cpu["image"]
            mask_all = batch_cpu["mask"]
            valid_all = batch_cpu["valid"]
            total_valid = float(valid_all.sum())
            if total_valid <= 0:
                raise RuntimeError("有效像素为 0，不能构成一次成功更新")
            optimizer.zero_grad(set_to_none=True)
            bce_batch, dice_batch = 0.0, 0.0
            for m in range(accum):
                sl = slice(m * micro, (m + 1) * micro)
                img = image[sl].to(device, non_blocking=True)
                dates = batch_cpu["dates"][sl].to(device)
                keep = batch_cpu["keep"][sl].to(device)
                anchor = batch_cpu["anchor"][sl].to(device)
                mask = mask_all[sl].to(device)
                valid = valid_all[sl].to(device)
                out = model(img, dates, keep, anchor)
                bce_sum, dice_sum = _seg_terms(out["logits"][:, 0], mask, valid)
                loss = (float(s["loss"]["bce_weight"]) * bce_sum / total_valid
                        + float(s["loss"]["dice_weight"]) * dice_sum / batch)
                loss.backward()
                bce_batch += float(bce_sum.detach())
                dice_batch += float(dice_sum.detach())
            clip = float(s["optim"]["grad_clip"])
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            bce_total += bce_batch
            dice_total += dice_batch
            epoch_bce += bce_batch
            epoch_dice += dice_batch
            epoch_valid += total_valid
            epoch_samples += batch
            samples += batch
            if not np.isfinite(bce_batch) or not np.isfinite(dice_batch):
                raise FloatingPointError("非有限损失")

        dev = evaluate_pooled(model, val_ds, val_indices, device,
                              batch_size=eval_batch,
                              threshold=float(s["eval"]["threshold"]))
        model.train()
        sel = str(s["eval"].get("selection_metric", "dev_event_mean_iou"))
        if sel == "dev_event_mean_iou":
            value = dev["event_mean_iou"]
        elif sel == "dev_pooled_iou":
            value = dev["pooled_iou"]
        else:
            raise ValueError(f"未知 eval.selection_metric: {sel!r}")
        if value is None or not np.isfinite(value):
            raise RuntimeError(f"选模指标 {sel} 未定义")
        improved = value > best_score
        if improved:
            best_score, best_step = value, step
            best_dev = dev
        elapsed = elapsed_before + time.time() - started
        row = {
            "global_step": step,
            "epoch": stream.epoch,
            "train_loss": (float(s["loss"]["bce_weight"]) * epoch_bce / max(epoch_valid, 1e-12)
                           + float(s["loss"]["dice_weight"]) * epoch_dice / max(epoch_samples, 1)),
            "dev": dev,
            "val_iou": value,
            "dev_pooled_iou": dev["pooled_iou"],
            "dev_event_mean_iou": dev["event_mean_iou"],
            "selection_metric": sel,
            "lr": [float(g["lr"]) for g in optimizer.param_groups],
            "samples_exposed": int(stream.exposures.sum()),
            "seconds": elapsed,
        }
        history.append(row)
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        _save(last_path, model, optimizer, scheduler, step=step, best_step=best_step,
              best_score=best_score, cfg=cfg, meta=meta, stream=stream, history=history,
              elapsed=elapsed)
        if improved:
            _save(best_path, model, optimizer, scheduler, step=step, best_step=best_step,
                  best_score=best_score, cfg=cfg, meta=meta, stream=stream, history=history,
                  elapsed=elapsed)
        if step in report_steps:
            _save(run_dir / f"step{step}.pt", model, optimizer, scheduler, step=step,
                  best_step=best_step, best_score=best_score, cfg=cfg, meta=meta,
                  stream=stream, history=history, elapsed=elapsed)
            log(f"[checkpoint] step{step}.pt")
        ev_text = " ".join(f"{ev}={iou:.4f}" for ev, iou in dev["by_event"].items())
        log(f"[step {step:4d}] bce={float(s['loss']['bce_weight']) * epoch_bce / max(epoch_valid, 1e-12):.4f} "
            f"dice={float(s['loss']['dice_weight']) * epoch_dice / max(epoch_samples, 1):.4f} "
            f"dev_mean={dev['event_mean_iou']:.4f} dev_pooled={dev['pooled_iou']:.4f} "
            f"{ev_text} best={best_score:.4f}@{best_step} time={elapsed/60:.1f}m")
        if stop_after is not None and step >= int(stop_after):
            break

    elapsed = elapsed_before + time.time() - started
    summary = {
        "experiment": s["experiment"], "seed": int(s["seed"]), "run_dir": str(run_dir),
        "n_train": len(train_indices), "n_val": len(val_indices), "params": model.count_params(),
        "global_step": step, "target_updates": target, "complete": step == target,
        "batch_size": batch, "microbatch": micro, "accumulation": accum,
        "selection_metric": sel, "best_val_iou": best_score, "best_step": best_step,
        "best_dev_event_mean_iou": best_dev["event_mean_iou"] if best_dev else None,
        "best_dev_pooled_iou": best_dev["pooled_iou"] if best_dev else None,
        "best_dev_by_event": best_dev["by_event"] if best_dev else None,
        "seconds": elapsed, "device": str(device), "history": history,
        "best_checkpoint": str(best_path), "last_checkpoint": str(last_path),
        "samples_exposed": int(stream.exposures.sum()),
        "unique_samples_exposed": int((stream.exposures > 0).sum()),
        "exposures_min": int(stream.exposures.min()),
        "exposures_max": int(stream.exposures.max()),
        "module_activity": model.module_activity(),
        "peak_alloc_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
        "batchnorm_note": f"decoder BatchNorm 统计基于 microbatch {micro}（四组一致），不等价于历史 batch64",
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[done] updates={step}/{target} {sel}={best_score:.5f}@{best_step} {elapsed/60:.1f}min")
    return summary
