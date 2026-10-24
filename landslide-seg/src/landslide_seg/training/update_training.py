"""Exact-update control for the existing SegFormer trainer and optimizer."""
from __future__ import annotations

import itertools
import json
import random
import time
from collections import Counter

import numpy as np
import torch

from ..data.segformer_dataset import collate
from .budget import resolve_update_budget
from .metrics import confusion, summarize


class FullBatchStream:
    """Deterministic shuffles, carrying the tail across epochs; resumable cursor."""
    def __init__(self, dataset, batch_size, seed, state=None):
        if not len(dataset) or batch_size < 1:
            raise ValueError("Nonempty dataset and positive batch_size required")
        self.dataset, self.batch_size, self.seed = dataset, int(batch_size), int(seed)
        self.epoch, self.offset = 0, 0
        self.exposures = np.zeros(len(dataset), dtype=np.int64)
        if state:
            if state["n"] != len(dataset) or state["batch_size"] != self.batch_size or state["seed"] != self.seed:
                raise ValueError("Sampler identity differs from checkpoint")
            self.epoch, self.offset = state["epoch"], state["offset"]
            self.exposures = np.asarray(state["exposures"], dtype=np.int64)
        self.order = self._order()

    def _order(self):
        gen = torch.Generator().manual_seed(self.seed * 100_003 + self.epoch)
        return torch.randperm(len(self.dataset), generator=gen).tolist()

    def __iter__(self):
        return self

    def __next__(self):
        samples = []
        for _ in range(self.batch_size):
            if self.offset == len(self.order):
                self.epoch += 1
                self.offset = 0
                self.order = self._order()
            idx = self.order[self.offset]
            self.dataset.set_epoch(self.epoch)
            if hasattr(self.dataset, 'exposure'):
                self.dataset.exposure = int(self.exposures[idx])
            samples.append(self.dataset[idx])
            self.offset += 1
            self.exposures[idx] += 1
        return collate(samples)

    def state_dict(self):
        return {"n": len(self.dataset), "batch_size": self.batch_size, "seed": self.seed,
                "epoch": self.epoch, "offset": self.offset, "exposures": self.exposures.tolist()}


@torch.no_grad()
def evaluate_events(model, loader, cfg, device):
    model.eval()
    counts = []
    by_event = {}
    threshold = float(cfg["segformer"]["eval"]["threshold"])
    # S4 门控诊断：与像素评价同一次前向，不额外跑一遍 DEV。
    acc = {"sum": 0.0, "sq": 0.0, "n": 0, "pos": 0.0, "pos_n": 0,
           "bg": 0.0, "bg_n": 0, "corr": 0.0, "corr_n": 0, "fp": 0.0, "fp_n": 0}
    for batch in loader:
        out = model(batch["image"].to(device))
        prob = out["logits"].float()[:, 0].sigmoid()
        gate = getattr(model, "last_gate", None)
        corr = getattr(model, "last_corr_mag", None)
        prob = prob.cpu().numpy()
        gt, valid = batch["mask"].numpy().astype(bool), batch["valid"].numpy().astype(bool)
        if gate is not None or corr is not None:
            import torch.nn.functional as F
            size = gt.shape[-2:]
            g = (F.interpolate(gate.float(), size=size, mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
                 if gate is not None else None)
            c = (F.interpolate(corr.float(), size=size, mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
                 if corr is not None else None)
            pred = prob >= threshold
            n_img = prob.shape[0]
            for i in range(n_img):
                m = valid[i]
                if g is not None:
                    gp = g[i][m]
                    if gp.size:
                        acc["sum"] += float(gp.sum()); acc["sq"] += float((gp ** 2).sum()); acc["n"] += gp.size
                    pos = m & gt[i]
                    if pos.any():
                        acc["pos"] += float(g[i][pos].sum()); acc["pos_n"] += int(pos.sum())
                    neg = m & ~gt[i]
                    if neg.any():
                        acc["bg"] += float(g[i][neg].sum()); acc["bg_n"] += int(neg.sum())
                if c is not None:
                    fp_mask = m & ~gt[i] & pred[i]
                    cp = c[i][m]
                    if cp.size:
                        acc["corr"] += float(cp.sum()); acc["corr_n"] += cp.size
                    if fp_mask.any():
                        acc["fp"] += float(c[i][fp_mask].sum()); acc["fp_n"] += int(fp_mask.sum())
        for i, event in enumerate(batch["event_id"]):
            c = confusion(prob[i] >= threshold, gt[i], valid[i])
            counts.append(c)
            by_event.setdefault(event, []).append(c)
    events = {ev: summarize(cs) for ev, cs in sorted(by_event.items())}
    values = [m["iou"] for m in events.values() if m["iou"] is not None and np.isfinite(m["iou"])]
    result = {"pooled": summarize(counts), "by_event": events,
              "event_mean_iou": float(np.mean(values)) if values else None}
    if acc["n"] or acc["corr_n"]:
        info = {
            "gate_mean": None, "gate_var": None, "gate_pos_mean": None, "gate_bg_mean": None,
            "corr_abs_mean": acc["corr"] / acc["corr_n"] if acc["corr_n"] else None,
            "corr_abs_mean_fp": acc["fp"] / acc["fp_n"] if acc["fp_n"] else None,
            "alpha": float(model.alpha.detach()) if hasattr(model, "alpha") else None,
            "fusion": getattr(model, "fusion_mode", None),
        }
        if acc["n"]:
            n = float(acc["n"])
            mean = acc["sum"] / n
            info.update({
                "gate_mean": mean,
                "gate_var": max(0.0, acc["sq"] / n - mean ** 2),
                "gate_pos_mean": acc["pos"] / acc["pos_n"] if acc["pos_n"] else None,
                "gate_bg_mean": acc["bg"] / acc["bg_n"] if acc["bg_n"] else None,
            })
        result["spectral_gate"] = info
    return result


def run_updates(cfg, model, optimizer, scheduler, train_ds, val_ds, train_recs, val_recs,
                stats, meta, seed_info, run_dir, device, resume, log):
    # Import only after segformer_trainer has finished loading (shared implementation).
    from .segformer_trainer import _train_epoch, _make_loader, _save_checkpoint
    s, tc = cfg["segformer"], cfg["segformer"]["train"]
    budget = resolve_update_budget(tc)
    target, interval = budget.stop_updates, budget.eval_every_updates
    if target <= 0 or interval <= 0 or s.get("sampling", {}).get("mode", "natural") != "natural":
        raise ValueError("Update mode requires positive budget/interval and natural sampling")
    label = str(s["eval"].get("selection_metric", "dev_event_mean_iou"))
    last, best = run_dir / "last.pt", run_dir / "best.pt"
    history, step, best_step, score, skipped_total, previous_seconds = [], 0, -1, -1.0, 0, 0.0
    stream_state = None
    if resume:
        if not last.is_file():
            raise FileNotFoundError(f"No checkpoint to resume: {last}")
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        if ckpt["config"] != cfg or ckpt["meta"]["manifest"] != meta["manifest"] or ckpt["meta"]["stats_sha256"] != meta["stats_sha256"] or ckpt['meta'].get('use_quality_sha256') != meta.get('use_quality_sha256'):
            raise ValueError("Resume configuration/data differs from saved checkpoint")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        step, best_step, score = ckpt["global_step"], ckpt["best_step"], ckpt["best_val_iou"]
        stream_state, history = ckpt["stream_state"], ckpt["history"]
        skipped_total, previous_seconds = ckpt["skipped_total"], ckpt["elapsed_seconds"]
        rng = ckpt["rng"]
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        if "python" in rng:
            random.setstate(rng["python"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        del ckpt
    stream = FullBatchStream(train_ds, tc["batch_size"], s["seed"], stream_state)
    val_loader = _make_loader(val_ds, tc["batch_size"], False, s["seed"], 0)
    # Eval DataLoader consumes only its own RNG; cannot perturb model dropout on resume.
    val_loader.generator = torch.Generator().manual_seed(0)
    train_eval_ds = type(train_ds)(train_recs, stats, augment=False, preload=False,
                                   extra=getattr(train_ds, "extra", "none"),
                                   sidecar=getattr(train_ds, "sidecar_path", None))
    train_eval_loader = _make_loader(train_eval_ds, tc["batch_size"], False, s["seed"], 0)
    train_eval_loader.generator = torch.Generator().manual_seed(0)
    if not resume:
        (run_dir / "resolved_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
        (run_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in history), encoding="utf-8")
    started = time.time()
    log(f"[updates] {s['experiment']} device={device} start={step} target={target} train={len(train_recs)} dev={len(val_recs)} interval={interval}")
    while step < target:
        stop = min(target, ((step // interval) + 1) * interval)
        total_loss, total_count, total_coarse, stalled = 0.0, 0, 0.0, 0
        while step < stop:
            loss, skipped, n, coarse = _train_epoch(model, itertools.islice(stream, stop - step), optimizer, scheduler, cfg, device)
            skipped_total += skipped
            step += n
            total_loss += loss * n
            total_coarse += coarse * n
            total_count += n
            stalled = stalled + 1 if n == 0 else 0
            if stalled >= 3:
                raise RuntimeError("No valid optimizer updates in three complete chunks")
        dev = evaluate_events(model, val_loader, cfg, device)
        train_metric = evaluate_events(model, train_eval_loader, cfg, device)
        integrity = model.probe_integrity() if hasattr(model, "probe_integrity") else None
        if integrity and (not integrity["encoder_unchanged"] or integrity["encoder_has_grad"] or integrity["encoder_training"]):
            raise RuntimeError("Frozen encoder probe integrity failed")
        # 选模口径由 eval.selection_metric 决定；默认沿用事件平均，官方协议用 pooled。
        sel = str(cfg["segformer"]["eval"].get("selection_metric", "dev_event_mean_iou"))
        if sel == "dev_pooled_iou":
            value = (dev.get("pooled") or {}).get("iou")
            label = "dev_pooled_iou"
        elif sel == "dev_event_mean_iou":
            value = dev["event_mean_iou"]
            label = "dev_event_mean_iou"
        else:
            raise ValueError(f"未知 eval.selection_metric: {sel!r}")
        if value is None:
            raise RuntimeError(f"No defined selection metric ({label})")
        improved = value > score
        if improved:
            score, best_step = value, step
        elapsed = previous_seconds + time.time() - started
        row = {"global_step": step, "epoch": stream.epoch, "train_loss": total_loss / max(total_count, 1),
               "coarse_loss": total_coarse / max(total_count, 1), "train_clean": train_metric,
               "dev": dev, "val_iou": value, "selection_metric": label,
               "lr": [float(g["lr"]) for g in optimizer.param_groups], "skipped_batches": skipped_total,
               "samples_exposed": int(stream.exposures.sum()), "seconds": elapsed}
        if "spectral_gate" in dev:
            row["dev_spectral"] = dev["spectral_gate"]
        if integrity is not None:
            row["probe_integrity"] = integrity
        history.append(row)
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        extra = {"global_step": step, "best_step": best_step, "stream_state": stream.state_dict(),
                 "history": history, "stats": stats, "skipped_total": skipped_total, "elapsed_seconds": elapsed}
        _save_checkpoint(last, model, optimizer, scheduler, stream.epoch, score, stream.epoch, cfg, meta, seed_info, extra=extra)
        if improved:
            _save_checkpoint(best, model, optimizer, scheduler, stream.epoch, score, stream.epoch, cfg, meta, seed_info, extra=extra)
        if step in budget.report_steps and step != target:
            # 预定步权重：与时间段一致的 1000/1500/2000 选模窗口证据
            _save_checkpoint(run_dir / f"step{step}.pt", model, optimizer, scheduler,
                             stream.epoch, score, stream.epoch, cfg, meta, seed_info, extra=extra)
        event_text = " ".join(f"{ev}={m['iou']:.4f}" for ev, m in dev["by_event"].items())
        log(f"[step {step:4d}] loss={row['train_loss']:.4f} train={train_metric['pooled']['iou']:.4f} dev_mean={value:.4f} {event_text} best={score:.4f}@{best_step} time={elapsed/60:.1f}m")
        # Test/recovery hook only; production configs omit this field.
        if not resume and tc.get("pause_after_updates") and step >= int(tc["pause_after_updates"]):
            break
    elapsed = previous_seconds + time.time() - started
    summary = {"experiment": s["experiment"], "seed": s["seed"], "run_dir": str(run_dir),
               "n_train": len(train_recs), "n_val": len(val_recs), "params": model.count_params(),
               "global_step": step, "target_updates": target,
               "stop_updates": target, "schedule_updates": budget.schedule_updates,
               "report_steps": list(budget.report_steps),
               "complete": step == target,
               "selection_metric": label, "best_val_iou": score, "best_step": best_step,
               "seconds": elapsed, "device": str(device), "history": history,
               "best_checkpoint": str(best), "last_checkpoint": str(last), "skipped_batches": skipped_total,
               "samples_exposed": int(stream.exposures.sum()), "unique_samples_exposed": int((stream.exposures > 0).sum()),
               "exposures_min": int(stream.exposures.min()), "exposures_max": int(stream.exposures.max()),
               "dev_events": dict(Counter(r["event_id"] for r in val_recs)),
               "peak_alloc_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
               "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None}
    (run_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (run_dir / "sample_exposures.json").write_text(json.dumps(dict(zip([r["sample_id"] for r in train_recs], stream.exposures.tolist()))), encoding="utf-8")
    if getattr(train_ds, 'mixed', False):
        from ..data.optical_observation import training_subset
        with (run_dir / 'band_views.jsonl').open('w', encoding='utf-8') as handle:
            for rec, count in zip(train_recs, stream.exposures):
                for exposure in range(int(count)):
                    handle.write(json.dumps(dict(sample_id=rec['sample_id'], exposure=exposure,
                        subset=training_subset(s['seed'], rec['sample_id'], exposure)))+'\n')
    log(f"[done] updates={step}/{target} dev_mean={score:.5f}@{best_step} {elapsed/60:.1f}min")
    return summary
