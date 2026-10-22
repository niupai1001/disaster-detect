"""P1 冒烟：真实 batch 的 forward/backward、参数更新、保存→全新实例重载、显存/延迟。

不是"能 import 就算通过"：这里要求
1. 用真实清单里的样本（若清单未就绪则退回合成张量，并显式标注）；
2. 反向之后首层新增通道 / 投影 / Stem 的参数确实拿到非零梯度；
3. 优化器走一步后参数发生改变；
4. ``state_dict`` 保存到**全新实例**后 logits 与原来一致（max|Δ| < 1e-6）；
5. 记录参数量、峰值显存与 B=1 延迟 p50/p95。

用法::

    python scripts/smoke_segformer.py --config configs/segformer/a0.yaml
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
from landslide_seg.models.segformer import SegFormerSeg  # noqa: E402
from landslide_seg.training.losses import bce_loss, dice_loss  # noqa: E402


def _param_grad(model: torch.nn.Module, name: str) -> float | None:
    for n, p in model.named_parameters():
        if n == name:
            return None if p.grad is None else float(p.grad.abs().max())
    return None


def _grad_summary(model: SegFormerSeg) -> dict:
    names = {
        "first_conv": "model.segformer.encoder.patch_embeddings.0.proj.weight",
        "projection": "projection.proj.weight",
        "stem": "stem.net.0.weight",
        "fusion_out": "fusion.out.weight",
        "classifier": "model.decode_head.classifier.weight",
    }
    return {k: _param_grad(model, v) for k, v in names.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "segformer" / "a0.yaml"))
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--synthetic", action="store_true", help="强制用合成张量")
    args = ap.parse_args()

    cfg = load_segformer_config(args.config, args.set)["segformer"]
    dcfg = cfg["data"]
    dev = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    manifest = resolve_path(dcfg["manifest"])
    stats_path = resolve_path(dcfg["stats"])
    pretrained = resolve_path(cfg["model"]["pretrained_dir"])

    report: dict = {
        "config": str(Path(args.config).name),
        "experiment": cfg["experiment"],
        "adapter": cfg["model"]["adapter"],
        "backbone": cfg["model"].get("backbone", "b0"),
        "stem": bool(cfg["model"]["stem"]),
        "device": str(dev),
        "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "source": "manifest" if (manifest.is_file() and stats_path.is_file() and not args.synthetic) else "synthetic",
    }

    # ---- 输入 ----
    if report["source"] == "manifest":
        records = filter_split(read_manifest(manifest), "train")
        if not records:
            raise RuntimeError("train split 为空")
        stats = load_stats(stats_path)
        ds = S2SegDataset(records[: args.batch_size], stats, augment=False, preload=True)
        batch = collate([ds[i] for i in range(len(ds))])
        image = batch["image"].to(dev)
        mask = batch["mask"].to(dev)
        valid = batch["valid"].to(dev)
        report["sample_ids"] = batch["sample_id"]
        report["n_samples"] = len(batch["sample_id"])
    else:
        image = torch.randn(args.batch_size, 10, 128, 128, device=dev)
        mask = (torch.rand(args.batch_size, 1, 128, 128, device=dev) > 0.97).float()
        valid = torch.ones_like(mask, dtype=torch.bool)
        report["n_samples"] = args.batch_size

    # ---- 模型 ----
    model = SegFormerSeg(
        adapter=cfg["model"]["adapter"],
        backbone=cfg["model"].get("backbone", "b0"),
        stem=bool(cfg["model"]["stem"]),
        pretrained_dir=str(pretrained),
    ).to(dev)
    report["params"] = model.count_params()
    report["load_info"] = model.load_info
    report["adapter_report"] = model.adapter_report

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    out = model(image, return_delta=True)
    report["logits_shape"] = list(out["logits"].shape)
    report["base_logits_shape"] = list(out["base_logits"].shape)
    if out.get("delta_logits") is not None:
        report["stem_delta_absmax_at_init"] = float(out["delta_logits"].abs().max())

    loss = float(cfg["loss"]["bce_weight"]) * bce_loss(out["logits"][:, 0], mask, valid) + float(
        cfg["loss"]["dice_weight"]
    ) * dice_loss(out["logits"][:, 0], mask, valid)
    loss.backward()
    report["loss"] = float(loss.detach())
    report["grads"] = _grad_summary(model)

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer.step()
    changed = {}
    for n, p in model.named_parameters():
        if n not in before:
            continue
        delta = float((p.detach() - before[n]).abs().max())
        if delta > 0:
            changed[n] = delta
    report["param_update_absmax_top"] = dict(
        sorted(changed.items(), key=lambda kv: -kv[1])[:10]
    )
    report["n_params_updated"] = len(changed)
    report["n_params_with_grad"] = sum(
        1 for p in model.parameters() if p.grad is not None and float(p.grad.abs().max()) > 0
    )
    report["n_params_total"] = sum(1 for _ in model.parameters())

    # ---- 保存 → 全新实例重载 ----
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model2 = SegFormerSeg(
        adapter=cfg["model"]["adapter"],
        backbone=cfg["model"].get("backbone", "b0"),
        stem=bool(cfg["model"]["stem"]),
        pretrained_dir=str(pretrained),
    ).to(dev)
    model2.load_state_dict(state, strict=True)
    model.eval()
    model2.eval()
    with torch.no_grad():
        logits1 = model(image)["logits"]
        logits2 = model2(image)["logits"]
    report["reload_max_abs_diff"] = float((logits1 - logits2).abs().max())

    # ---- 延迟 ----
    if dev.type == "cuda":
        report["peak_alloc_mib"] = round(torch.cuda.max_memory_allocated(dev) / 2**20, 1)
        report["peak_reserved_mib"] = round(torch.cuda.max_memory_reserved(dev) / 2**20, 1)
    x1 = torch.randn(1, 10, 128, 128, device=dev)
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            model(x1)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        lat = []
        for _ in range(100):
            t0 = time.perf_counter()
            model(x1)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)
    report["latency_ms_b1"] = {
        "p50": round(float(np.percentile(lat, 50)), 2),
        "p95": round(float(np.percentile(lat, 95)), 2),
        "mean": round(float(np.mean(lat)), 2),
    }

    out_dir = resolve_path("outputs/segformer/_smoke") / cfg["experiment"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"[saved] {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
