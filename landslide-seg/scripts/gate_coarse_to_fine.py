"""P2 运行门槛：真实样本的前向/反传/梯度/更新/条件敏感性与重载。

对 A1-AUX、CF1、CF0、U0 各做一次真实 batch 检查，并单独验证：
- CF1 的最终输出**依赖**显式条件通道（替换 P 后输出改变）；
- CF0 的最终输出**不依赖**被替换的 P（条件恒为 0）；
- 零初始化精分头在若干步后确实拿到梯度（首步上游为 0 属预期）。

产物：``outputs/segformer/coarse_to_fine/gates.json``。

用法::

    python scripts/gate_coarse_to_fine.py
    python scripts/gate_coarse_to_fine.py --models cf1 u0
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
from landslide_seg.models.factory import build_model  # noqa: E402
from landslide_seg.training.losses import bce_loss, dice_loss  # noqa: E402

CONFIG_DIR = ROOT / "configs" / "segformer" / "coarse_to_fine"


def _seg_loss(cfg, logits, mask, valid):
    return float(cfg["loss"]["bce_weight"]) * bce_loss(logits[:, 0], mask, valid) + float(
        cfg["loss"]["dice_weight"]
    ) * dice_loss(logits[:, 0], mask, valid)


def _total_loss(cfg, out, mask, valid):
    total = _seg_loss(cfg, out["logits"], mask, valid)
    cw = float(cfg["loss"].get("coarse_weight", 0.0) or 0.0)
    if cw > 0:
        total = total + cw * _seg_loss(cfg, out["base_logits"], mask, valid)
    return total


def run_one(name: str, device: torch.device) -> dict:
    cfg = load_segformer_config(CONFIG_DIR / f"{name}.yaml")["segformer"]
    pretrained = resolve_path(cfg["model"]["pretrained_dir"]) if cfg["model"].get("type", "segformer") == "segformer" else None
    model = build_model(cfg, pretrained_dir=pretrained).to(device)

    records = filter_split(read_manifest(resolve_path(cfg["data"]["manifest"])), "train")[:8]
    stats = load_stats(resolve_path(cfg["data"]["stats"]))
    ds = S2SegDataset(records, stats, preload=True)
    batch = collate([ds[i] for i in range(4)])
    image = batch["image"].to(device)
    mask = batch["mask"].to(device)
    valid = batch["valid"].to(device)

    report = {
        "name": name,
        "model_type": cfg["model"].get("type", "segformer"),
        "refine": cfg["model"].get("refine", "none"),
        "coarse_weight": cfg["loss"].get("coarse_weight", 0.0),
        "params": model.count_params(),
        "model_identity": getattr(model, "load_info", None),
        "device": str(device),
    }

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    t0 = time.time()
    losses = []
    for step in range(1, 6):
        opt.zero_grad(set_to_none=True)
        out = model(image)
        loss = _total_loss(cfg, out, mask, valid)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name}: 非有限损失")
        loss.backward()
        grads = {n: float(p.grad.abs().max()) for n, p in model.named_parameters() if p.grad is not None}
        opt.step()
        losses.append(float(loss.detach()))
        if step == 1:
            report["grads_step1"] = {
                "max": max(grads.values()),
                "nonzero_params": sum(1 for v in grads.values() if v > 0),
                "total_params": len(grads),
                "refiner_out": grads.get("refiner.out.weight"),
                "stem": grads.get("stem.net.0.weight"),
                "encoder_conv1": grads.get("model.segformer.encoder.patch_embeddings.0.proj.weight")
                or grads.get("model.encoder.conv1.weight"),
            }
        if step == 5:
            report["grads_step5"] = {
                "refiner_out": grads.get("refiner.out.weight"),
                "refiner_body": grads.get("refiner.body.0.weight"),
                "stem": grads.get("stem.net.0.weight"),
            }
    report["losses"] = [round(v, 5) for v in losses]
    report["finite"] = all(np.isfinite(v) for v in losses)
    report["seconds"] = round(time.time() - t0, 2)

    # 条件敏感性：CF1 替换 P 应改变输出；CF0 应不敏感
    if cfg["model"].get("refine", "none") in ("guided", "zero_cond"):
        model.eval()
        with torch.no_grad():
            base = model(image)["logits"]
            zeros = torch.zeros_like(base)
            replaced = model(image, cond_override=zeros)["logits"]
        report["cond_sensitivity_max_abs"] = float((base - replaced).abs().max())
        report["cond_sensitive"] = bool(report["cond_sensitivity_max_abs"] > 1e-6)

    # 重载：state_dict → 全新实例
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    fresh = build_model(cfg, pretrained_dir=pretrained).to(device)
    fresh.load_state_dict(state, strict=True)
    model.eval()
    fresh.eval()
    with torch.no_grad():
        a = model(image)["logits"]
        b = fresh(image)["logits"]
    report["reload_max_abs_diff"] = float((a - b).abs().max())
    report["reload_ok"] = report["reload_max_abs_diff"] < 1e-6
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["a1_aux", "cf1", "cf0", "u0"])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = {"gates": []}
    for name in args.models:
        print(f"[gate] {name}", flush=True)
        rep = run_one(name, device)
        print(
            f"  params={rep['params']['total']} losses={rep['losses']} "
            f"cond_sensitive={rep.get('cond_sensitive')} reload_ok={rep['reload_ok']}",
            flush=True,
        )
        out["gates"].append(rep)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_dir = resolve_path("outputs/segformer/coarse_to_fine")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gates.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"[saved] {out_dir / 'gates.json'}")


if __name__ == "__main__":
    main()
