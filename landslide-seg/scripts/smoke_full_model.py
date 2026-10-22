"""M1 完整模型闭环验证：前向 → 损失 → 反向 → 保存 → 重载 → 推理 → 可视化。

数据来源：**合成 12/10 波段张量**（`pipeline.make_synthetic_batch`），
因为真实 L4S 数据缺少 B8A，无法合法构造 TiM/AnySat 输入（见 docs/blockers.md）。

因此本脚本通过只能证明：
- 三个官方权重真实加载；
- 双骨干 → 融合 → 语义 → 粗细两阶段 的整条数据流接通；
- 允许训练的参数确实收到梯度并更新；
- checkpoint 可保存/重载且推理结果一致。

它**不能**证明任何精度或科学结论。

用法:
    python scripts/smoke_full_model.py --steps 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from landslide_seg.data.pipeline import make_synthetic_batch  # noqa: E402
from landslide_seg.inference.visualize import rgb_from_bands, save_panel  # noqa: E402
from landslide_seg.models.build import build_model  # noqa: E402
from landslide_seg.training.losses import CombinedLoss  # noqa: E402
from landslide_seg.training.trainer import Trainer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--tim", action="store_true", default=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = ROOT / "outputs"
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"[env] device={device}")

    t0 = time.perf_counter()
    model = build_model(device=device, tim_enabled=args.tim)
    print(f"[build] {time.perf_counter() - t0:.1f}s")
    pr = model.parameter_report()
    print(f"[params] total={pr['total']/1e6:.1f}M trainable={pr['trainable']/1e6:.1f}M")
    for k, v in pr["groups"].items():
        print(f"   {k:<10} {v['params']/1e6:8.2f}M  trainable={v['trainable']/1e6:6.2f}M")

    trainer = Trainer(
        model,
        CombinedLoss(lambda_coarse=0.5, lambda_presence=0.2, lambda_region_text=0.1),
        lr=1e-4,
        device=device,
        out_dir=out_dir,
    )

    batch = make_synthetic_batch(args.batch_size, args.size, seed=0, device=device)

    # --- 1) 前向 + 形状记录 -------------------------------------------
    model.eval()
    with torch.no_grad():
        out = model(batch.x12, batch.x10, batch.s2_dates, return_features=True)
    shapes = {
        "input_x12": list(batch.x12.shape),
        "input_x10": list(batch.x10.shape),
        "s2_dates": list(batch.s2_dates.shape),
        "fine_logits": list(out.fine_logits.shape),
        "coarse_logits": list(out.coarse_logits.shape),
        "presence_logits": list(out.presence_logits.shape),
        "sem_map": list(out.sem_map.shape),
        "region_embedding": list(out.region_embedding.shape) if out.region_embedding is not None else None,
        "pyramid": [list(p.shape) for p in out.features["pyramid"]],
        "source_channels": out.features["channels"],
        "fusion_gate": model.fusion.gate_report(),
    }
    if "terramind_info" in out.features["extras"]:
        info = out.features["extras"]["terramind_info"]
        shapes["terramind"] = {
            "tim_ran": info.tim_ran,
            "num_tokens": info.num_tokens,
            "feature_shapes": [list(s) for s in info.feature_shapes],
            "latency_s": info.tim_latency_s,
        }
    for k, v in out.features["extras"].items():
        if k.startswith("anysat_info@"):
            shapes[k] = {
                "patch_size_m": v.patch_size_m,
                "out_shape": list(v.out_shape),
                "num_patches": v.num_patches,
                "latency_s": v.latency_s,
            }
    print("[shapes]")
    print(json.dumps(shapes, indent=2, ensure_ascii=False))

    # --- 2) 反向 + 参数更新 -------------------------------------------
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    before = {n: p.detach().clone() for n, p in trainable[:8]}
    logs = []
    for i in range(args.steps):
        log = trainer.train_step(batch)
        logs.append(log)
        loss = log["loss"]
        print(
            f"[step {i+1}] total={loss['total']:.4f} fine={loss['fine']:.4f} "
            f"coarse={loss['coarse']:.4f} presence={loss['presence']:.4f} "
            f"region_text={loss['region_text']:.4f} "
            f"(applied={loss['applied_region_text']}) "
            f"lat={log['latency_s']:.2f}s"
        )
    changed = sum(
        1 for n, p in trainable[:8] if not torch.allclose(before[n], p.detach(), atol=1e-9)
    )
    print(f"[update check] {changed}/{len(before)} 个被抽样参数发生了更新")

    frozen_ok = True
    for name, mod in (("terramind", model.terramind), ("anysat", model.anysat)):
        if mod is None:
            continue
        n_train = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        print(f"[frozen check] {name}: trainable={n_train}")
        frozen_ok &= n_train == 0

    # --- 3) 保存 / 重载 / 推理 ----------------------------------------
    ckpt = trainer.save(
        out_dir / "checkpoints" / "m1_smoke.pt",
        extra={"shapes": shapes, "note": "M1 interface smoke test, synthetic input"},
    )
    print(f"[ckpt] {ckpt} ({ckpt.stat().st_size/1e6:.1f} MB)")

    model2 = build_model(device=device, tim_enabled=args.tim)
    trainer2 = Trainer(model2, CombinedLoss(), lr=1e-4, device=device, out_dir=out_dir)
    trainer2.load(ckpt)

    # 严谨对比：eval 模式 + 固定随机种子（TiM 采样本身是随机的）+ 关闭分支丢弃
    import random as _random

    def _fixed_forward(m):
        m.eval()
        torch.manual_seed(1234)
        _random.seed(1234)
        with torch.no_grad():
            return m(
                batch.x12, batch.x10, batch.s2_dates,
                return_features=False, 
            ).fine_logits

    # 分支丢弃只在 training=True 时生效，eval 下两条路径一致
    logits_a = _fixed_forward(model)
    logits_b = _fixed_forward(model2)
    max_diff = (logits_a - logits_b).abs().max().item()
    print(f"[reload check] eval+固定种子下 max|Δlogits| = {max_diff:.3e}")

    # 参数级校验：可训练部分逐张量比对
    skip = ("terramind.", "anysat.", "semantic.text.")
    sd1 = {k: v for k, v in model.state_dict().items() if not k.startswith(skip)}
    sd2 = {k: v for k, v in model2.state_dict().items() if not k.startswith(skip)}
    max_w = max(
        (sd1[k].float() - sd2[k].float()).abs().max().item() for k in sd1
    ) if sd1 else 0.0
    print(f"[reload check] 非骨干权重 max|Δ| = {max_w:.3e} ({len(sd1)} 个张量)")

    pred = model2.predict(batch.x12, batch.x10, batch.s2_dates, threshold=0.5)
    print(f"[predict] prob={tuple(pred['prob'].shape)} "
          f"presence={pred['presence'].flatten().tolist()}")

    # --- 4) 可视化 -----------------------------------------------------
    vis_paths = []
    for i in range(args.batch_size):
        rgb = rgb_from_bands(batch.x12[i].detach().cpu().numpy())
        gt = batch.mask[i].detach().cpu().numpy()
        coarse = pred["coarse_prob"][i, 0].detach().cpu().numpy()
        fine = pred["mask"][i, 0].detach().cpu().numpy()
        p = save_panel(
            out_dir / "visualizations" / f"m1_smoke_{i}.png",
            rgb, gt, coarse, fine, (fine > 0.5).astype(float),
            title=f"SYNTHETIC {batch.sample_ids[i]} (not real Sentinel-2) "
                  f"lat={logs[-1]['latency_s']:.2f}s",
            presence=float(pred["presence"][i, 0]),
        )
        vis_paths.append(str(p))
        print(f"[vis] {p}")

    report = {
        "status": "SMOKE_ONLY",
        "device": device,
        "synthetic_input": True,
        "warning": "输入为合成张量，不能作为滑坡分割有效性证据；真实数据验收见 docs/blockers.md",
        "params": pr,
        "shapes": shapes,
        "train_steps": logs,
        "params_updated": f"{changed}/{len(before)}",
        "backbones_frozen": frozen_ok,
        "reload_max_abs_diff": max_diff,
        "checkpoint": str(ckpt),
        "visualizations": vis_paths,
        "total_latency_s": time.perf_counter() - t0,
    }
    path = out_dir / "logs" / "smoke_full_model.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / "docs" / "model_shapes.json").write_text(
        json.dumps(shapes, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
