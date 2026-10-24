"""S4 门控诊断：best 权重在 DEV 上的 gate 分布、alpha、修正幅度与目标/背景/FP 差异。

只读 checkpoint 与 DEV 缓存；不训练。检验学习门是否退化成全图常数，并给出
``outputs/segformer/v4r2_spectral/gate_diagnostics.json`` 与 PNG。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from landslide_seg.config import resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import (  # noqa: E402
    S2SegDataset, collate, filter_split, load_stats, read_manifest,
)
from landslide_seg.training.segformer_trainer import load_checkpoint  # noqa: E402

ARMS = {
    "s4_add": ("S4-ADD", "outputs/segformer/v4r2_spectral/SegFormer-B0-S4Add/1234"),
    "s4_fixed_gate": ("S4-FIXEDGATE", "outputs/segformer/v4r2_spectral/SegFormer-B0-S4FixedGate/1234"),
    "s4_learn_gate": ("S4-LEARNGATE", "outputs/segformer/v4r2_spectral/SegFormer-B0-S4LearnGate/1234"),
    "s4_capctrl": ("S4-CAPCTRL", "outputs/segformer/v4r2_spectral/SegFormer-B0-S4CapCtrl/1234"),
}
OUT = ROOT / "outputs" / "segformer" / "v4r2_spectral" / "gate_diagnostics.json"
FIG = OUT.with_suffix(".png")


def audit(arm: str, checkpoint: Path, device: str) -> dict:
    checkpoint = checkpoint if checkpoint.is_absolute() else (ROOT / checkpoint)
    model, ckpt = load_checkpoint(checkpoint, device=device)
    cfg = ckpt["config"]["segformer"]
    stats = ckpt.get("stats") or load_stats(resolve_path(cfg["data"]["stats"]))
    records = filter_split(read_manifest(resolve_path(cfg["data"]["manifest"])), "val")
    ds = S2SegDataset(records, stats, augment=False, preload=True,
                      extra=str(cfg["model"].get("extra_channel", "none")))
    threshold = float(cfg["eval"]["threshold"])
    dev = next(model.parameters()).device
    model.eval()
    gate_all, corr_all = [], []
    pos, bg, fp, fn = [], [], [], []
    per_image_std = []
    with torch.no_grad():
        for start in range(0, len(ds), 16):
            batch = collate([ds[i] for i in range(start, min(start + 16, len(ds)))])
            out = model(batch["image"].to(dev))
            gate = getattr(model, "last_gate", None)
            corr = getattr(model, "last_corr_mag", None)
            gt = batch["mask"].numpy().astype(bool)
            valid = batch["valid"].numpy().astype(bool)
            pred = out["logits"][:, 0].sigmoid().cpu().numpy() >= threshold
            if corr is not None:
                c = torch.nn.functional.interpolate(corr.float(), size=gt.shape[-2:],
                                                   mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
                corr_all.append(c[valid])
            if gate is not None:
                g = torch.nn.functional.interpolate(gate.float(), size=gt.shape[-2:],
                                                   mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
                gate_all.append(g[valid])
                pos.append(g[valid & gt]); bg.append(g[valid & ~gt])
                fp.append(g[valid & ~gt & pred]); fn.append(g[valid & gt & ~pred])
                for i in range(g.shape[0]):
                    v = g[i][valid[i]]
                    if v.size:
                        per_image_std.append(float(v.std()))
    result = {"arm": arm, "run_dir": str(Path(ARMS[arm][1])), "checkpoint": str(checkpoint.relative_to(ROOT)),
              "alpha": float(model.alpha.detach()) if hasattr(model, "alpha") else None,
              "fusion": getattr(model, "fusion_mode", None),
              "n_dev": len(records)}
    if corr_all:
        c = np.concatenate(corr_all)
        result["corr_abs_mean"] = float(c.mean())
        result["corr_abs_p95"] = float(np.percentile(c, 95))
    if gate_all:
        g = np.concatenate(gate_all)
        result.update({
            "gate_mean": float(g.mean()), "gate_std": float(g.std()),
            "gate_p05": float(np.percentile(g, 5)), "gate_p50": float(np.percentile(g, 50)),
            "gate_p95": float(np.percentile(g, 95)),
            "gate_pos_mean": float(np.concatenate(pos).mean()),
            "gate_bg_mean": float(np.concatenate(bg).mean()),
            "gate_fp_mean": float(np.concatenate(fp).mean()) if sum(x.size for x in fp) else None,
            "gate_fn_mean": float(np.concatenate(fn).mean()) if sum(x.size for x in fn) else None,
            "gate_image_std_mean": float(np.mean(per_image_std)),
            "gate_constant_fraction": float(np.mean(np.abs(g - g.mean()) < 0.02)),
        })
    return result


def figure(results: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gate_arms = [r for r in results if r.get("gate_mean") is not None]
    corr_arms = [r for r in results if r.get("corr_abs_mean") is not None]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if gate_arms:
        x = np.arange(len(gate_arms))
        width = 0.26
        axes[0].bar(x - width, [r["gate_pos_mean"] for r in gate_arms], width, label="GT 滑坡")
        axes[0].bar(x, [r["gate_bg_mean"] for r in gate_arms], width, label="背景")
        axes[0].bar(x + width, [r.get("gate_fp_mean") or 0 for r in gate_arms], width, label="FP")
        axes[0].set_xticks(x, [ARMS[r["arm"]][0] for r in gate_arms], rotation=20)
        axes[0].set_ylabel("gate 均值")
        axes[0].set_title("门控在目标/背景/误检上的均值")
        axes[0].legend(fontsize=8)
        axes[0].grid(axis="y", alpha=0.3)
    if corr_arms:
        axes[1].bar([ARMS[r["arm"]][0] for r in corr_arms], [r["corr_abs_mean"] for r in corr_arms],
                    color="#6b8e23")
        axes[1].set_ylabel("|修正| 均值")
        axes[1].set_title("融合支路修正幅度（越小越接近零贡献）")
        axes[1].tick_params(axis="x", rotation=20)
        axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print("figure", FIG)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--only", nargs="+", default=None)
    args = ap.parse_args()
    results = []
    for arm in (args.only or list(ARMS)):
        ck = ROOT / ARMS[arm][1] / "best.pt"
        if not ck.is_file():
            results.append({"arm": arm, "status": "MISSING_CHECKPOINT"})
            continue
        print("audit", arm, flush=True)
        rec = audit(arm, ck, args.device)
        rec["status"] = "COMPLETED"
        results.append(rec)
    payload = {"scope": "S4 best 权重在 DEV669 上的门控与修正诊断；不训练",
               "threshold": 0.5, "results": results,
               "note": "gate_constant_fraction 是 |gate-全局均值|<0.02 的像素比例；"
                       "接近 1 说明门近似全图常数，不能宣称空间选择性。"}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    figure([r for r in results if r.get("status") == "COMPLETED"])
    print(json.dumps([{k: r.get(k) for k in ("arm", "alpha", "gate_mean", "gate_std",
                                            "gate_pos_mean", "gate_bg_mean", "gate_fp_mean",
                                            "corr_abs_mean", "gate_constant_fraction")}
                      for r in results], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
