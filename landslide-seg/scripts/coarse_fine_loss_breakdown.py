"""粗/细损失分项与参数组 LR 记录（best checkpoint，val）。

计划 §4 要求分别记录 coarse BCE/Dice、fine BCE/Dice、总损失与各参数组 LR。
训练日志已记 total 与 coarse；这里在 best checkpoint 上补齐 BCE/Dice 分项与 LR 快照。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch  # noqa: E402

from landslide_seg.config import load_segformer_config, resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import (  # noqa: E402
    S2SegDataset,
    collate,
    filter_split,
    load_stats,
    read_manifest,
)
from landslide_seg.training.losses import bce_loss, dice_loss  # noqa: E402
from landslide_seg.training.segformer_trainer import load_checkpoint  # noqa: E402


def main() -> None:
    cfg = load_segformer_config(
        ROOT / "configs" / "segformer" / "coarse_to_fine" / "base.yaml"
    )["segformer"]
    root = resolve_path(cfg["output_dir"]) / cfg["data"]["version"]
    records = filter_split(read_manifest(resolve_path(cfg["data"]["manifest"])), "val")
    stats = load_stats(resolve_path(cfg["data"]["stats"]))
    ds = S2SegDataset(records, stats, preload=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cw = 0.5

    out = {}
    for exp in ("A1", "A1_AUX", "CF1", "CF0", "CF1_B", "U0"):
        ckpt_path = root / exp / "1234" / "best.pt"
        if not ckpt_path.is_file():
            continue
        model, ck = load_checkpoint(ckpt_path, device=str(device))
        model.eval()
        sums = {k: 0.0 for k in ("fine_bce", "fine_dice", "coarse_bce", "coarse_dice")}
        n = 0
        with torch.no_grad():
            for i in range(0, len(ds), 32):
                b = collate([ds[j] for j in range(i, min(i + 32, len(ds)))])
                o = model(b["image"].to(device))
                mask, valid = b["mask"].to(device), b["valid"].to(device)
                fine = o["logits"].float()[:, 0]
                base = o["base_logits"].float()[:, 0]
                bs = mask.size(0)
                sums["fine_bce"] += float(bce_loss(fine, mask, valid)) * bs
                sums["fine_dice"] += float(dice_loss(fine, mask, valid)) * bs
                if base is not o["logits"] and exp != "U0":
                    sums["coarse_bce"] += float(bce_loss(base, mask, valid)) * bs
                    sums["coarse_dice"] += float(dice_loss(base, mask, valid)) * bs
                n += bs
        vals = {k: round(v / max(n, 1), 5) for k, v in sums.items()}
        fine_total = 0.5 * vals["fine_bce"] + 0.5 * vals["fine_dice"]
        if exp == "U0":
            total = fine_total
            vals["coarse_bce"] = vals["coarse_dice"] = None
        else:
            coarse_total = 0.5 * vals["coarse_bce"] + 0.5 * vals["coarse_dice"]
            total = fine_total + cw * coarse_total
        groups = ck.get("optimizer", {}).get("param_groups", [])
        lrs = [round(float(g["lr"]), 8) for g in groups]
        out[exp] = {
            **vals,
            "fine_total": round(fine_total, 5),
            "total": round(total, 5),
            "coarse_weight": 0.0 if exp == "U0" else cw,
            "best_epoch": ck.get("epoch"),
            "lrs_at_best": lrs,
        }
        print(exp, out[exp], flush=True)

    out_dir = resolve_path("outputs/segformer/coarse_to_fine")
    (out_dir / "loss_breakdown.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    line = ["# 粗/细损失分项（best checkpoint, val, threshold-free）", "",
            "| 组 | fine BCE | fine Dice | coarse BCE | coarse Dice | total | best epoch | LR 组 |",
            "|---|---:|---:|---:|---:|---:|---:|---|"]
    for exp, r in out.items():
        line.append(
            f"| {exp} | {r['fine_bce']} | {r['fine_dice']} | {r['coarse_bce']} | {r['coarse_dice']} | "
            f"{r['total']} | {r['best_epoch']} | {r['lrs_at_best']} |"
        )
    (out_dir / "loss_breakdown.md").write_text("\n".join(line), encoding="utf-8")
    print("\n".join(line))


if __name__ == "__main__":
    main()
