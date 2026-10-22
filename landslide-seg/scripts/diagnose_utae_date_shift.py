"""诊断：多帧/截至日期组在外部事件上塌陷，是否由日期外推造成。

对同一个已冻结 checkpoint，用三种日期输入各评测一次外部集合：

1. 真实日期（原样）；
2. 整体平移，使外部事件落到训练日期范围内；
3. 常数日期（去掉时间信息，只留多帧内容）。

若 (2) 明显恢复而 (1) 塌陷，说明多帧收益的失效更像"日期外推"，而不是
"多帧信息无用"。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from landslide_seg.config import resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import load_stats, read_manifest  # noqa: E402
from landslide_seg.data.temporal_v6 import TemporalV6, collate_v6  # noqa: E402
from landslide_seg.models.utae_dates import UTAERealDates  # noqa: E402
from landslide_seg.training.metrics import confusion, summarize  # noqa: E402

from train_utae_cv import load_cv_config  # noqa: E402


@torch.no_grad()
def run(model, rows, stats, mode, device, transform, threshold=0.5):
    ds = TemporalV6(rows, stats, mode=mode, augment=False, seed=1234, dem=True)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_v6)
    counts, n_px, pos = [], 0, 0
    for b in loader:
        dates = transform(b["dates"])
        logits = model(b["image"].to(device), dates.to(device), b["keep"].to(device),
                       b["anchor"].to(device))["logits"].float()
        prob = logits[:, 0].sigmoid().cpu().numpy()
        gt = b["mask"].numpy().astype(bool)
        va = b["valid"].numpy().astype(bool)
        for i in range(len(prob)):
            counts.append(confusion(prob[i] >= threshold, gt[i], va[i]))
            n_px += int(va[i].sum())
            pos += int(((prob[i] >= threshold) & va[i]).sum())
    s = summarize(counts)
    return {"iou": round(float(s["iou"]), 4), "dice": round(float(s["dice"]), 4),
            "precision": round(float(s["precision"]), 4), "recall": round(float(s["recall"]), 4),
            "positive_rate": round(pos / max(n_px, 1), 5)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", required=True, choices=("single", "multi", "history"))
    ap.add_argument("--checkpoint", default="best")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shift", type=float, default=None,
                    help="平移天数；默认自动把外部事件最大日期挪到训练最大日期")
    args = ap.parse_args()
    cfg = load_cv_config(resolve_path(args.config))
    prot = cfg["protocol"]
    dev = torch.device(args.device)
    pool = read_manifest(resolve_path(prot["pool"]))
    train = [r for r in pool if r["protocol_role"] == prot["roles"]["train"]]
    outer = read_manifest(resolve_path(prot["outer"]))
    stats = load_stats(resolve_path(prot["stats"]))
    day = lambda r: int((np.datetime64(r["date"]) - np.datetime64("2015-12-03")).astype(int))  # noqa: E731
    tmax = max(day(r) for r in train)
    omax = max(day(r) for r in outer)
    shift = args.shift if args.shift is not None else float(omax - tmax)

    run_dir = ROOT / cfg["output_dir"] / cfg["fold"] / args.mode / str(cfg["seed"])
    ckpt = torch.load(run_dir / f"{args.checkpoint}.pt", map_location="cpu", weights_only=False)
    model = UTAERealDates(in_channels=int(cfg["data"]["in_channels"]), num_classes=1).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out = {
        "run": str(run_dir.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint": args.checkpoint, "epoch": int(ckpt["epoch"]),
        "train_max_day": int(tmax), "outer_max_day": int(omax), "shift_days": shift,
        "real_dates": run(model, outer, stats, args.mode, dev, lambda d: d),
        "shifted_dates": run(model, outer, stats, args.mode, dev, lambda d: d - shift),
        "constant_dates": run(model, outer, stats, args.mode, dev, lambda d: torch.zeros_like(d)),
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    p = run_dir / "eval" / f"date_intervention_{args.checkpoint}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
