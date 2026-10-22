"""U-TAE 配对评测：best / last × VAL / TEST × 两套像素口径。

计划要求「统一评测 best 和 last，提供 valid 与全像素两套指标」。
本脚本对指定 mode 的 run 产生 ``eval_{tag}.json``，字段与 SegFormer 评测保持同名，
便于汇总脚本复用。

用法::

    python scripts/evaluate_utae.py --mode single --checkpoints best,last --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
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

from landslide_seg.config import resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import load_stats, read_manifest  # noqa: E402
from landslide_seg.data.temporal_dataset import TemporalPairDataset, collate  # noqa: E402
from landslide_seg.models.utae_seg import UTAESeg  # noqa: E402
from landslide_seg.training.metrics import confusion, summarize  # noqa: E402
from landslide_seg.training.segformer_metrics import label_objects, pixel_average_precision  # noqa: E402

BINS = [("1_4", 1, 4), ("5_9", 5, 9), ("10_19", 10, 19), ("20_49", 20, 49),
        ("50_199", 50, 199), ("ge200", 200, 10 ** 9)]
STATS = "manifests/official_single_geo/stats_dem.json"
TRAINVAL = "manifests/official_single/trainval.jsonl"
TEST = "manifests/official_single/test.jsonl"
THRESH = 0.5


def bin_of(a: int) -> str:
    for n, lo, hi in BINS:
        if lo <= a <= hi:
            return n
    return "ge200"


@torch.no_grad()
def run(model, loader, device) -> dict:
    model.eval()
    c_valid, c_all, by_event = [], [], {}
    pv, gv = [], []                       # 全量概率/标签，用于 pooled AP
    ev_prob = {}                          # 逐事件概率/标签，用于逐事件 AP
    scale = {b: {"n": 0, "hit": 0} for b, _, _ in BINS}
    for batch in loader:
        logits = model(batch["image"].to(device), batch["dates"].to(device))["logits"].float()
        prob = logits[:, 0].sigmoid().cpu().numpy()
        gt = batch["mask"].numpy().astype(bool)
        va = batch["valid"].numpy().astype(bool)
        for i, ev in enumerate(batch["event_id"]):
            cv = confusion(prob[i] >= THRESH, gt[i], va[i])
            ca = confusion(prob[i] >= THRESH, gt[i], np.ones_like(va[i]))
            c_valid.append(cv); c_all.append(ca)
            by_event.setdefault(ev, []).append(cv)
            pv.append(prob[i][va[i]]); gv.append(gt[i][va[i]])
            ev_prob.setdefault(ev, []).append((prob[i][va[i]], gt[i][va[i]]))
            pred = (prob[i] >= THRESH) & va[i]
            lab, n = label_objects(gt[i] & va[i], 2)
            if n:
                areas = np.bincount(lab.ravel(), minlength=n + 1)[1:]
                hit = np.bincount(lab[pred].ravel(), minlength=n + 1)[1:]
                for j, a in enumerate(areas):
                    b = bin_of(int(a))
                    scale[b]["n"] += 1
                    scale[b]["hit"] += int(hit[j] > 0)
    sv, sa = summarize(c_valid), summarize(c_all)
    sv["ap"] = float(pixel_average_precision(np.concatenate(pv), np.concatenate(gv)))
    per_event = {ev: summarize(cs) for ev, cs in sorted(by_event.items())}
    for ev, pairs in ev_prob.items():
        per_event[ev]["ap"] = float(pixel_average_precision(
            np.concatenate([x[0] for x in pairs]), np.concatenate([x[1] for x in pairs])))
    ev_ious = [v["iou"] for v in per_event.values() if v["iou"] is not None]
    return {"n": len(c_valid), "valid_scope": sv, "all_pixels": sa,
            "event_mean_iou": float(np.mean(ev_ious)) if ev_ious else None,
            "by_event": {k: {"iou": v["iou"], "ap": v.get("ap")} for k, v in per_event.items()},
            "scale_recall": {b: (scale[b]["hit"] / scale[b]["n"] if scale[b]["n"] else None)
                             for b, _, _ in BINS},
            "scale_n": {b: scale[b]["n"] for b, _, _ in BINS}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("single", "multi"), required=True)
    ap.add_argument("--checkpoints", default="best,last")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    run_dir = ROOT / "outputs" / "utae_pair" / f"utae_{args.mode}" / "1234"
    stats = load_stats(resolve_path(STATS))

    rows = read_manifest(resolve_path(TRAINVAL))
    val = [r for r in rows if r["split"] == "val"]
    test = read_manifest(resolve_path(TEST))
    loaders = {
        "val": DataLoader(TemporalPairDataset(val, stats, mode=args.mode), batch_size=16,
                          shuffle=False, num_workers=0, collate_fn=collate),
        "test": DataLoader(TemporalPairDataset(test, stats, mode=args.mode), batch_size=16,
                           shuffle=False, num_workers=0, collate_fn=collate),
    }
    model = UTAESeg(in_channels=11, num_classes=1).to(dev)
    out = {}
    for ck in [c.strip() for c in args.checkpoints.split(",") if c.strip()]:
        p = run_dir / f"{ck}.pt"
        if not p.is_file():
            print(f"[skip] {p}")
            continue
        model.load_state_dict(torch.load(p, map_location=dev)["model"])
        out[ck] = {}
        for split, loader in loaders.items():
            res = run(model, loader, dev)
            out[ck][split] = res
            tag = f"{ck}_{split}"
            (run_dir / "eval").mkdir(exist_ok=True)
            (run_dir / "eval" / f"eval_{tag}.json").write_text(
                json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"[{args.mode}] {tag:11s} valid IoU {res['valid_scope']['iou']:.4f} | "
                  f"全像素 IoU {res['all_pixels']['iou']:.4f} | AP {res['valid_scope']['ap']:.4f} | "
                  f"事件均值 {res['event_mean_iou']:.4f}")
    (run_dir / "eval_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
