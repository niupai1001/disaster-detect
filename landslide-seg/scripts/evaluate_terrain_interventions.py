"""地形分支的非重训干预诊断：同权重下关分支、改融合系数、打乱/平均地形。

只前向、不更新参数，用已验收的地形评分迁移分割权重回答“地形到底贡献了什么”。
正常条件必须与模型自身 forward 逐元素一致，防止诊断公式与实现漂移。
这是同源标注下的贡献检查，不是独立易发性或概率校准。
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
from landslide_seg.config import file_sha256, resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import (  # noqa: E402
    S2SegDataset, collate, filter_split, load_stats, read_manifest,
)
from landslide_seg.training.segformer_metrics import (  # noqa: E402
    evaluate_tile_v2, pixel_average_precision, summarize_by_event_v2, summarize_tiles_v2,
)
from landslide_seg.training.segformer_trainer import load_checkpoint  # noqa: E402

CONDITIONS = ("normal", "alpha0", "fixed_alpha_0.01", "shuffled_terrain", "mean_terrain")


def _logits(model, image, condition, gen):
    xv, xt = image[:, :model.n_visual], image[:, model.n_visual:]
    out = model.visual(xv)
    base, size = out["logits"], out["logits"].shape[-2:]
    if condition == "alpha0":
        return base
    if condition == "shuffled_terrain":
        xt = xt[torch.randperm(xt.shape[0], generator=gen, device=xt.device)]
    elif condition == "mean_terrain":
        xt = xt.mean(0, keepdim=True).expand_as(xt)
    alpha = 0.01 if condition == "fixed_alpha_0.01" else float(model.alpha)
    return base + alpha * torch.tanh(model.head(model.terrain(xt), size))


def _metrics(model, ds, cfg, device, condition, seed):
    s = cfg["segformer"]
    threshold = float(s["eval"]["threshold"])
    thresholds_px = [float(x) * 100.0 for x in s["eval"].get("scale_thresholds_ha", [0.5, 2.0])]
    gen = torch.Generator(device=device).manual_seed(seed)
    rows, probs, gts = [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ds), 32):
            batch = collate([ds[i] for i in range(start, min(start + 32, len(ds)))])
            image = batch["image"].to(device)
            logits = _logits(model, image, condition, gen)
            if condition == "normal":
                ref = model(image)["logits"]
                if not torch.allclose(logits, ref, atol=1e-5):
                    raise RuntimeError("正常条件与模型 forward 不一致：诊断公式已漂移")
            prob = logits[:, 0].sigmoid().cpu().numpy()
            gt = batch["mask"].numpy().astype(bool)
            valid = batch["valid"].numpy().astype(bool)
            for i, sid in enumerate(batch["sample_id"]):
                rows.append(evaluate_tile_v2(
                    prob[i], gt[i], valid[i], threshold=threshold, thresholds_px=thresholds_px,
                    iou_threshold=float(s["eval"]["object_iou_threshold"]),
                    iou_strict=float(s["eval"]["object_iou_strict"]),
                    boundary_tol=int(s["eval"]["boundary_tol_px"]),
                    sample_id=sid, event_id=batch["event_id"][i]))
                probs.append(prob[i][valid[i]])
                gts.append(gt[i][valid[i]])
    overall = summarize_tiles_v2(rows)
    overall["pixel"]["average_precision"] = float(pixel_average_precision(
        np.concatenate(probs), np.concatenate(gts)))
    by_event = summarize_by_event_v2(rows)
    tiny = overall["objects"].get("fine_bins", {}).get("1_4", {})
    return {"n_samples": len(rows), "pixel": overall["pixel"], "objects": overall["objects"],
            "boundary": overall["boundary"], "by_event": by_event,
            "event_mean_iou": float(np.mean([v["pixel"]["iou"] for v in by_event.values()])),
            "tiny_1_4": {"recall": tiny.get("object_recall_evaluable"),
                         "n": tiny.get("n_objects_evaluable")}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ckpt_path = resolve_path(args.checkpoint)
    device = torch.device(args.device)
    model, ckpt = load_checkpoint(ckpt_path, device=args.device)
    if not hasattr(model, "terrain"):
        raise SystemExit("该 checkpoint 没有地形分支")
    cfg = ckpt["config"]
    s = cfg["segformer"]
    stats = load_stats(resolve_path(s["data"]["stats"]))
    records = filter_split(read_manifest(resolve_path(s["data"]["manifest"])), args.split)
    ds = S2SegDataset(records, stats, augment=False, preload=False,
                      extra=str(s["model"].get("extra_channel", "none")),
                      sidecar=s["data"].get("extra_sidecar"))
    out = {"checkpoint": str(ckpt_path).replace("\\", "/"),
           "checkpoint_sha256": file_sha256(ckpt_path),
           "split": args.split, "n_records": len(records),
           "alpha": float(model.alpha), "alpha_learnable": bool(model.alpha_learnable),
           "terrain_init": getattr(model, "terrain_init", None),
           "conditions": {}}
    for condition in CONDITIONS:
        res = _metrics(model, ds, cfg, device, condition, seed=int(s["seed"]))
        out["conditions"][condition] = res
        print(f"[{condition}] mean={res['event_mean_iou']:.4f} "
              f"AP={res['pixel']['average_precision']:.4f} "
              f"objRec={res['objects']['object_recall_evaluable']:.4f}", flush=True)
    base = out["conditions"]["normal"]
    out["delta"] = {c: {"event_mean_iou": out["conditions"][c]["event_mean_iou"] - base["event_mean_iou"],
                        "average_precision": out["conditions"][c]["pixel"]["average_precision"]
                        - base["pixel"]["average_precision"]}
                    for c in CONDITIONS if c != "normal"}
    out["claim_limit"] = ("同源分割标注下的贡献检查；不更新参数，不是独立易发性标签，"
                          "也不证明迁移正确性")
    path = resolve_path(args.out or "outputs/research_completion/terrain_interventions.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"alpha": out["alpha"], "delta": out["delta"]}, ensure_ascii=False, indent=1))
    print(path)


if __name__ == "__main__":
    main()
