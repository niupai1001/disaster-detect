"""真实数据上的完整模型推理（Sen12Landslides + B01/B09 回补）。

这是 M1 第 8 项（"真实有效小批量"）的验证脚本。每一步都记录 provenance：

1. Sen12 提供 B02…B12（含 B8A）10 波段 + MASK + SCL + 日期 + 坐标系；
2. 用 STAC 在**同一天**检索到同一 L2A 产品，回补 B01/B09（60m→10m，注明是插值）；
3. 组装 12 波段，走完整模型（TerraMind TiM + AnySat + 语义 + 粗细两阶段）；
4. 与 GT 掩膜比较，输出可视化与 JSON。

对齐校验：把 STAC 的 B08 与 Sen12 的 B08 做相关，确认两个来源落在同一网格上。

用法:
    python scripts/real_inference.py --root data/sen12 --limit 3
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

from landslide_seg.data.band_mapping import S2_L2A_BANDS, normalize_anysat  # noqa: E402
from landslide_seg.data.s2_backfill import backfill_b01_b09, read_asset_window, find_same_day_item  # noqa: E402
from landslide_seg.data.sen12 import list_sen12_files, load_sen12  # noqa: E402
from landslide_seg.inference.visualize import save_panel  # noqa: E402
from landslide_seg.training.metrics import confusion, summarize  # noqa: E402


def align_check(sample, index: int) -> dict:
    """用 STAC 的 B08 与 Sen12 的 B08 相关性验证空间对齐。"""
    import datetime as _dt

    from pyproj import Transformer

    date = sample.attrs["_time_ref"] + _dt.timedelta(days=int(sample.time[index]))
    tr = Transformer.from_crs(sample.attrs.get("crs"), "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(sample.attrs["center_x"], sample.attrs["center_y"])
    item = find_same_day_item(lon, lat, date)
    if item is None:
        return {"available": False}
    half = 128 * 10.0 / 2
    cx, cy = sample.attrs["center_x"], sample.attrs["center_y"]
    bounds = (cx - half, cy - half, cx + half, cy + half)
    ref = read_asset_window(item["assets"]["nir"]["href"], bounds, sample.attrs["crs"], (128, 128))
    ours = sample.bands["B08"][index].astype(np.float32)
    ok = (ref > 0) & (ours > 0)
    corr = float(np.corrcoef(ref[ok], ours[ok])[0, 1]) if ok.sum() > 100 else float("nan")
    return {
        "available": True,
        "item_id": item["id"],
        "corr_B08": round(corr, 4),
        "n_common_px": int(ok.sum()),
        "sen12_mean": round(float(ours[ok].mean()), 1),
        "stac_mean": round(float(ref[ok].mean()), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data" / "sen12"))
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = list_sen12_files(args.root, limit=args.limit)
    print(f"[env] device={device}  files={len(files)}")

    from landslide_seg.models.build import build_model

    model = build_model(device=device)
    model.eval()

    out_dir = ROOT / "outputs"
    (out_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    report = {"device": device, "samples": [], "metrics": None}

    counters, ids = [], []
    for p in files:
        s = load_sen12(p)
        pre, post = s.event_index()
        if s.mask[post].sum() == 0:
            print(f"[skip] {p.name}: post 时相无滑坡像素")
            continue

        bf = backfill_b01_b09(s, post)
        if bf is None:
            print(f"[skip] {p.name}: 同一天无 L2A 产品，拒绝跨日替代")
            continue

        # 组装 12 波段（DN）
        arr12 = np.zeros((12, 128, 128), dtype=np.float32)
        for i, name in enumerate(S2_L2A_BANDS):
            if name in bf.bands:
                arr12[i] = bf.bands[name]
            else:
                arr12[i] = s.bands[name][post].astype(np.float32)
        x10 = np.stack([s.bands[b][post] for b in
                        ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")]
                       ).astype(np.float32)

        valid = s.valid_mask(post).copy()
        # B01/B09 有 nodata（窗口越界）的像素标记为无效，而不是当 0 用
        gap = (bf.bands["B01"] <= 0) | (bf.bands["B09"] <= 0)
        valid &= ~gap

        gt = s.mask[post].astype(np.int64)
        x12_t = torch.from_numpy(arr12)[None].to(device)
        x10_t = torch.from_numpy(x10)[None].to(device)
        dates = torch.tensor([[float(s.doy(post))]], device=device)

        with torch.no_grad():
            pred = model.predict(x12_t, x10_t, dates, threshold=args.threshold)

        prob = pred["prob"][0, 0].cpu().numpy()
        pmask = (prob > args.threshold).astype(np.uint8)
        if float(pred["presence"][0, 0]) <= args.threshold:
            pmask = np.zeros_like(pmask)
        c = confusion(pmask, gt, valid)
        counters.append(c)
        ids.append(p.name)

        ac = align_check(s, post)
        rec = {
            "file": p.name,
            "event_date": s.attrs.get("event_date"),
            "post_date": (s.attrs["_time_ref"].isoformat()
                          if s.attrs.get("_time_ref") else None),
            "backfill": {
                "item_id": bf.item_id,
                "datetime": bf.datetime,
                "cloud_cover": bf.cloud_cover,
                "method": bf.method,
                "gap_ratio": round(float(gap.mean()), 4),
            },
            "align_check": ac,
            "gt_px": int(gt.sum()),
            "pred_px": int(pmask.sum()),
            "valid_ratio": round(float(valid.mean()), 4),
            "confusion": c,
            "presence": round(float(pred["presence"][0, 0]), 4),
        }
        report["samples"].append(rec)
        print(f"[real] {p.name} gt={rec['gt_px']} pred={rec['pred_px']} "
              f"iou={c['tp']/max(c['tp']+c['fp']+c['fn'],1):.4f} "
              f"align_corr={ac.get('corr_B08')} presence={rec['presence']}")

        rgb = np.stack([arr12[3], arr12[2], arr12[1]], -1)
        rgb = (rgb - np.percentile(rgb, 2)) / max(
            np.percentile(rgb, 98) - np.percentile(rgb, 2), 1e-6)
        coarse_prob = pred["coarse_prob"][0, 0].cpu().numpy()
        save_panel(
            out_dir / "visualizations" / f"real_{p.stem}.png",
            np.clip(rgb, 0, 1),
            gt,
            coarse_prob,          # 粗阶段概率（固定 0-1 显示）
            prob,                 # 细阶段概率
            pmask,                # 阈值化结果
            valid=valid,
            scl=s.scl[post],
            title=f"REAL {p.name} | {bf.item_id} | IoU={c['tp']/max(c['tp']+c['fp']+c['fn'],1):.3f}",
            presence=float(pred["presence"][0, 0]),
        )

    if counters:
        report["metrics"] = summarize(counters)
        print("[metrics]", json.dumps(report["metrics"], indent=1))
        print("[note] 未训练权重下的指标没有科学意义，此步只证明链路可用")

    path = out_dir / "logs" / "real_inference.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
