"""检查同源监督评分头在 DEV 上的表现：预测的块比例 vs 真实块比例。"""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from landslide_seg.data.segformer_dataset import S2SegDataset, load_stats, read_manifest  # noqa: E402
from landslide_seg.models.terrain import TerrainEncoder, TerrainScoreHead, valid_block_fraction  # noqa: E402
from landslide_seg.config import resolve_path, file_sha256  # noqa: E402
from landslide_seg.training.segformer_metrics import pixel_average_precision  # noqa: E402

BLOCK = 16


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = resolve_path(args.out)
    if out.exists():
        raise FileExistsError(f"不覆盖已有地形评价: {out}")
    dev = torch.device(args.device)
    ckpath = resolve_path(args.checkpoint)
    ck = torch.load(ckpath, map_location="cpu", weights_only=False)
    for key, path in (("stats_sha256", ROOT / "manifests/v4r1/stats_geoboth_raw.json"),
                      ("pool_sha256", ROOT / "manifests/v4r1/pool_100.jsonl")):
        if ck.get(key) != file_sha256(path):
            raise ValueError(f"{key} 与当前协议不符")
    enc = TerrainEncoder(2, pretrained=False).to(dev)
    enc.load_state_dict(ck["encoder"])
    head = TerrainScoreHead(enc.out_channels).to(dev)
    head.load_state_dict(ck["head"])
    enc.eval(), head.eval()

    stats = load_stats(ROOT / "manifests/v4r1/stats_geoboth_raw.json")
    rows = [r for r in read_manifest(ROOT / "manifests/v4r1/pool_100.jsonl") if r["split"] == "val"]
    ds = S2SegDataset(rows, stats, extra="both", sidecar="data/v4r1_geo")

    preds, targets, weights = [], [], []
    with torch.no_grad():
        for start in range(0, len(ds), 32):
            idx = list(range(start, min(start + 32, len(ds))))
            items = [ds[i] for i in idx]
            x = torch.stack([item["image"] for item in items])
            m = torch.stack([item["mask"] for item in items]).to(dev).unsqueeze(1)
            v = torch.stack([item["valid"] for item in items]).to(dev).unsqueeze(1)
            score = head(enc(x[:, 10:].to(dev)), (128, 128))
            prob = torch.sigmoid(F.avg_pool2d(score, BLOCK)).squeeze(1)
            tgt, w = valid_block_fraction(m, v, BLOCK)
            tgt, w = tgt.squeeze(1), w.squeeze(1)
            keep = w > 0
            preds.append(prob[keep].cpu().numpy())
            targets.append(tgt[keep].cpu().numpy())
    p = np.concatenate(preds)
    t = np.concatenate(targets)
    report = {
        "weights": str(ckpath), "checkpoint_sha256": file_sha256(ckpath),
        "target_version": ck.get("target_version", 1),
        "split": "DEV (hiroshima+thrissur), 669 tiles",
        "scope": "同源分割标注的有效区区块占比；不是独立易发性或概率校准",
        "n_tiles": len(ds),
        "n_blocks": int(len(p)),
        "block_mae": round(float(np.abs(p - t).mean()), 4),
        "pearson": round(float(np.corrcoef(p, t)[0, 1]), 4),
        "positive_block_rate": round(float((t > 0).mean()), 4),
        "pred_mean": round(float(p.mean()), 4),
        "pred_std": float(p.std()), "pred_min": float(p.min()), "pred_max": float(p.max()),
        "auprc_known_segmentation_blocks": pixel_average_precision(p, (t > 0).astype(np.uint8)),
        "target_mean": round(float(t.mean()), 4),
        "mean_pred_on_positive_blocks": round(float(p[t > 0].mean()), 4) if (t > 0).any() else None,
        "mean_pred_on_empty_blocks": round(float(p[t == 0].mean()), 4) if (t == 0).any() else None,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
