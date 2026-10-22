"""检查同源监督评分头在 DEV 上的表现：预测的块比例 vs 真实块比例。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from landslide_seg.data.segformer_dataset import S2SegDataset, load_stats, read_manifest  # noqa: E402
from landslide_seg.models.terrain import TerrainEncoder, TerrainScoreHead  # noqa: E402

BLOCK = 16


def main() -> None:
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ROOT / "weights/terrain/v4r1raw_scoring.pt", map_location="cpu", weights_only=False)
    enc = TerrainEncoder(2, pretrained=False).to(dev)
    enc.load_state_dict(ck["encoder"])
    head = TerrainScoreHead(enc.out_channels).to(dev)
    head.load_state_dict(ck["head"])
    enc.eval(), head.eval()

    stats = load_stats(ROOT / "manifests/v4r1/stats_geoboth_raw.json")
    rows = [r for r in read_manifest(ROOT / "manifests/v4r1/pool_100.jsonl") if r["split"] == "val"]
    ds = S2SegDataset(rows[:200], stats, extra="both", sidecar="data/v4r1_geo")

    preds, targets, weights = [], [], []
    with torch.no_grad():
        for start in range(0, len(ds), 32):
            idx = list(range(start, min(start + 32, len(ds))))
            x = torch.stack([ds[i]["image"] for i in idx])
            m = torch.stack([ds[i]["mask"] for i in idx]).to(dev).unsqueeze(1)
            v = torch.stack([ds[i]["valid"] for i in idx]).to(dev).unsqueeze(1)
            score = head(enc(x[:, 10:].to(dev)), (128, 128))
            prob = F.avg_pool2d(torch.sigmoid(score) * v, BLOCK).squeeze(1)
            w = F.avg_pool2d(v.float(), BLOCK).squeeze(1)
            tgt = F.avg_pool2d(m * v, BLOCK).squeeze(1)
            keep = w > 0.99
            preds.append(prob[keep].cpu().numpy())
            targets.append(tgt[keep].cpu().numpy())
    p = np.concatenate(preds)
    t = np.concatenate(targets)
    report = {
        "weights": "weights/terrain/v4r1raw_scoring.pt",
        "split": "DEV (hiroshima+thrissur), 200 tiles",
        "n_blocks": int(len(p)),
        "block_mae": round(float(np.abs(p - t).mean()), 4),
        "pearson": round(float(np.corrcoef(p, t)[0, 1]), 4),
        "positive_block_rate": round(float((t > 0).mean()), 4),
        "pred_mean": round(float(p.mean()), 4),
        "target_mean": round(float(t.mean()), 4),
        "mean_pred_on_positive_blocks": round(float(p[t > 0].mean()), 4) if (t > 0).any() else None,
        "mean_pred_on_empty_blocks": round(float(p[t == 0].mean()), 4) if (t == 0).any() else None,
    }
    out = ROOT / "outputs/segformer/s2_10band_v4r1_geo_raw/scoring_head_eval.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
