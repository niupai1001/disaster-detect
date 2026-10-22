"""评估地形自监督编码器：遮挡重建精度，以及特征里是否含有地形/滑坡信息。

三个问题分别用三个指标回答：
1. 遮挡重建：遮住 50% 的 16×16 块，让网络补回高程，与三种基线比较（常数填充、
   图块内可见像素均值、以及"完全不做"的离散度）。
2. 特征是否表示地形：冻结编码器，用线性岭回归从 64 维全局特征预测图块平均高程
   与平均坡度，报告在 DEV 上的 R²。
3. 特征是否与滑坡有关：同样用线性回归预测图块内滑坡像素比例（TRAIN 拟合，DEV 检验），
   报告 R² 与相关系数。这一步只是诊断，不是易发性模型。

用法::

    python scripts/evaluate_terrain_ssl.py --weights weights/terrain/v4r1raw_ssl.pt
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

from landslide_seg.data.segformer_dataset import S2SegDataset, load_stats, read_manifest  # noqa: E402
from landslide_seg.models.terrain import TerrainEncoder  # noqa: E402

BLOCK = 16
GRID = 128 // BLOCK


class ReconstructHead(torch.nn.Module):
    def __init__(self, in_chans: int = 64) -> None:
        super().__init__()
        n = torch.nn
        self.body = n.Sequential(
            n.Conv2d(in_chans, 64, 3, padding=1, bias=False), n.GroupNorm(8, 64), n.GELU(),
            n.Upsample(scale_factor=2),
            n.Conv2d(64, 32, 3, padding=1, bias=False), n.GroupNorm(8, 32), n.GELU(),
            n.Upsample(scale_factor=2),
            n.Conv2d(32, 16, 3, padding=1, bias=False), n.GroupNorm(8, 16), n.GELU(),
            n.Conv2d(16, 1, 1),
        )

    def forward(self, f):
        return self.body(f)


def block_mask(batch: int, device, rng: torch.Generator, ratio: float = 0.5):
    m = torch.rand(batch, 1, GRID, GRID, generator=rng) < ratio
    m = m.repeat_interleave(BLOCK, 2).repeat_interleave(BLOCK, 3)
    return m.to(device)


def ridge_r2(Xtr, ytr, Xte, yte, alpha: float = 1.0):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    A1 = np.concatenate([A, np.ones((len(A), 1))], 1)
    B1 = np.concatenate([B, np.ones((len(B), 1))], 1)
    w = np.linalg.solve(A1.T @ A1 + alpha * np.eye(A1.shape[1]), A1.T @ ytr)
    pred = B1 @ w
    ss_res = float(((yte - pred) ** 2).sum())
    ss_tot = float(((yte - yte.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    corr = float(np.corrcoef(pred, yte)[0, 1]) if len(yte) > 2 else float("nan")
    return {"r2": round(r2, 4), "pearson": round(corr, 4), "n_test": int(len(yte))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/terrain/v4r1raw_ssl.pt")
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-dev", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="outputs/segformer/s2_10band_v4r1_geo_raw/ssl_eval.json")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    stats = load_stats(ROOT / "manifests/v4r1/stats_geoboth_raw.json")
    rows = read_manifest(ROOT / "manifests/v4r1/pool_100.jsonl")
    train_rows = [r for r in rows if r["split"] == "train"]
    dev_rows = [r for r in rows if r["split"] == "val"]
    rng_np = np.random.default_rng(1234)
    train_rows = [train_rows[i] for i in rng_np.choice(len(train_rows), min(args.n_train, len(train_rows)), replace=False)]
    dev_rows = [dev_rows[i] for i in rng_np.choice(len(dev_rows), min(args.n_dev, len(dev_rows)), replace=False)]

    ds_tr = S2SegDataset(train_rows, stats, extra="both", sidecar="data/v4r1_geo")
    ds_dev = S2SegDataset(dev_rows, stats, extra="both", sidecar="data/v4r1_geo")

    ck = torch.load(ROOT / args.weights, map_location="cpu", weights_only=False)
    enc = TerrainEncoder(2, pretrained=False).to(dev)
    enc.load_state_dict(ck["encoder"])
    dec = ReconstructHead(enc.out_channels).to(dev)
    dec.load_state_dict(ck["decoder"])
    enc.eval(), dec.eval()

    dem_mean = float(stats["mean"][10])
    dem_std = float(stats["std"][10])

    def batch(ds, idx):
        items = [ds[i] for i in idx]
        x = torch.stack([it["image"] for it in items])
        m = torch.stack([it["mask"] for it in items])
        v = torch.stack([it["valid"] for it in items])
        return x, m, v

    # ---- 1) 遮挡重建 ----
    rng = torch.Generator().manual_seed(7)
    abs_err, base_const, base_visible, base_spread, tiles = [], [], [], [], []
    with torch.no_grad():
        for start in range(0, len(ds_dev), args.batch):
            idx = list(range(start, min(start + args.batch, len(ds_dev))))
            x, m, v = batch(ds_dev, idx)
            terrain = x[:, 10:].to(dev)
            mask = block_mask(len(idx), dev, rng)
            recon = dec(enc(terrain * (~mask).float())).cpu()[:, 0]
            dem_norm = x[:, 10]
            dem_m = dem_norm * dem_std + dem_mean
            recon_m = recon * dem_std + dem_mean
            sel = mask.cpu()[:, 0] & v
            abs_err.append((recon_m - dem_m)[sel].abs())
            base_const.append((dem_mean - dem_m)[sel].abs())
            vis = (~mask.cpu()[:, 0]) & v
            tile_mean = (dem_m * vis).sum((1, 2)) / vis.sum((1, 2)).clamp_min(1)
            base_visible.append((tile_mean[:, None, None] - dem_m)[sel].abs())
            base_spread.append((dem_m[sel] - dem_m[sel].mean()).abs())

    def cat(xs):
        return float(torch.cat([t.flatten() for t in xs]).mean())

    recon_mae = cat(abs_err)
    const_mae = cat(base_const)
    visible_mae = cat(base_visible)
    spread_mae = cat(base_spread)
    rel = recon_mae / max(spread_mae, 1e-6)

    # ---- 2/3) 线性探针 ----
    def feats_targets(ds):
        F, elev, slope, frac = [], [], [], []
        with torch.no_grad():
            for start in range(0, len(ds), args.batch):
                idx = list(range(start, min(start + args.batch, len(ds))))
                x, m, v = batch(ds, idx)
                f = enc(x[:, 10:].to(dev)).mean((2, 3)).cpu().numpy()
                F.append(f)
                dem_m = (x[:, 10] * dem_std + dem_mean).numpy()
                slp = (x[:, 11] * float(stats["std"][11]) + float(stats["mean"][11])).numpy() / 100.0
                vv = v.numpy()
                w = vv.sum((1, 2)).clip(1)
                elev.append((dem_m * vv).sum((1, 2)) / w)
                slope.append((slp * vv).sum((1, 2)) / w)
                frac.append((m.numpy() * vv).sum((1, 2)) / w)
        return (np.concatenate(F), np.concatenate(elev), np.concatenate(slope), np.concatenate(frac))

    Ftr, Etr, Str, Rtr = feats_targets(ds_tr)
    Fdv, Edv, Sdv, Rdv = feats_targets(ds_dev)

    report = {
        "weights": args.weights,
        "split": "DEV (hiroshima+thrissur)",
        "reconstruction": {
            "mae_m_masked": round(recon_mae, 2),
            "mae_m_baseline_train_mean": round(const_mae, 2),
            "mae_m_baseline_tile_visible_mean": round(visible_mae, 2),
            "elevation_spread_mae_m": round(spread_mae, 2),
            "relative_to_spread": round(rel, 3),
            "pixels_evaluated": int(sum(t.numel() for t in abs_err)),
        },
        "probe": {
            "elevation": ridge_r2(Ftr, Etr, Fdv, Edv),
            "slope": ridge_r2(Ftr, Str, Fdv, Sdv),
            "landslide_fraction": ridge_r2(Ftr, Rtr, Fdv, Rdv),
        },
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
