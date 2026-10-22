"""验证 12 波段的装配、辅助波段回补与多分辨率语义是否正确。

检查项：
  V1 波段顺序：认证缓存里的 10 个原始波段必须与 Sen12 源文件**逐像素完全相等**
     （任何索引错位——例如把 B8A 与 B09 弄反——都会在这里暴露）
  V2 回补一致性：缓存里的 B01/B09 必须与重新从同一 STAC 产品读出的窗口完全相等
  V3 原生分辨率：B01/B09 是 60 m 上采样到 10 m，其高频能量应显著低于 10 m 波段
  V4 上采样保真：把 10 m 的 B01 聚合回 60 m，应与直接读 60 m 的像素高度一致
  V5 多分辨率金字塔：P0→P3 的高频能量应单调下降（分辨率逐级变粗）
  V6 骨干有效分辨率：TerraMind(14×14 上采样) 的高频能量应远低于 AnySat(10 m)
  V7 AnySat 两尺度：160 m 分支应比 40 m 分支更平滑

用法: python scripts/verify_bands.py --n 3
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

from landslide_seg.config import load_config, set_seed  # noqa: E402
from landslide_seg.data.band_mapping import S2_L2A_BANDS, normalize_anysat  # noqa: E402
from landslide_seg.data.certified import filter_records, load_sample, read_manifest  # noqa: E402
from landslide_seg.data.s2_backfill import find_same_day_item, read_asset_window  # noqa: E402
from landslide_seg.data.sen12 import SEN12_S2_BANDS, load_sen12  # noqa: E402
from landslide_seg.models.build import build_from_config  # noqa: E402


def hf_energy(x: np.ndarray) -> float:
    """高频能量：梯度幅值均值 / 通道标准差，衡量"细节量"。"""
    gx = np.abs(np.diff(x, axis=-1)).mean()
    gy = np.abs(np.diff(x, axis=-2)).mean()
    return float((gx + gy) / 2 / (x.std() + 1e-6))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    set_seed(1234)
    cfg = load_config(ROOT / "configs" / "default.yaml")
    recs = filter_records(read_manifest(ROOT / "manifests" / "certified.jsonl"), split="test")[: args.n]
    report: dict = {"samples": [], "checks": {}}

    worst = {"V1_max_abs": 0.0, "V2_max_abs": 0.0}
    hf = {"b01": [], "b09": [], "b04": [], "b08": []}
    v4_err = []
    for rec in recs:
        s = load_sen12(rec["source_uri"])
        idx = rec["time_index"]
        smp = load_sample(rec)
        x12 = smp["x12"]

        # V1 波段顺序：10 个原始波段逐像素比对
        per_band = {}
        for name in SEN12_S2_BANDS:
            i = S2_L2A_BANDS.index(name)
            d = float(np.abs(x12[i] - s.bands[name][idx].astype(np.float32)).max())
            per_band[name] = d
            worst["V1_max_abs"] = max(worst["V1_max_abs"], d)

        # V2 回补一致性
        import datetime as _dt

        from pyproj import Transformer

        date = s.attrs["_time_ref"] + _dt.timedelta(days=int(s.time[idx]))
        tr = Transformer.from_crs(s.attrs["crs"], "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(s.attrs["center_x"], s.attrs["center_y"])
        item = find_same_day_item(lon, lat, date)
        cx, cy = s.attrs["center_x"], s.attrs["center_y"]
        bounds = (cx - 640, cy - 640, cx + 640, cy + 640)
        d01 = float(np.abs(x12[0] - read_asset_window(item["assets"]["coastal"]["href"], bounds, s.attrs["crs"], (128, 128))).max())
        d09 = float(np.abs(x12[9] - read_asset_window(item["assets"]["nir09"]["href"], bounds, s.attrs["crs"], (128, 128))).max())
        worst["V2_max_abs"] = max(worst["V2_max_abs"], d01, d09)

        # V3 原生分辨率 -> 高频能量
        for key, band in (("b01", 0), ("b09", 9), ("b04", 3), ("b08", 7)):
            hf[key].append(hf_energy(x12[band]))
        # V4 上采样保真：60m 直读 vs 10m 聚合到 60m（用 2x2 均值近似 10m->20m 再比对不严谨，
        # 这里改为比较 10m 的 6x6 块均值 与 60m 直读的对应像元，取中位绝对差）
        # 1280 m / 60 m ≈ 21.3 px，取 21×21 近似原生 60 m 网格
        direct60 = read_asset_window(
            item["assets"]["coastal"]["href"], bounds, s.attrs["crs"], (21, 21), transpose=True
        )
        agg = torch.nn.functional.interpolate(
            torch.from_numpy(x12[0])[None, None], size=(21, 21), mode="area"
        )[0, 0].numpy()
        v4_err.append(float(np.median(np.abs(agg - direct60))))

        report["samples"].append({
            "sample_id": rec["sample_id"], "event": rec["event_id"], "time_index": idx,
            "V1_per_band_max_abs": per_band,
            "V2_b01_max_abs": d01, "V2_b09_max_abs": d09,
            "hf_energy": {k: round(v[-1], 4) for k, v in hf.items()},
        })

    # V5-V7 多分辨率
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_from_config(cfg, device=device)
    model.eval()
    smp = load_sample(recs[0])
    x12 = torch.from_numpy(smp["x12"])[None].to(device)
    x10 = torch.from_numpy(smp["x10"])[None].to(device)
    dates = torch.tensor([[float(smp["doy"])]], device=device)
    with torch.no_grad():
        out = model(x12, x10, dates, return_features=True)
        tm = model.terramind(torch.from_numpy(smp["x12"])[None].to(device))
        f40 = model.anysat(normalize_anysat(x10).unsqueeze(1), dates, patch_size_m=40)
        f160 = model.anysat(normalize_anysat(x10).unsqueeze(1), dates, patch_size_m=160)

    tm_up = torch.nn.functional.interpolate(
        tm["tokens"][-1].transpose(1, 2).reshape(1, -1, 14, 14), size=(128, 128), mode="bilinear"
    )
    py = out.features["pyramid"]
    report["checks"] = {
        "V1_band_order_max_abs": worst["V1_max_abs"],
        "V1_pass": worst["V1_max_abs"] == 0.0,
        "V2_backfill_max_abs": worst["V2_max_abs"],
        "V2_pass": worst["V2_max_abs"] == 0.0,
        "V3_hf_energy_mean": {k: round(float(np.mean(v)), 4) for k, v in hf.items()},
        "V3_pass": float(np.mean(hf["b01"])) < float(np.mean(hf["b08"]))
                  and float(np.mean(hf["b09"])) < float(np.mean(hf["b08"])),
        "V4_resample_median_abs_err_DN": round(float(np.median(v4_err)), 2),
        # 直接把不同尺寸的特征图做差分没有可比性（步长不同）。
        # 正确做法：全部上采样回 128×128，再比较"还剩多少高频"。
        "V5_pyramid_hf_energy_upsampled": [
            round(
                hf_energy(
                    torch.nn.functional.interpolate(
                        p[0].mean(0)[None, None], size=(128, 128), mode="bilinear"
                    )[0, 0].cpu().numpy()
                ),
                4,
            )
            for p in py
        ],
        "V6_hf_terramind_vs_anysat": [
            round(hf_energy(tm_up[0].mean(0).cpu().numpy()), 4),
            round(hf_energy(f40["feature"][0].mean(0).cpu().numpy()), 4),
        ],
        "V7_hf_anysat40_vs_160": [
            round(hf_energy(f40["feature"][0].mean(0).cpu().numpy()), 4),
            round(hf_energy(f160["feature"][0].mean(0).cpu().numpy()), 4),
        ],
    }
    v5 = report["checks"]["V5_pyramid_hf_energy_upsampled"]
    report["checks"]["V5_pass"] = all(v5[i] >= v5[i + 1] for i in range(len(v5) - 1))
    report["checks"]["V6_pass"] = (
        report["checks"]["V6_hf_terramind_vs_anysat"][0] < report["checks"]["V6_hf_terramind_vs_anysat"][1]
    )
    report["checks"]["V7_pass"] = (
        report["checks"]["V7_hf_anysat40_vs_160"][1] < report["checks"]["V7_hf_anysat40_vs_160"][0]
    )

    path = ROOT / "outputs" / "logs" / "verify_bands.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for k, v in report["checks"].items():
        print(f"{k}: {v}")
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
