"""严格的辐射定标验证：把 Sen12 的 DN 与官方 L2A 产品的 BOA 反射率逐像元比对。

做法：
1. 取认证样本（记录里有 STAC item id、日期、CRS、bbox）；
2. 在 **Microsoft Planetary Computer** 的 `sentinel-2-l2a` 集合里检索**同一景**产品
   （独立于我们回补 B01/B09 时用的 AWS sentinel-cogs，属于不同分发方）；
3. 用产品元数据里的 `scale` / `offset` 把整数像元换成 BOA 反射率；
4. 重投影到样本网格，与 Sen12 的 DN（按 /10000 换算）逐像元比较；
5. 报告各波段的中位绝对误差、90 分位误差，以及"是否在同一产品内自洽"。

判据：
- 中位绝对误差 < 0.01（即 100 DN）→ 认为辐射量级一致；
- 若某波段系统性偏移（中位误差符号一致且 > 0.01），说明两岸的基线/偏移处理不同。

注意：这不是"和 CDSE 官方站比对"，而是"和另一家独立分发的同一 ESA 产品比对"。
若要真正对 CDSE 端点比对，需要 CDSE 的 OAuth 凭据（本脚本预留 `--cdse-token`）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from landslide_seg.data.band_mapping import S2_L2A_BANDS  # noqa: E402
from landslide_seg.data.certified import filter_records, load_sample, read_manifest  # noqa: E402

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

#: Sentinel-2 波段名 -> Planetary Computer asset key
PC_ASSET = {
    "B01": "B01", "B02": "B02", "B03": "B03", "B04": "B04",
    "B05": "B05", "B06": "B06", "B07": "B07", "B08": "B08",
    "B8A": "B8A", "B09": "B09", "B11": "B11", "B12": "B12",
}


def pc_token() -> str:
    """Planetary Computer 的 SAS token（公开端点，无需账号）。"""
    import urllib.request

    with urllib.request.urlopen(
        f"{PC_STAC.replace('/stac/v1', '')}/sas/v1/token/sentinel-2-l2a", timeout=60
    ) as r:
        return json.loads(r.read())["token"]


def search_pc(item_id: str, lon: float, lat: float, date: str) -> dict | None:
    """按点+日期检索，并用 MGRS 瓦片号对齐到同一景（PC 与 Earth Search 的 id 命名不同）。"""
    import re
    import urllib.request

    m = re.search(r"S2[AB]_([0-9]{2}[A-Z]{3})_", item_id)
    tile = m.group(1) if m else None
    body = {
        "collections": ["sentinel-2-l2a"],
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "datetime": f"{date}T00:00:00Z/{date}T23:59:59Z",
        "limit": 20,
    }
    req = urllib.request.Request(
        f"{PC_STAC}/search", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        feats = json.loads(r.read()).get("features", [])
    if tile:
        same_tile = [f for f in feats if f"T{tile}" in f["id"]]
        if same_tile:
            return same_tile[0]
    return feats[0] if feats else None


def read_band(asset: dict, bounds, crs: str, out_hw=(128, 128),
              transpose: bool = True) -> tuple[np.ndarray, dict]:
    """读一个波段并重投影到样本网格，返回 (反射率数组, 定标信息)。"""
    import rasterio
    from rasterio.warp import Resampling, reproject
    from rasterio.windows import from_bounds

    xmin, ymin, xmax, ymax = bounds
    h, w = out_hw
    with rasterio.open(asset["href"]) as src:
        win = from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)
        data = src.read(1, window=win, boundless=True, fill_value=0).astype(np.float32)
        src_tf = src.window_transform(win)
        src_crs = src.crs
    dst = np.zeros((h, w), dtype=np.float32)
    from rasterio.transform import from_bounds as _fb

    reproject(
        source=data, destination=dst,
        src_transform=src_tf, src_crs=src_crs,
        dst_transform=_fb(xmin, ymin, xmax, ymax, w, h), dst_crs=crs,
        resampling=Resampling.nearest, src_nodata=0, dst_nodata=0,
    )
    meta = asset.get("raster:bands", [{}])[0]
    scale = float(meta.get("scale", 1e-4))
    offset = float(meta.get("offset", 0.0))
    refl = dst * scale + offset
    return (refl.T if transpose else refl), {"scale": scale, "offset": offset}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--cdse-token", default=None, help="预留：CDSE OAuth token（本机无凭据时为 None）")
    args = ap.parse_args()

    token = pc_token()
    recs = filter_records(read_manifest(ROOT / "manifests" / "certified.jsonl"), split="test")[: args.n]
    report = {"source": "Microsoft Planetary Computer (independent L2A distribution)",
              "cdse_used": bool(args.cdse_token), "samples": []}

    for rec in recs:
        smp = load_sample(rec)
        x12 = smp["x12"]
        cx, cy = rec["center_xy"]
        bounds = (cx - 640, cy - 640, cx + 640, cy + 640)
        from pyproj import Transformer

        tr = Transformer.from_crs(rec["crs"], "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(cx, cy)
        item_id = rec["backfill_source"]["item_id"]
        item = search_pc(item_id, lon, lat, rec["date"])
        if item is None:
            report["samples"].append({"sample_id": rec["sample_id"], "error": "PC 无同日产品"})
            continue

        per_band = {}
        for band in S2_L2A_BANDS:
            key = PC_ASSET[band]
            if key not in item["assets"]:
                per_band[band] = {"error": "asset missing"}
                continue
            asset = item["assets"][key]
            href = asset["href"] + ("&" if "?" in asset["href"] else "?") + token
            ref, cal = read_band({"href": href, "raster:bands": asset.get("raster:bands", [{}])},
                                 bounds, rec["crs"])
            ours = x12[S2_L2A_BANDS.index(band)] / 10000.0        # Sen12 DN -> 反射率
            ok = (ref > 0) & (ours > 0) & np.isfinite(ref)
            d = (ours[ok] - ref[ok])
            per_band[band] = {
                "n": int(ok.sum()),
                "median_abs_diff": float(np.median(np.abs(d))) if ok.sum() else None,
                "p90_abs_diff": float(np.percentile(np.abs(d), 90)) if ok.sum() else None,
                "median_signed_diff": float(np.median(d)) if ok.sum() else None,
                "ours_median": float(np.median(ours[ok])) if ok.sum() else None,
                "ref_median": float(np.median(ref[ok])) if ok.sum() else None,
                "scale": cal["scale"], "offset": cal["offset"],
            }
        medians = [v["median_abs_diff"] for v in per_band.values() if v.get("median_abs_diff") is not None]
        report["samples"].append({
            "sample_id": rec["sample_id"], "pc_item": item["id"], "date": rec["date"],
            "overall_median_abs_diff": float(np.median(medians)) if medians else None,
            "bands": per_band,
        })
        print(f"[{rec['sample_id']}] PC={item['id']} 中位|Δref|={np.median(medians):.5f}"
              if medians else f"[{rec['sample_id']}] 无数据")

    path = ROOT / "outputs" / "logs" / "verify_radiometry.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n波段  中位|Δ反射率|  p90|Δ|  中位有符号Δ   Sen12中位  PC中位")
    for s in report["samples"]:
        if "bands" not in s:
            continue
        print(f"--- {s['sample_id']}")
        for band, v in s["bands"].items():
            if v.get("median_abs_diff") is None:
                continue
            print(f"  {band:4s} {v['median_abs_diff']:.5f}  {v['p90_abs_diff']:.5f}  "
                  f"{v['median_signed_diff']:+.5f}  {v['ours_median']:.4f}  {v['ref_median']:.4f}")
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
