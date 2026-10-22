"""审计输入统计与原始数组的单位是否一致。

做法：对每个“统计文件 + 缓存/侧表”组合，取若干图块，比较统计文件里的均值与
原始数组的均值。两者应当接近（比值约 1）。若比值约 100 或 0.01，说明统计与
数据使用了不同单位（例如坡度按“度”写、数组按 0.01° 存）。

只读检查，不修改任何文件。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from landslide_seg.data.segformer_dataset import S2_10_BANDS, read_manifest  # noqa: E402


def stats_of(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def npz_rows(manifest: Path, n: int = 8, key: str = "cache", role: str | None = None) -> list[dict]:
    rows = read_manifest(manifest)
    if role:
        rows = [r for r in rows if r.get("protocol_role") == role or r.get("split") == role]
    step = max(1, len(rows) // n)
    return rows[::step][:n]


def raw_means_from_npz(rows: list[dict], keys: list[str]) -> dict:
    acc = {k: [] for k in keys}
    for r in rows:
        with np.load(r["cache"]) as d:
            v = np.asarray(d["valid"], dtype=bool)
            for k in keys:
                if k in d:
                    acc[k].append(np.asarray(d[k])[v].astype(np.float64).mean())
    return {k: (float(np.mean(v)) if v else None) for k, v in acc.items()}


def raw_means_from_sidecar(sidecar: Path, rows: list[dict]) -> dict:
    meta = json.loads((sidecar / "meta.json").read_text(encoding="utf-8"))
    pos = {sid: i for i, sid in enumerate(meta["rows"])}
    dem = np.load(sidecar / "dem.npy", mmap_mode="r")
    slope = np.load(sidecar / "slope.npy", mmap_mode="r")
    d_acc, s_acc = [], []
    for r in rows:
        i = pos[r["sample_id"]]
        with np.load(r["cache"]) as z:
            v = np.asarray(z["valid"], dtype=bool)
        d_acc.append(np.asarray(dem[i])[v].astype(np.float64).mean())
        s_acc.append(np.asarray(slope[i])[v].astype(np.float64).mean())
    return {"dem": float(np.mean(d_acc)), "slope": float(np.mean(s_acc))}


def compare(label: str, stats_path: Path, raw: dict) -> dict:
    st = stats_of(stats_path)
    out = {"label": label, "stats": str(stats_path.relative_to(ROOT)).replace("\\", "/"),
           "channels": len(st["mean"]), "rows": []}
    for i, ch in enumerate(st.get("channels") or S2_10_BANDS):
        if i >= len(st["mean"]):
            break
        key = ch.lower()
        if key in raw and raw[key]:
            ratio = st["mean"][i] / raw[key] if raw[key] else float("nan")
        elif ch in S2_10_BANDS and "x10_idx" in raw:
            ratio = float("nan")
        else:
            ratio = None
        out["rows"].append({"channel": ch, "stats_mean": round(float(st["mean"][i]), 3),
                            "raw_mean": None if key not in raw or not raw[key] else round(raw[key], 3),
                            "ratio": None if ratio is None or not np.isfinite(ratio) else round(float(ratio), 3)})
    return out


def main() -> None:
    reports = []

    # 1) v4r1 十波段
    rows = npz_rows(ROOT / "manifests/v4r1/pool_100.jsonl", role="train")
    st = stats_of(ROOT / "manifests/v4r1/stats_core25.json")
    r = {"label": "v4r1 十波段（stats_core25）", "stats": "manifests/v4r1/stats_core25.json",
         "channels": len(st["mean"]), "rows": []}
    per_band = [[] for _ in range(10)]
    for row in rows:
        with np.load(row["cache"]) as d:
            x = np.asarray(d["x10"]).astype(np.float64)
            v = np.asarray(d["valid"], dtype=bool)
            for b in range(10):
                per_band[b].append(x[b][v].mean())
    for b, name in enumerate(S2_10_BANDS):
        m = float(np.mean(per_band[b]))
        r["rows"].append({"channel": name, "stats_mean": round(float(st["mean"][b]), 3),
                          "raw_mean": round(m, 3), "ratio": round(float(st["mean"][b]) / m, 3)})
    reports.append(r)

    # 2) v4r1 地形侧表 vs 旧统计（已知缺陷）与修正统计
    side_raw = raw_means_from_sidecar(ROOT / "data/v4r1_geo", rows)
    for name, label in (("stats_geoboth.json", "v4r1 地形（旧统计，作废）"),
                        ("stats_geoboth_raw.json", "v4r1 地形（修正统计）")):
        p = ROOT / "manifests/v4r1" / name
        if p.is_file():
            reports.append(compare(label, p, side_raw))

    # 3) 官方可用子集：DEM / 坡度 / 组合
    off_rows = npz_rows(ROOT / "manifests/official_single_geo/trainval.jsonl", n=8)
    off_raw = raw_means_from_npz([x for x in off_rows if x.get("split") == "train"], ["dem", "slope"])
    for name, label in (("stats_dem.json", "官方子集 DEM"),
                        ("stats_slope.json", "官方子集 坡度"),
                        ("stats_geoboth.json", "官方子集 DEM＋坡度")):
        p = ROOT / "manifests/official_single_geo" / name
        if p.is_file():
            reports.append(compare(label, p, off_raw))

    # 4) 四折地形统计（DEM only）
    p = ROOT / "manifests/event_cv_v5/F1/stats_dem.json"
    fold_rows = npz_rows(ROOT / "manifests/event_cv_v5/F1/pool_100.jsonl", role="TRAIN")
    fold_raw = raw_means_from_npz(fold_rows, ["dem"])
    if p.is_file():
        reports.append(compare("四折 F1 DEM", p, fold_raw))

    print(json.dumps(reports, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
