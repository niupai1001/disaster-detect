"""为四折事件协议补 DEM／坡度统计（本折 TRAIN 有效像素拟合）。

十波段统计沿用 ``manifests/event_cv_v5/<F>/stats_train.json``；新增通道按与
``build_official_geo.py`` 完全相同的口径拟合：

- 只取本折 ``protocol_role == TRAIN`` 的图块；
- 只取该图块 ``valid`` 为真的像素（与分割评价范围一致）；
- ``dem`` 单位米、``slope`` 存储为 0.01°（int16），统计量按度输出（除以 100）。

产出：
- ``manifests/event_cv_v5/<F>/stats_dem.json``     11 通道（十波段 + DEM）
- ``manifests/event_cv_v5/<F>/stats_geoboth.json`` 12 通道（十波段 + DEM + SLOPE）
- ``manifests/event_cv_v5/<F>/geo_stats_report.json`` 记录拟合像素数、范围与哈希

用法::

    python scripts/build_event_fold_geo_stats.py            # 四折全部
    python scripts/build_event_fold_geo_stats.py --fold F1  # 单折
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

FOLDS = ("F1", "F2", "F3", "F4")
MAN = ROOT / "manifests" / "event_cv_v5"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_fold(fold: str) -> dict:
    fold_dir = MAN / fold
    base_stats_path = fold_dir / "stats_train.json"
    base = json.loads(base_stats_path.read_text(encoding="utf-8"))
    pool = read_rows(fold_dir / "pool_100.jsonl")
    train = [r for r in pool if r.get("protocol_role") == "TRAIN"]
    if not train:
        raise RuntimeError(f"{fold}: pool 中没有 TRAIN 行")

    dem_parts, slp_parts = [], []
    for i, r in enumerate(train, 1):
        with np.load(r["cache"]) as d:
            v = np.asarray(d["valid"], dtype=bool)
            dem_parts.append(np.asarray(d["dem"])[v].astype(np.int64))
            slp_parts.append(np.asarray(d["slope"])[v].astype(np.int64))
        if i % 500 == 0:
            print(f"  [{fold}] {i}/{len(train)}", flush=True)
    dem_flat = np.concatenate(dem_parts)
    slp_flat = np.concatenate(slp_parts)
    if dem_flat.size != slp_flat.size:
        raise RuntimeError("DEM 与坡度像素数不一致")

    stats_paths = {}
    for name, arr, scale, unit in (
        ("dem", dem_flat, 1.0, "m"),
        ("slope", slp_flat, 1.0, "0.01deg"),
    ):
        st = dict(base)
        st["mean"] = list(base["mean"]) + [float(arr.mean() / scale)]
        st["std"] = list(base["std"]) + [float(max(arr.std() / scale, 1e-6))]
        st["channels"] = list(base["bands"]) + [name.upper()]
        st["extra_channel"] = name
        st["extra_unit"] = unit
        st["fit_split"] = f"{base['fit_split']}（{name} 由本折 TRAIN 有效像素拟合）"
        st["note"] = (
            f"十波段沿用 {fold}/stats_train.json；{name} 由本折 TRAIN 有效像素拟合"
            f"（n={arr.size:,} 像素，{len(train)} 图块）"
        )
        p = fold_dir / f"stats_{name}.json"
        p.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
        stats_paths[name] = p

    # geoboth：十波段 + DEM + SLOPE，顺序与 S2SegDataset(extra="both") 一致
    st = dict(base)
    st["mean"] = list(base["mean"]) + [float(dem_flat.mean()), float(slp_flat.mean())]
    st["std"] = list(base["std"]) + [float(max(dem_flat.std(), 1e-6)),
                                     float(max(slp_flat.std(), 1e-6))]
    st["channels"] = list(base["bands"]) + ["DEM", "SLOPE"]
    st["extra_channel"] = "both"
    st["extra_unit"] = "m,deg"
    st["fit_split"] = f"{base['fit_split']}（DEM 与坡度由本折 TRAIN 有效像素拟合）"
    st["note"] = (f"十波段沿用 {fold}/stats_train.json；DEM 单位米、坡度按度，"
                  f"均由本折 TRAIN 有效像素拟合（n={dem_flat.size:,} 像素）")
    geoboth = fold_dir / "stats_geoboth.json"
    geoboth.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    stats_paths["geoboth"] = geoboth

    report = {
        "fold": fold,
        "n_train_tiles": len(train),
        "fit_events": sorted({r["event_id"] for r in train}),
        "n_pixels": int(dem_flat.size),
        "dem": {"mean": float(dem_flat.mean()), "std": float(dem_flat.std()),
                "min": int(dem_flat.min()), "max": int(dem_flat.max())},
        "slope_deg": {"mean": float(slp_flat.mean() / 100.0), "std": float(slp_flat.std() / 100.0),
                      "p50": float(np.percentile(slp_flat, 50) / 100.0),
                      "p99": float(np.percentile(slp_flat, 99) / 100.0),
                      "max": float(slp_flat.max() / 100.0)},
        "inputs": {str(base_stats_path.relative_to(ROOT)).replace("\\", "/"): sha256(base_stats_path),
                   str((fold_dir / "pool_100.jsonl").relative_to(ROOT)).replace("\\", "/"):
                       sha256(fold_dir / "pool_100.jsonl")},
        "outputs": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in stats_paths.values()},
        "fit_ids_sha256": hashlib.sha256(
            "\n".join(sorted(r["sample_id"] for r in train)).encode()).hexdigest(),
    }
    (fold_dir / "geo_stats_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="all", choices=("all", *FOLDS))
    args = ap.parse_args()
    folds = FOLDS if args.fold == "all" else (args.fold,)
    out = {}
    for f in folds:
        print(f"[build] {f}", flush=True)
        out[f] = build_fold(f)
        r = out[f]
        print(f"  n_train={r['n_train_tiles']} pix={r['n_pixels']:,} "
              f"dem {r['dem']['mean']:.1f}±{r['dem']['std']:.1f} m | "
              f"slope {r['slope_deg']['mean']:.2f}° p99 {r['slope_deg']['p99']:.1f}°", flush=True)
    print(json.dumps({k: {kk: v[kk] for kk in ("n_train_tiles", "n_pixels", "dem", "slope_deg")}
                      for k, v in out.items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
