"""拟合 S1 新增四个指数通道的 TRAIN 统计（只读缓存，不训练）。

S1 的十波段仍沿用 ``stats_core25.json``（559 块 TRAIN 核心、合法像素）。
本脚本在同一批 559 块上，用与模型完全相同的
``harmonized_reflectance`` + ``spectral_indices`` 计算 NDVI/NDMI/NBR/REND，
再拟合这四个通道的均值与标准差，写入独立 JSON；DEV 不参与。

用法::

    python -X utf8 scripts/build_s1_extra_stats.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from landslide_seg.data.segformer_dataset import S2_10_BANDS, normalize_s2  # noqa: E402
from landslide_seg.models.spectral_contract import harmonized_reflectance, spectral_indices  # noqa: E402

EPS = 1e-4
INDEX_CHANNELS = ["NDVI", "NDMI", "NBR", "REND"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/v4r1/pool_100.jsonl")
    ap.add_argument("--protocol", default="manifests/v4r1/protocol.json")
    ap.add_argument("--stats", default="manifests/v4r1/stats_core25.json")
    ap.add_argument("--out", default="manifests/v4r1/stats_s1_indices4.json")
    args = ap.parse_args()

    manifest = ROOT / args.manifest
    protocol_path = ROOT / args.protocol
    stats_path = ROOT / args.stats
    out_path = ROOT / args.out
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]

    fit_ids = list(stats["fit_sample_ids"])
    by_id = {r["sample_id"]: r for r in rows}
    missing = [sid for sid in fit_ids if sid not in by_id]
    if missing:
        raise SystemExit(f"统计拟合样本不在清单中: {missing[:3]}")
    train_ids = {r["sample_id"] for r in rows if r["split"] == "train"}
    if not set(fit_ids) <= train_ids:
        raise SystemExit("统计拟合样本必须全部属于 TRAIN")
    if set(fit_ids) != set(protocol["pools"]["25"]):
        raise SystemExit("拟合成员与协议登记的 559 块核心不一致")

    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32)
    total = torch.zeros(4, dtype=torch.float64)
    total_sq = torch.zeros(4, dtype=torch.float64)
    n_pix = 0
    with torch.no_grad():
        for k, sid in enumerate(fit_ids, 1):
            rec = by_id[sid]
            d = np.load(rec["cache"])
            raw = d["x10"].astype(np.float32)
            valid = d["valid"].astype(bool)
            x = torch.from_numpy(normalize_s2(raw, stats))[None]
            rho = harmonized_reflectance(x, mean, std)
            idx = spectral_indices(rho, EPS)[0]                      # [4,128,128]
            m = torch.from_numpy(valid)
            sel = idx[:, m].to(torch.float64)
            total += sel.sum(dim=1)
            total_sq += (sel ** 2).sum(dim=1)
            n_pix += int(m.sum())
            if k % 100 == 0:
                print(f"  {k}/{len(fit_ids)}", flush=True)

    if n_pix == 0:
        raise SystemExit("没有有效像素可用于统计")
    fit_mean = (total / n_pix).tolist()
    var = (total_sq / n_pix - (total / n_pix) ** 2).clamp_min(0.0)
    fit_std = var.sqrt().tolist()
    if not all(np.isfinite(fit_mean + fit_std)):
        raise SystemExit("新增通道统计出现非有限值")

    fit_events = {}
    for sid in fit_ids:
        ev = protocol["members"][sid]["event_id"]
        fit_events[ev] = fit_events.get(ev, 0) + 1

    payload = {
        "data_version": stats["data_version"],
        "bands": list(S2_10_BANDS),
        "channels": INDEX_CHANNELS,
        "unit": "reflectance = official harmonized DN / 10000, negative clipped to 0",
        "eps": EPS,
        "radiometry": "harmonized_dn_v1",
        "fit_split": stats["fit_split"],
        "fit_source": args.stats,
        "n_samples": len(fit_ids),
        "n_pixels_per_channel": [n_pix] * 4,
        "mean": fit_mean,
        "std": fit_std,
        "quantiles": {},
        "note": ("S1 新增四个指数通道的 TRAIN-only 统计；十波段归一化仍由 stats_core25.json 提供，"
                 "本文件不替代也不修改它。DEV/FINAL/DIAG 未参与。"),
        "fit_sample_ids": fit_ids,
        "fit_events": fit_events,
        "source_manifest_sha256": sha256(manifest),
        "source_protocol_sha256": sha256(protocol_path),
        "source_stats_sha256": sha256(stats_path),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_path), "n_samples": len(fit_ids), "n_pixels": n_pix,
                      "mean": fit_mean, "std": fit_std}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
