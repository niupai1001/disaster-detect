"""数据契约审计：真实 L4S 数据的波段、量级、掩膜统计，以及**严格模式的拒绝行为**。

这个脚本的核心产出不是"跑通"，而是证明：
1. L4S 真实数据只有 B01–B12（无 B8A）+ SLOPE + DEM；
2. 严格模式下把它喂给 TerraMind(12) / AnySat(10) 会抛出 MissingBandError；
3. 因此"完整双骨干 + 真实滑坡 Mask"的正式训练在本数据上是 **BLOCKED**。

用法:
    python scripts/audit_data_contract.py --data-root N:/Users/yzh/Disater
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from landslide_seg.data.band_mapping import (  # noqa: E402
    ANY_SAT_S2_BANDS,
    L4S_BANDS,
    S2_L2A_BANDS,
    MissingBandError,
    estimate_reflectance_scale,
    missing_bands,
    to_anysat,
    to_terramind,
)
from landslide_seg.data.loader import L4SDataset, write_manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="N:/Users/yzh/Disater")
    ap.add_argument("--probe", type=int, default=32)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "logs" / "data_contract.json"))
    args = ap.parse_args()

    root = Path(args.data_root)
    report: dict = {"data_root": str(root), "splits": {}}
    all_samples = []

    for split in ("train", "validation", "test"):
        ds = L4SDataset(root, split=split, limit=args.probe)
        if len(ds) == 0:
            continue
        cover, fg_px = [], []
        for i in range(len(ds)):
            s = ds[i]
            all_samples.append(s)
            cover.append(s.meta.mask_coverage)
            fg_px.append(int(s.mask.sum()))
        report["splits"][split] = {
            "probed": len(ds),
            "mask_coverage_mean": float(np.mean(cover)),
            "empty_tiles": int(sum(1 for c in cover if c == 0)),
            "fg_pixels_median": float(np.median(fg_px)),
            "fg_pixels_min": int(np.min(fg_px)),
            "fg_pixels_max": int(np.max(fg_px)),
        }
        if split == "train":
            report["radiometry"] = ds.stats

    report["band_contract"] = {
        "l4s_bands": list(L4S_BANDS),
        "l4s_channel_count": len(L4S_BANDS),
        "terramind_requires": list(S2_L2A_BANDS),
        "terramind_missing": missing_bands(L4S_BANDS, S2_L2A_BANDS),
        "anysat_requires": list(ANY_SAT_S2_BANDS),
        "anysat_missing": missing_bands(L4S_BANDS, ANY_SAT_S2_BANDS),
    }

    # 严格模式必须拒绝
    probe = all_samples[0]
    refusals = {}
    for name, fn in (("terramind", to_terramind), ("anysat", to_anysat)):
        try:
            fn(probe.array, L4S_BANDS, policy="strict")
            refusals[name] = "UNEXPECTED_PASS"
        except MissingBandError as exc:
            refusals[name] = f"refused: {exc.missing}"
    report["strict_mode"] = refusals

    # smoke 模式下各通道的实际来源（证明标记生效）
    tm = to_terramind(probe.array, L4S_BANDS, policy="synthetic_smoke")
    report["smoke_mapping"] = {
        "terramind_source_bands": list(tm.source_bands),
        "synthetic_bands": tm.synthetic_bands,
        "fully_observed": tm.is_fully_observed,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_manifest(all_samples, out.parent / "l4s_manifest_sample.csv")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
