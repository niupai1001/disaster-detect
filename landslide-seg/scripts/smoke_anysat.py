"""AnySat Base 接口验证（真实官方权重，单期 S2，多物理尺度）。

用法:
    python scripts/smoke_anysat.py --scales 40 80 160
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from landslide_seg.data.band_mapping import normalize_anysat  # noqa: E402
from landslide_seg.data.pipeline import make_synthetic_batch  # noqa: E402
from landslide_seg.encoders.anysat_scale import AnySatScaleAdapter  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=int, nargs="+", default=[40, 80, 160])
    ap.add_argument("--size", type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}")

    adapter = AnySatScaleAdapter(
        repo_dir=ROOT / "vendor" / "AnySat-main",
        weight_path=ROOT / "weights" / "AnySat_base.pth",
        patch_sizes_m=tuple(args.scales),
        freeze=True,
    ).to(device)
    print("[model]", json.dumps(adapter.describe(), ensure_ascii=False, indent=2))

    batch = make_synthetic_batch(batch_size=1, size=args.size, seed=0)
    s2 = normalize_anysat(batch.x10).unsqueeze(1).to(device)   # [B,1,10,H,W]
    dates = batch.s2_dates.to(device)

    results = []
    for p in args.scales:
        out = adapter(s2, dates, patch_size_m=p)
        f = out["feature"]
        info = out["info"]
        print(
            f"[scale {p:>3}m] patches={info.num_patches:>4} "
            f"dense={tuple(f.shape)} stride={out['stride_m']}m "
            f"latency={info.latency_s:.2f}s"
        )
        results.append(
            {
                "patch_size_m": p,
                "num_patches": info.num_patches,
                "out_shape": list(f.shape),
                "stride_m": out["stride_m"],
                "latency_s": info.latency_s,
                "finite": bool(torch.isfinite(f).all()),
            }
        )

    report = {
        "device": device,
        "describe": adapter.describe(),
        "input": {"shape": list(s2.shape), "s2_dates": dates.tolist(),
                  "synthetic": True, "note": "date is a placeholder (day 180)"},
        "scales": results,
    }
    path = ROOT / "outputs" / "logs" / "smoke_anysat.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
