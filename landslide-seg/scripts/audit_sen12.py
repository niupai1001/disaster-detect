"""Sen12Landslides 可用性审计 + 真实数据前向。

做三件事：
1. 逐文件核对波段、DN 量级、SCL 有效像素、掩膜；
2. 用**真实** Sentinel-2 10 波段跑 AnySat dense 前向（严格模式，不补任何波段）；
3. 尝试 TerraMind 12 波段路径，**预期**因缺 B01/B09 抛 MissingBandError —— 如实记录。

用法:
    python scripts/audit_sen12.py --root data/sen12 --limit 3
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

from landslide_seg.data.band_mapping import MissingBandError, normalize_anysat  # noqa: E402
from landslide_seg.data.sen12 import describe, list_sen12_files, load_sen12  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data" / "sen12"))
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--skip-forward", action="store_true")
    args = ap.parse_args()

    files = list_sen12_files(args.root, limit=args.limit)
    if not files:
        print(f"[ERR] {args.root} 下没有 *_s2_*.nc")
        return
    out_dir = ROOT / "outputs" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"root": str(args.root), "files": [], "any_sat_forward": [],
                    "terramind_attempt": None}
    samples = []
    for p in files:
        s = load_sen12(p)
        samples.append(s)
        d = describe(s)
        report["files"].append(d)
        print(
            f"[file] {d['file']} steps={d['steps']} crs={d['crs']} "
            f"event={d['event_date']} pre/post={d['pre_post']} doy={d['doy_post']} "
            f"mask_px={d['mask_positive_px']} valid={d['valid_ratio']:.2%}"
        )
        print(f"       band_median(DN)={d['band_median']}")

    # ---- AnySat：真实 10 波段 ----
    if not args.skip_forward:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        from landslide_seg.encoders.anysat_scale import AnySatScaleAdapter

        adapter = AnySatScaleAdapter(
            repo_dir=ROOT / "vendor" / "AnySat-main",
            weight_path=ROOT / "weights" / "AnySat_base.pth",
            patch_sizes_m=(40, 160),
            freeze=True,
        ).to(device)

        for s in samples:
            inp = s.as_model_inputs()
            x10 = torch.from_numpy(inp["x10"]).unsqueeze(0)          # [1,10,H,W]
            s2 = normalize_anysat(x10).unsqueeze(1).to(device)        # [1,1,10,H,W]
            dates = torch.tensor([[float(inp["doy"])]], device=device)
            rec = {"file": s.path.name, "doy": inp["doy"],
                   "missing_for_terramind": inp["meta"]["missing_for_terramind"]}
            for pm in (40, 160):
                out = adapter(s2, dates, patch_size_m=pm)
                f = out["feature"]
                rec[f"patch_{pm}m"] = {
                    "out_shape": list(f.shape),
                    "num_patches": out["info"].num_patches,
                    "latency_s": round(out["info"]["latency_s"] if isinstance(out["info"], dict) else out["info"].latency_s, 3),
                    "finite": bool(torch.isfinite(f).all()),
                    "mean": round(float(f.mean()), 4),
                    "std": round(float(f.std()), 4),
                }
                print(f"[anysat] {s.path.name} @{pm}m -> {tuple(f.shape)} "
                      f"finite={rec[f'patch_{pm}m']['finite']}")
            report["any_sat_forward"].append(rec)

    # ---- TerraMind：预期缺失 ----
    try:
        samples[0].assert_terramind_ready()
        report["terramind_attempt"] = "UNEXPECTED_PASS"
    except MissingBandError as exc:
        report["terramind_attempt"] = {"refused": True, "missing": exc.missing}
        print(f"[terramind] 按预期拒绝：缺 {exc.missing}（这是正确行为，不是成功）")

    path = out_dir / "audit_sen12.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
