"""推理延迟/显存基准（batch 1，128×128，FP32）。

补全成本表：训练侧已记录耗时与峰值显存，这里补推理侧，
因为评测时用了 `--skip-efficiency` 以节省时间。

用法::

    python scripts/bench_p2m1_latency.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402
import torch  # noqa: E402

from landslide_seg.training.segformer_trainer import load_checkpoint  # noqa: E402

ARMS = {
    "B2-AllTrain": "outputs/segformer/p2m1/segformer_b2_latedetail_alltrain_s1234/1234/best.pt",
    "B0-AllTrain-ctl": "outputs/segformer/p2m1/segformer_b0_latedetail_alltrain_s1234_ctl/1234/best.pt",
    "P13-AllEvents": "outputs/segformer/p13_events/segformer_b0_latedetail_allevents_n1251_s1234/1234/best.pt",
    "P13-FewEvents": "outputs/segformer/p13_events/segformer_b0_latedetail_fewevents_n1251_s1234/1234/best.pt",
}
WARMUP, ITERS = 20, 100


def main() -> None:
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 10, 128, 128, device=dev)
    out = {}
    for name, rel in ARMS.items():
        p = ROOT / rel
        if not p.is_file():
            print(f"[skip] {name}")
            continue
        model, _ = load_checkpoint(p, device=str(dev))
        model.eval()
        with torch.no_grad():
            for _ in range(WARMUP):
                model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            lat = []
            for _ in range(ITERS):
                t0 = time.perf_counter()
                model(x)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                lat.append((time.perf_counter() - t0) * 1000)
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
                model(x)
                peak = round(torch.cuda.max_memory_allocated(dev) / 2**20, 1)
            else:
                peak = None
        out[name] = {
            "params": sum(p.numel() for p in model.parameters()),
            "batch": 1, "size": [128, 128], "dtype": "float32",
            "p50_ms": round(float(np.percentile(lat, 50)), 2),
            "p95_ms": round(float(np.percentile(lat, 95)), 2),
            "mean_ms": round(float(np.mean(lat)), 2),
            "inference_peak_alloc_mib": peak,
            "warmup": WARMUP, "iters": ITERS,
        }
        print(f"{name:18s} p50={out[name]['p50_ms']:.2f}ms p95={out[name]['p95_ms']:.2f}ms "
              f"peak={peak}MiB")
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    dest = ROOT / "reports" / "p2m1" / "latency.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {dest}")


if __name__ == "__main__":
    main()
