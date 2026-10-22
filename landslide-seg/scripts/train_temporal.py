"""主模型四组正交试验的训练入口（单期/多期 × 光谱开/关）。

用法::

    python scripts/train_temporal.py --config configs/segformer/v6_main/temporal_single_baseline.yaml --device cuda:0
    python scripts/train_temporal.py --config ... --resume

数据固定为官方可用子集 3012/758/945 的十波段单期缓存 + 同源 15 期缓存；
不使用 DEM。目标标签、valid、归一化统计与历史单期主模型一致。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch  # noqa: E402

from landslide_seg.config import (  # noqa: E402
    PROJECT_ROOT, file_sha256, load_segformer_config, manifest_identity, resolve_path, set_seed,
)
from landslide_seg.data.segformer_dataset import filter_split, load_stats, read_manifest  # noqa: E402
from landslide_seg.data.temporal_v6 import TemporalV6  # noqa: E402
from landslide_seg.models.spectral_temporal import build_temporal_model  # noqa: E402
from landslide_seg.training.temporal_trainer import run_temporal_training  # noqa: E402


def _run_dir(cfg: dict) -> Path:
    s = cfg["segformer"]
    return resolve_path(s["output_dir"]) / s["data"]["version"] / s["experiment"] / str(s["seed"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--stop-after", type=int, default=None)
    ap.add_argument("--microbatch", type=int, default=None)
    ap.add_argument("--experiment", default=None)
    args = ap.parse_args()

    cfg = load_segformer_config(args.config)
    s = cfg["segformer"]
    if args.max_steps is not None:
        s["train"]["max_steps"] = int(args.max_steps)
    if args.eval_every is not None:
        s["train"]["eval_every_updates"] = int(args.eval_every)
    if args.microbatch is not None:
        s["train"]["microbatch"] = int(args.microbatch)
        s["train"]["accumulation"] = int(s["train"]["batch_size"]) // int(args.microbatch)
    if args.experiment is not None:
        s["experiment"] = args.experiment

    dev = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    seed = int(s["seed"])
    seed_info = set_seed(seed, bool(s.get("deterministic", True)))

    manifest = resolve_path(s["data"]["manifest"])
    test_manifest = resolve_path(s["data"]["test_manifest"])
    stats_path = resolve_path(s["data"]["stats"])
    protocol_path = resolve_path(s["data"]["protocol"])
    temporal_root = resolve_path(s["data"]["temporal_root"])
    quality_dir = resolve_path(s["data"]["quality"])
    pretrained = resolve_path(s["model"]["pretrained_dir"])

    rows = read_manifest(manifest)
    train = filter_split(rows, "train")
    val = filter_split(rows, "val")
    test = read_manifest(test_manifest)
    if len(train) != 3012 or len(val) != 758 or len(test) != 945:
        raise RuntimeError(f"官方子集规模不符：train={len(train)} val={len(val)} test={len(test)}")
    if {r["sample_id"] for r in train} & {r["sample_id"] for r in test}:
        raise RuntimeError("TRAIN 与 TEST 样本重叠")
    if {r["sample_id"] for r in val} & {r["sample_id"] for r in test}:
        raise RuntimeError("VAL 与 TEST 样本重叠")

    stats = load_stats(stats_path)
    mode = str(s["temporal"]["mode"])
    if mode not in ("single", "multi"):
        raise ValueError("主模型只比较 single/multi")
    run_dir = _run_dir(cfg)
    if not args.resume and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run 目录非空；换实验名或 --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_temporal_model(s, stats).to(dev)
    train_ds = TemporalV6(train, stats, mode=mode, augment=True, seed=seed,
                          root=s["data"]["temporal_root"], quality=s["data"]["quality"],
                          preload=bool(s["data"].get("preload", False)))
    val_ds = TemporalV6(val, stats, mode=mode, augment=False, seed=seed,
                        root=s["data"]["temporal_root"], quality=s["data"]["quality"])

    quality_meta = []
    qpath = quality_dir / "complete.json"
    if qpath.is_file():
        quality_meta = [json.loads(qpath.read_text(encoding="utf-8")),
                        {"sha256": file_sha256(qpath)}]

    meta = {
        "experiment": s["experiment"], "seed": seed, "seed_info": seed_info,
        "data_version": s["data"]["version"], "temporal_mode": mode,
        "manifest": manifest_identity(manifest), "test_manifest": manifest_identity(test_manifest),
        "stats_sha256": file_sha256(stats_path),
        "protocol_sha256": file_sha256(protocol_path),
        "temporal_index_sha256": file_sha256(temporal_root / "index.json"),
        "quality_sidecar": quality_meta,
        "date_policy": {"reference": train_ds.date_reference, "unit": train_ds.date_unit,
                        "single_frame": "time_index 的真实日期（非 0、非帧序号）",
                        "multi_frame": "15 帧真实日期；时间注意力用相对 anchor 的间隔",
                        "missing": "SCL=0/非有限/显式 FillValue 填 TRAIN 均值；云与 SCL255 仅记录"},
        "source_sha256": {
            str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): file_sha256(p)
            for p in sorted((PROJECT_ROOT / "src/landslide_seg").rglob("*.py"))
        },
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("torch", "transformers", "numpy", "scipy", "PyYAML")},
        "weight": {"dir": str(pretrained).replace("\\", "/"),
                   "sha256": file_sha256(pretrained / "model.safetensors")},
        "params": model.count_params(),
        "spatial_init_digest": model.spatial_digest(),
        "adapter_report": model.adapter_report,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": str(dev),
        "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "batch_size": int(s["train"]["batch_size"]),
        "microbatch": int(s["train"]["microbatch"]),
        "accumulation": int(s["train"]["batch_size"]) // int(s["train"]["microbatch"]),
        "max_steps": int(s["train"]["max_steps"]),
        "target_tile_exposures": int(s["train"]["max_steps"]) * int(s["train"]["batch_size"]),
        "train_samples": [r["sample_id"] for r in train],
        "val_samples": [r["sample_id"] for r in val],
        "test_samples": [r["sample_id"] for r in test],
        "batchnorm_note": "decoder BatchNorm 统计基于 microbatch，四组一致；不等价历史 batch64",
    }
    if not args.resume:
        (run_dir / "resolved_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1),
                                                      encoding="utf-8")
        (run_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    model.train()
    summary = run_temporal_training(
        cfg, model, train_ds, val_ds, list(range(len(train))), list(range(len(val))),
        meta, run_dir, dev, resume=args.resume, stop_after=args.stop_after,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=1,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
