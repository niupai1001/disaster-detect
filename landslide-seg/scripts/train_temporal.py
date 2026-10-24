"""项目固定事件留出协议下的时间控制训练入口（单期/多期 × 汇聚方式）。

用法::

    python scripts/check_research_protocol.py --config configs/segformer/v4r1_temporal_control/single.yaml
    python scripts/train_temporal.py --config configs/segformer/v4r1_temporal_control/single.yaml --device cuda:0

数据为 event_ood_v4r1 清单（TRAIN 2223 / DEV 669）的十波段时间缓存；
不读 TEST，不使用 DEM；目标标签、valid、归一化统计与单期基线一致。
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
import numpy as np  # noqa: E402

from landslide_seg.config import (  # noqa: E402
    PROJECT_ROOT, file_sha256, load_segformer_config, manifest_identity, resolve_path, set_seed,
)
from landslide_seg.data.segformer_dataset import filter_split, load_stats, read_manifest  # noqa: E402
from landslide_seg.data.temporal_v6 import TemporalV6  # noqa: E402
from landslide_seg.models.spectral_temporal import build_temporal_model  # noqa: E402
from landslide_seg.training.budget import resolve_update_budget  # noqa: E402
from landslide_seg.training.temporal_trainer import run_temporal_training  # noqa: E402


def _run_dir(cfg: dict) -> Path:
    s = cfg["segformer"]
    return resolve_path(s["output_dir"]) / s["data"]["version"] / s["experiment"] / str(s["seed"])


def _module_digest(module) -> str | None:
    if module is None:
        return None
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


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
    ap.add_argument("--set", action="append", default=[], help="key.sub=value 覆盖，与预检一致")
    args = ap.parse_args()

    cfg = load_segformer_config(args.config, args.set)
    s = cfg["segformer"]
    if args.max_steps is not None:
        s["train"]["max_steps"] = int(args.max_steps)
        if "stop_updates" in s["train"]:
            s["train"]["stop_updates"] = int(args.max_steps)
    if args.eval_every is not None:
        s["train"]["eval_every_updates"] = int(args.eval_every)
    if args.microbatch is not None:
        s["train"]["microbatch"] = int(args.microbatch)
        s["train"]["accumulation"] = int(s["train"]["batch_size"]) // int(args.microbatch)
    if args.experiment is not None:
        s["experiment"] = args.experiment
    if "PLACEHOLDER" in str(s["experiment"]):
        raise RuntimeError("history 是条件组：先按步骤2–4结果确定 aggregation，并改掉实验名中的 PLACEHOLDER")
    budget = resolve_update_budget(s["train"])

    dev = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    seed = int(s["seed"])
    seed_info = set_seed(seed, bool(s.get("deterministic", True)))

    manifest = resolve_path(s["data"]["manifest"])
    stats_path = resolve_path(s["data"]["stats"])
    protocol_path = resolve_path(s["data"]["protocol"])
    temporal_root = resolve_path(s["data"]["temporal_root"])
    quality = s["data"].get("quality")
    quality_dir = resolve_path(quality) if quality else None
    pretrained = resolve_path(s["model"]["pretrained_dir"])

    rows = read_manifest(manifest)
    train = filter_split(rows, "train")
    val = filter_split(rows, "val")
    if len(train) + len(val) != len(rows):
        raise RuntimeError("清单角色必须只有 train/val")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    subset = s["data"].get("subset_protocol")
    if subset:
        # 派生共同子集：成员以子集登记为准，仍必须是原 TRAIN 的子集且 DEV 不变
        derived = json.loads(resolve_path(subset).read_text(encoding="utf-8"))
        if resolve_path(derived["manifest"]) != manifest:
            raise RuntimeError("配置清单与派生登记不一致")
        expected_train = set(derived["train_ids"])
        expected_dev = set(derived["dev_ids"])
    else:
        expected_train = set(protocol["pools"]["100"])
        expected_dev = set(protocol["dev_ids"])
    if {r["sample_id"] for r in train} != expected_train:
        raise RuntimeError("TRAIN 成员与冻结协议不一致" if not subset else "TRAIN 成员与派生登记不一致")
    if {r["sample_id"] for r in val} != expected_dev:
        raise RuntimeError("DEV 成员与冻结协议不一致")
    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("清单存在重复 sample_id")
    test_manifest = s["data"].get("test_manifest")
    test = read_manifest(resolve_path(test_manifest)) if test_manifest else []
    if test and ({r["sample_id"] for r in train} | {r["sample_id"] for r in val}) & {
            r["sample_id"] for r in test}:
        raise RuntimeError("TRAIN/DEV 与 TEST 样本重叠")

    stats = load_stats(stats_path)
    mode = str(s["temporal"]["mode"])
    aggregation = str(s["temporal"].get("aggregation", "date"))
    if mode not in ("single", "multi", "history"):
        raise ValueError("temporal.mode 必须是 single/multi/history")
    if aggregation not in ("identity", "mean", "content", "date"):
        raise ValueError("temporal.aggregation 必须是 identity/mean/content/date")
    if (mode == "single") != (aggregation == "identity"):
        raise ValueError("single 必须配 aggregation=identity；multi/history 必须配 mean/content/date")
    run_dir = _run_dir(cfg)
    if not args.resume and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run 目录非空；换实验名或 --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_temporal_model(s, stats).to(dev)
    root_arg = str(temporal_root)
    quality_arg = str(quality_dir) if quality_dir is not None else "data/s2_temporal_v4r1_none"
    window = s["temporal"].get("window_days")
    plan = s["temporal"].get("plan")
    plan_meta = None
    if plan:
        plan_path = resolve_path(plan)
        plan_rows = json.loads(plan_path.read_text(encoding="utf-8"))
        plan_meta = {"path": str(plan_path).replace("\\", "/"), "sha256": file_sha256(plan_path),
                     "n_samples": len(plan_rows),
                     "n_frames": sorted({len(v) for v in plan_rows.values()})}
    train_ds = TemporalV6(train, stats, mode=mode, augment=True, seed=seed,
                          n_frames=s["temporal"].get("n_frames"),
                          window_days=window, plan=plan,
                          root=root_arg, quality=quality_arg,
                          preload=bool(s["data"].get("preload", False)))
    val_ds = TemporalV6(val, stats, mode=mode, augment=False, seed=seed,
                        n_frames=s["temporal"].get("n_frames"),
                        window_days=window, plan=plan,
                        root=root_arg, quality=quality_arg)

    plan = train_ds.frame_plan() + val_ds.frame_plan()
    (run_dir / "frame_selection.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in plan), encoding="utf-8")
    frame_stats = {}
    for split in ("train", "val"):
        n = [p["n_frames"] for p in plan if p["split"] == split]
        if n:
            frame_stats[split] = {"n_samples": len(n), "min": int(min(n)),
                                  "median": float(np.median(n)), "max": int(max(n))}

    quality_meta = None
    qpath = quality_dir / "complete.json" if quality_dir is not None else None
    if qpath is not None and qpath.is_file():
        qmeta = json.loads(qpath.read_text(encoding="utf-8"))
        if qmeta.get("index_sha256") != file_sha256(temporal_root / "index.json"):
            raise ValueError("质量侧表与当前时间缓存索引不一致")
        quality_meta = {"path": str(qpath).replace("\\", "/"),
                        "sha256": file_sha256(qpath),
                        "index_sha256": qmeta["index_sha256"]}

    meta = {
        "experiment": s["experiment"], "seed": seed, "seed_info": seed_info,
        "data_version": s["data"]["version"], "temporal_mode": mode,
        "manifest": manifest_identity(manifest),
        "test_manifest": manifest_identity(test_manifest) if test_manifest else None,
        "stats_sha256": file_sha256(stats_path),
        "protocol_sha256": file_sha256(protocol_path),
        "subset_protocol": ({"path": str(resolve_path(subset)).replace("\\", "/"),
                             "sha256": file_sha256(resolve_path(subset))}
                            if subset else None),
        "temporal": {
            "mode": mode, "aggregation": aggregation,
            "n_frames": s["temporal"].get("n_frames"),
            "window_days": window,
            "plan": plan_meta,
            "frame_selection_rule": ("参考帧必保留；只用参考日期前 N 天内的合法帧，"
                                     "其余按日期排序等秩取样v1，不足时报NOT_ELIGIBLE"
                                     if window is not None else
                                     "参考帧必保留；其余合法帧按日期排序等秩取样v1，无重复，不足时报NOT_ELIGIBLE"),
            "root": str(temporal_root).replace("\\", "/"),
            "index_sha256": file_sha256(temporal_root / "index.json"),
            "quality_path": str(quality_dir).replace("\\", "/") if quality_dir is not None else None,
            "quality_sidecar": quality_meta,
            "frames": frame_stats,
            "history_rule": "观测日期≤参考日期并按日期排序；不按数组下标，不假定源数组有序",
        },
        "date_policy": {"reference": train_ds.date_reference, "unit": train_ds.date_unit,
                        "single_frame": "time_index 的真实日期（非 0、非帧序号）",
                        "multi_frame": "15 帧真实日期；时间汇聚用相对 anchor 的间隔",
                        "missing": ("未启用缺测侧表（quality 目录不存在或没有 complete.json）；"
                                    if quality_meta is None else
                                    "启用匹配的质量侧表，缺测像素填 TRAIN 均值；")
                        + "云与 SCL255 仅记录"},
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
        "temporal_init_digest": _module_digest(model.pools),
        "adapter_report": model.adapter_report,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": str(dev),
        "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "batch_size": int(s["train"]["batch_size"]),
        "microbatch": int(s["train"]["microbatch"]),
        "accumulation": int(s["train"]["batch_size"]) // int(s["train"]["microbatch"]),
        "max_steps": int(budget.stop_updates),
        "stop_updates": int(budget.stop_updates),
        "schedule_updates": int(budget.schedule_updates),
        "warmup_updates": budget.warmup_updates(float(s["optim"]["warmup_frac"])),
        "report_steps": list(budget.report_steps),
        "target_tile_exposures": int(budget.stop_updates) * int(s["train"]["batch_size"]),
        "train_samples": [r["sample_id"] for r in train],
        "val_samples": [r["sample_id"] for r in val],
        "test_samples": [r["sample_id"] for r in test],
        "batchnorm_note": (f"decoder BatchNorm 统计基于 microbatch "
                           f"{int(s['train']['microbatch'])}，四组一致；不等价历史 batch64"),
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
