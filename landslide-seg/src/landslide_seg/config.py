"""配置加载与合并。

训练入口只认 YAML：模型结构、损失权重、优化器、数据、随机种子、输出目录
全部来自配置文件；命令行只做 `--set a.b=c` 形式的覆盖。
这样"跑出来的东西"和"配置里写的"不会脱节。
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping

import yaml

#: 项目根目录（本文件位于 ``src/landslide_seg/config.py``）
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(p: str | os.PathLike) -> Path:
    """配置里的相对路径一律相对项目根目录解析，与当前工作目录无关。"""
    q = Path(p)
    return q if q.is_absolute() else (PROJECT_ROOT / q)


def file_sha256(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    """完整文件 SHA-256（权重身份用它，不用首尾块近似）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

DEFAULTS: dict[str, Any] = {
    "seed": 1234,
    "deterministic": True,
    "output_dir": "outputs",
    "data": {
        "source": "certified",
        "manifest": "manifests/certified.jsonl",
        "cache_dir": "data/certified",
        "batch_size": 2,
        "boa_offset_dn": 1000.0,   # 加回基线 04.00 的 -1000 DN 偏移（见 verify_radiometry）
        "num_workers": 0,
    },
    "model": {
        "channels": 128,
        "level_strides": [10, 20, 40, 80],
        "anysat_scales_m": [40, 160],
        "tim_enabled": True,
        "branch_dropout": 0.1,
        "freeze_backbones": True,
        "semantic": {
            "class_prompt": "a landslide scar in a satellite image",
            "use_attributes": False,
        },
    },
    "loss": {
        "lambda_coarse": 0.5,
        "lambda_presence": 0.0,
        "lambda_region_text": 0.1,
        "lambda_attribute": 0.0,
        "bce_weight": 0.5,
        "dice_weight": 0.5,
    },
    "optim": {"lr": 1e-4, "weight_decay": 1e-4, "grad_clip": 1.0},
    "train": {"epochs": 1, "steps_per_epoch": None, "log_every": 1},
    "eval": {"threshold": 0.5, "use_presence_gate": False},
}


#: SegFormer 路线独立 schema。刻意不复用 ``DEFAULTS``：旧模型的
#: coarse/presence/文本损失字段对新路线没有意义，混进来只会造成"配置写着但没生效"。
DEFAULTS_SEGFORMER: dict[str, Any] = {
    "segformer": {
        "experiment": "A0",
        "seed": 1234,
        "deterministic": True,
        "output_dir": "outputs/segformer",
        "data": {
            "version": "s2_10band_v1",
            "source_root": "data/sen12",
            "cache_dir": "data/s2_10band_v1",
            "manifest": "manifests/s2_10band_v1.jsonl",
            "quarantine": "manifests/s2_10band_v1_quarantine.jsonl",
            "stats": "data/s2_10band_v1/stats.json",
            "test_event": "italy",
            "val_event": "hiroshima",
            "max_cloud": 0.5,
            "min_valid_ratio": 0.5,
            "max_label_invalid_ratio": 0.6,
            "warn_days_after_event": 365,
            "post_event_window_days": 120,
            "image_size": 128,
            "num_workers": 0,
            "preload": True,
        },
        "model": {
            "type": "segformer",            # segformer | unet
            "backbone": "b0",
            "adapter": "direct10",
            "stem": False,
            "refine": "none",               # none | guided | zero_cond（CF1/CF0）
            "fusion": "sum",                # sum | aff（type=single 时）
            "fusion_width": 64,
            "pretrained_dir": "weights/segformer/segformer-b0-finetuned-ade-512-512",
            "pretrained_source": "nvidia/segformer-b0-finetuned-ade-512-512",
            "unet": {"encoder_name": "resnet18", "encoder_weights": "imagenet"},
        },
        "loss": {
            "bce_weight": 0.5,
            "dice_weight": 0.5,
            "coarse_weight": 0.0,
            "boundary_weight": 0.0,   # >0：精分 BCE 按 GT 边界带加权
            "boundary_width": 2,
        },
        "optim": {
            "encoder_lr": 6.0e-5,
            "head_lr": 6.0e-4,
            "first_conv_lr": None,      # T1：direct10 首层适配的独立学习率；None = 用 encoder_lr
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "warmup_frac": 0.05,
            "schedule": "poly",
            "poly_power": 1.0,
        },
        "sampling": {
            "mode": "natural",          # natural | balanced（T2）
            "event_power": 0.5,
            "scale_power": 0.5,
            "max_weight": 4.0,
        },
        "train": {
            "epochs": 100,
            "batch_size": 8,
            "amp": False,
            "amp_dtype": "bf16",
            "log_every": 20,
            "max_steps": None,
        },
        "eval": {
            "threshold": 0.5,
            "scale_thresholds_ha": [0.5, 2.0],
            "object_iou_threshold": 0.25,
            "object_iou_strict": 0.5,
            "boundary_tol_px": 1,
        },
    }
}


def load_segformer_config(
    path: str | os.PathLike | None = None, overrides: list[str] | None = None
) -> dict:
    """加载 SegFormer 配置（与旧 ``load_config`` 完全隔离）。"""
    cfg = copy.deepcopy(DEFAULTS_SEGFORMER)
    raw: dict = {}
    if path and Path(path).is_file():
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        base_ref = raw.pop("_base", None)
        if base_ref:
            bp = Path(base_ref)
            if not bp.is_absolute():
                bp = Path(path).resolve().parent / bp
            cfg = _deep_merge(cfg, yaml.safe_load(bp.read_text(encoding="utf-8")) or {})
        cfg = _deep_merge(cfg, raw)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set 需要 key=value 形式，得到 {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(value)
    return cfg


def _deep_merge(base: dict, over: Mapping) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return text


def load_config(path: str | os.PathLike | None = None, overrides: list[str] | None = None) -> dict:
    """加载 YAML 并套用 ``key.sub=value`` 覆盖。"""
    cfg = copy.deepcopy(DEFAULTS)
    if path and Path(path).is_file():
        cfg = _deep_merge(cfg, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set 需要 key=value 形式，得到 {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(value)
    return cfg


def set_seed(seed: int, deterministic: bool = True) -> dict:
    """固定随机状态，返回实际生效的设置（写进 checkpoint）。"""
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # AnySat 的注意力在开启 TF32 时数值会变，固定下来便于复现
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return {
        "seed": seed,
        "deterministic": deterministic,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    }


def file_identity(path: str | os.PathLike, chunk_mb: int = 8) -> dict:
    """权重身份：完整 SHA-256 太慢时用"大小 + 首尾块哈希"，仍足以区分版本。"""
    import hashlib

    p = Path(path)
    if not p.is_file():
        return {"path": str(p), "exists": False}
    size = p.stat().st_size
    chunk = chunk_mb * 1024 * 1024
    h = hashlib.sha256()
    with p.open("rb") as f:
        h.update(f.read(chunk))
        if size > 2 * chunk:
            f.seek(-chunk, os.SEEK_END)
            h.update(f.read(chunk))
    return {
        "path": str(p).replace("\\", "/"),
        "size_bytes": size,
        "sha256_head_tail": h.hexdigest()[:32],
    }


def manifest_identity(path: str | os.PathLike) -> dict:
    import hashlib

    p = Path(path)
    if not p.is_file():
        return {"path": str(p), "exists": False}
    data = p.read_bytes()
    ids = []
    for line in data.decode("utf-8").splitlines():
        if line.strip():
            try:
                ids.append(json.loads(line)["sample_id"])
            except Exception:
                pass
    return {
        "path": str(p).replace("\\", "/"),
        "sha256": hashlib.sha256(data).hexdigest()[:32],
        "n_samples": len(ids),
        "sample_ids": ids[:200],
    }
