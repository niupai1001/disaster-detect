"""按配置构建完整模型（含三个真实预训练权重的加载）。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import torch

from ..encoders.anysat_scale import AnySatScaleAdapter
from ..encoders.terramind_tim import TerraMindTiMAdapter
from ..models.semantic import RemoteCLIPTextEncoder, SemanticAdapter  # noqa: F401
from ..models.semantic_dynamic import ATTRIBUTE_VOCAB, DynamicSemantic
from ..models.full_model import LandslideSegV1

#: 仓库内默认路径（可用环境变量覆盖，禁止硬编码到个人目录）
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS = {
    "anysat": PROJECT_ROOT / "weights" / "AnySat_base.pth",
    "remoteclip": PROJECT_ROOT / "weights" / "RemoteCLIP-ViT-B-32.pt",
    "anysat_repo": PROJECT_ROOT / "vendor" / "AnySat-main",
}


def _resolve(env_key: str, default: Path) -> Path:
    p = Path(os.environ.get(env_key, default))
    if not p.exists():
        raise FileNotFoundError(f"{env_key} 指向的路径不存在: {p}")
    return p


def build_model(
    *,
    channels: int = 128,
    level_strides: Sequence[float] = (10, 20, 40, 80),
    anysat_scales_m: Sequence[int] = (40, 160),
    tim_enabled: bool = True,
    use_attributes: bool = True,
    class_prompts: Sequence[str] | None = None,
    attribute_prompts: Sequence[str] | None = None,
    branch_dropout: float = 0.1,
    freeze_backbones: bool = True,
    semantic_mode: str = "full",
    semantic_use_attributes: bool = True,
    semantic_use_relation: bool = True,
    device: str | torch.device = "cpu",
) -> LandslideSegV1:
    """构建完整 V1 模型。三个预训练权重必须真实存在，缺失直接报错。"""
    anysat_w = _resolve("ANYSAT_WEIGHTS", DEFAULT_WEIGHTS["anysat"])
    anysat_repo = _resolve("ANYSAT_REPO", DEFAULT_WEIGHTS["anysat_repo"])
    rc_w = _resolve("REMOTECLIP_WEIGHTS", DEFAULT_WEIGHTS["remoteclip"])

    tm = TerraMindTiMAdapter(tim_enabled=tim_enabled, freeze=freeze_backbones)
    ans = AnySatScaleAdapter(
        repo_dir=anysat_repo,
        weight_path=anysat_w,
        patch_sizes_m=anysat_scales_m,
        freeze=freeze_backbones,
    )
    text = RemoteCLIPTextEncoder(rc_w)
    names = list(ATTRIBUTE_VOCAB)
    class_text = list(class_prompts) if class_prompts else ["a landslide scar in a satellite image"]
    sem = DynamicSemantic(
        channels=channels,
        embed_dim=text.embed_dim,
        class_prototypes=text.encode(class_text),
        attribute_embeddings=text.encode([ATTRIBUTE_VOCAB[k] for k in names]),
        attribute_names=names,
        mode=semantic_mode,
        use_attributes=semantic_use_attributes,
        use_relation=semantic_use_relation,
    )

    model = LandslideSegV1(
        terramind=tm,
        anysat=ans,
        semantic=sem,
        channels=channels,
        level_strides=level_strides,
        anysat_scales_m=anysat_scales_m,
        branch_dropout=branch_dropout,
    )
    return model.to(device)


def build_from_config(cfg: dict, device: str | torch.device = "cpu") -> LandslideSegV1:
    """完全由配置构建模型（YAML 是唯一事实来源）。"""
    m = cfg["model"]
    sem = m.get("semantic", {})
    ep = [sem["class_prompt"]] if sem.get("class_prompt") else None
    model = build_model(
        channels=int(m["channels"]),
        level_strides=tuple(m["level_strides"]),
        anysat_scales_m=tuple(m["anysat_scales_m"]),
        tim_enabled=bool(m["tim_enabled"]),
        class_prompts=ep,
        attribute_prompts=None,
        branch_dropout=float(m.get("branch_dropout", 0.0)),
        freeze_backbones=bool(m.get("freeze_backbones", True)),
        semantic_mode=str(sem.get("mode", "full")),
        semantic_use_attributes=bool(sem.get("use_attributes", True)),
        semantic_use_relation=bool(sem.get("use_relation", True)),
        device=device,
    )
    ab = cfg.get("ablation", {})
    model.sem_use_align = bool(ab.get("sem_align", True))
    model.sem_use_loc = bool(ab.get("sem_loc", True))
    model.sem_use_ref = bool(ab.get("sem_ref", True))
    return model
