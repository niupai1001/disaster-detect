"""模型工厂：按配置构造 SegFormer / U-Net，统一参数分组与 checkpoint 重建入口。

配置字段（``segformer.model``）::

    type: segformer | unet
    adapter: direct10 | project10to3      # 仅 segformer
    backbone: b0 | b2                     # 仅 segformer
    stem: true | false                    # 仅 segformer
    refine: none | guided | zero_cond     # 仅 segformer + stem
    pretrained_dir: ...                   # 仅 segformer
    unet: {encoder_name: resnet18, encoder_weights: imagenet}

旧配置没有 ``type`` 时按 ``segformer`` 处理，保持历史 checkpoint 兼容。
"""

from __future__ import annotations

from pathlib import Path

import torch.nn as nn

from ..config import resolve_path
from .segformer import SegFormerSeg
from .feature_fusion import SingleHeadSeg
from .unet import UNetSeg
from .exchange import EXCHANGE_MODES

MODEL_TYPES = ("segformer", "unet", "single", "dofa", "probe", "temporal_spectral",
               "terrain_fusion", "spectral_fusion", "spectral_relation", "spectral_ladder",
               "spectral_gate", "use")
REFINE_MODES = ("none", "guided", "zero_cond")


def build_model(cfg_segformer: dict, pretrained_dir: str | Path | None = None) -> nn.Module:
    m = cfg_segformer["model"]
    typ = str(m.get("type", "segformer"))
    if typ not in MODEL_TYPES:
        raise ValueError(f"model.type 必须是 {MODEL_TYPES}，得到 {typ!r}")

    if typ == 'use':
        from .segformer_use import SegFormerUSE
        from .use_downstream import build_visual
        mode = m.get('use_downstream', 'bridge10')
        if mode != 'bridge10' and not m.get('freeze_use', False):
            raise ValueError('Independent USE downstream adapters require frozen USE')
        visual, channels = build_visual(mode, str(resolve_path(pretrained_dir or m.get('pretrained_dir', 'weights/segformer/segformer-b0-finetuned-ade-512-512'))))
        model = SegFormerUSE(visual, m.get('use'))
        if channels == 32:
            model.bridge = nn.Identity()
        elif channels != 10:
            model.bridge = nn.Conv2d(model.use.width, channels, 1)
        if mode != 'bridge10':
            model.adapter_report = {'interface': f'USE32 to {mode}', 'visual': visual.adapter_report}
        if m.get('use_pretrained'):
            import torch
            from ..config import file_sha256
            path = resolve_path(m['use_pretrained'])
            if file_sha256(path) != m.get('use_pretrained_sha256'):
                raise ValueError('USE pretraining checkpoint hash differs')
            state = torch.load(path, map_location='cpu', weights_only=False)
            model.use.load_state_dict(state['use'], strict=True)
            model.load_info = {'visual': visual.load_info, 'use_pretrained_sha256': file_sha256(path)}
        if m.get('freeze_use', False):
            model.freeze_use()
        return model

    if typ == "probe":
        from .frozen_probe import FrozenProbe
        from ..data.segformer_dataset import load_stats
        pr = m.get("probe", {}) or {}
        stats = load_stats(resolve_path(cfg_segformer["data"]["stats"]))
        return FrozenProbe(
            encoder=str(pr["encoder"]),
            pretrained_dir=pr.get("pretrained_dir", "weights/segformer/segformer-b0-finetuned-ade-512-512"),
            stats=stats,
            image_size=int(cfg_segformer["data"].get("image_size", 128)),
            wavelength_version=str(pr.get('wavelength_version', 'legacy_v5')),
        )

    if typ == "dofa":
        if float(cfg_segformer["loss"].get("coarse_weight", 0) or 0) > 0:
            raise ValueError("DOFA binary control has no auxiliary coarse head")
        from .dofa import DOFASeg
        return DOFASeg(m)

    if typ == "segformer":
        refine = str(m.get("refine", "none"))
        if refine not in REFINE_MODES:
            raise ValueError(f"model.refine 必须是 {REFINE_MODES}，得到 {refine!r}")
        if refine != "none" and not bool(m.get("stem", False)):
            raise ValueError("model.refine != none 需要 model.stem=true")
        exchange = str(m.get("exchange", "none"))
        if exchange not in EXCHANGE_MODES:
            raise ValueError(f"model.exchange 必须是 {EXCHANGE_MODES}，得到 {exchange!r}")
        if exchange != "none" and not bool(m.get("stem", False)):
            raise ValueError("model.exchange != none 需要 model.stem=true")
        pd = pretrained_dir if pretrained_dir is not None else m.get("pretrained_dir")
        constructor = SegFormerSeg
        extra = {}
        if m.get("probe"):
            if exchange != "none":
                raise ValueError("冻结编码器探针不接受 model.exchange（探针要求 stem=false）")
            from .segformer_probe import FrozenEncoderSegFormer
            from ..config import file_sha256, manifest_identity
            probe = m["probe"]
            if file_sha256(resolve_path(cfg_segformer["data"]["stats"])) != probe["stats_sha256"]:
                raise ValueError("Probe normalization differs from current configuration")
            if manifest_identity(resolve_path(cfg_segformer["data"]["manifest"]))["sha256"] != probe["manifest_sha256"]:
                raise ValueError("Probe manifest differs from current configuration")
            constructor, extra = FrozenEncoderSegFormer, {"probe": probe}
        from ..data.segformer_dataset import S2_10_BANDS
        in_chans = int(m.get("in_chans", len(S2_10_BANDS)))
        band_order = list(m.get("band_order", S2_10_BANDS))
        if len(band_order) != in_chans:
            raise ValueError(f"model.in_chans={in_chans} 与 band_order 长度 {len(band_order)} 不一致")
        return constructor(
            adapter=str(m.get("adapter", "direct10")),
            backbone=str(m.get("backbone", "b0")),
            stem=bool(m.get("stem", False)),
            refine=refine,
            pretrained_dir=str(resolve_path(pd)),
            exchange=exchange,
            exchange_stage=int(m.get("exchange_stage", 2)),
            in_chans=in_chans,
            band_order=band_order,
            **extra,
        )

    if typ == "spectral_ladder":
        from ..data.segformer_dataset import S2_10_BANDS, load_stats
        from .spectral_ladder import SpectralLadderSeg
        sp = m.get("spectral_ladder", {}) or {}
        level = str(sp.get("level", "S1"))
        variant = str(sp.get("variant", "treatment"))
        pd = pretrained_dir if pretrained_dir is not None else m.get("pretrained_dir")
        if level == "S1":
            extra = (["NDVI", "NDMI", "NBR", "REND"] if variant == "treatment"
                     else ["B04b", "B08b", "B8Ab", "B11b"])
            bands = list(S2_10_BANDS) + extra
        else:
            bands = list(S2_10_BANDS)
        visual = SegFormerSeg(
            adapter=str(m.get("adapter", "direct10")),
            backbone=str(m.get("backbone", "b0")),
            stem=True,
            refine="none",
            pretrained_dir=str(resolve_path(pd)),
            in_chans=len(bands),
            band_order=bands,
        )
        st = load_stats(resolve_path(cfg_segformer["data"]["stats"]))
        extra_path = sp.get("extra_stats")
        extra_stats = None
        if extra_path:
            import json
            extra_stats = json.loads(resolve_path(extra_path).read_text(encoding="utf-8"))
            extra_stats["source"] = str(extra_path).replace("\\", "/")
        return SpectralLadderSeg(visual, st, level=level, variant=variant,
                                rank=int(sp.get("rank", 4)), radiometry=str(sp.get("radiometry", "legacy_dn")),
                                extra_stats=extra_stats)

    if typ == "spectral_gate":
        from ..data.segformer_dataset import S2_10_BANDS, load_stats
        from .spectral_gate import SpectralGateSeg
        g = m.get("spectral_gate", {}) or {}
        pd = pretrained_dir if pretrained_dir is not None else m.get("pretrained_dir")
        visual = SegFormerSeg(
            adapter=str(m.get("adapter", "direct10")),
            backbone=str(m.get("backbone", "b0")),
            stem=True,
            refine="none",
            pretrained_dir=str(resolve_path(pd)),
            in_chans=len(S2_10_BANDS),
            band_order=list(S2_10_BANDS),
            capture_decoder=False,
        )
        st = load_stats(resolve_path(cfg_segformer["data"]["stats"]))
        return SpectralGateSeg(
            visual, st,
            fusion=str(g.get("fusion", "learn")),
            patch=int(g.get("patch", 4)),
            width=int(g.get("width", 64)),
            heads=int(g.get("heads", 4)),
            layers=int(g.get("layers", 2)),
            relation_mode=str(g.get("relation_mode", "measured")),
            identity=str(g.get("identity", "wavelength")),
            fixed_gate=float(g.get("fixed_gate", 0.5)),
        )

    if typ == "spectral_relation":
        from ..data.segformer_dataset import S2_10_BANDS, load_stats
        from .spectral_relation import SpectralRelationSeg
        sp = m.get("spectral_relation", {}) or {}
        pd = pretrained_dir if pretrained_dir is not None else m.get("pretrained_dir")
        visual = SegFormerSeg(
            adapter=str(m.get("adapter", "direct10")),
            backbone=str(m.get("backbone", "b0")),
            stem=True,
            refine="none",
            pretrained_dir=str(resolve_path(pd)),
            in_chans=len(S2_10_BANDS),
            band_order=list(S2_10_BANDS),
        )
        st = load_stats(resolve_path(cfg_segformer["data"]["stats"]))
        return SpectralRelationSeg(
            visual, st,
            patch=int(sp.get("patch", 4)),
            width=int(sp.get("width", 64)),
            heads=int(sp.get("heads", 4)),
            layers=int(sp.get("layers", 2)),
            relation_mode=str(sp.get("relation_mode", "measured")),
            identity=str(sp.get("identity", "wavelength")),
        )

    if typ == "spectral_fusion":
        from ..data.segformer_dataset import S2_10_BANDS, load_stats
        from .spectral_fusion import SpectralFusionSeg
        sp = m.get("spectral", {}) or {}
        implementation = str(sp.get("implementation", "legacy_hook"))
        pd = pretrained_dir if pretrained_dir is not None else m.get("pretrained_dir")
        visual = SegFormerSeg(
            adapter=str(m.get("adapter", "direct10")),
            backbone=str(m.get("backbone", "b0")),
            stem=True,
            refine="none",
            pretrained_dir=str(resolve_path(pd)),
            in_chans=len(S2_10_BANDS),
            band_order=list(S2_10_BANDS),
            capture_decoder=implementation != "explicit",
        )
        st = load_stats(resolve_path(cfg_segformer["data"]["stats"]))
        return SpectralFusionSeg(
            visual,
            mean=[float(v) for v in st["mean"][: len(S2_10_BANDS)]],
            std=[float(v) for v in st["std"][: len(S2_10_BANDS)]],
            feature=str(sp.get("feature", "indices")),
            fusion=str(sp.get("fusion", "gated")),
            rank=int(sp.get("rank", 8)),
            implementation=implementation,
            radiometry=str(sp.get("radiometry", "legacy_dn")),
        )

    if typ == "terrain_fusion":
        from ..data.segformer_dataset import S2_10_BANDS
        from .terrain import TerrainFusionSeg
        t = m.get("terrain", {}) or {}
        in_chans = int(m.get("in_chans", len(S2_10_BANDS) + 2))
        if in_chans != len(S2_10_BANDS) + 2:
            raise ValueError("terrain_fusion 需要 in_chans = 十波段 + DEM + SLOPE = 12")
        pd = pretrained_dir if pretrained_dir is not None else m.get("pretrained_dir")
        visual = SegFormerSeg(
            adapter=str(m.get("adapter", "direct10")),
            backbone=str(m.get("backbone", "b0")),
            stem=bool(m.get("stem", True)),
            refine=str(m.get("refine", "none")),
            pretrained_dir=str(resolve_path(pd)),
            in_chans=len(S2_10_BANDS),
            band_order=list(S2_10_BANDS),
        )
        start = str(t.get("start", "random"))
        if start not in ("random", "ssl", "ssl_supervised", "ssl_scoring_full_v2"):
            raise ValueError(f"model.terrain.start 必须是 random/ssl/ssl_supervised，得到 {start!r}")
        weights = t.get("weights")
        if start != "random" and not weights:
            raise ValueError(f"model.terrain.start={start} 需要 model.terrain.weights")
        head_weights = t.get("head_weights") if start == "ssl_supervised" else None
        if start == "ssl_supervised" and not head_weights:
            raise ValueError("model.terrain.start=ssl_supervised 需要 model.terrain.head_weights")
        joint = start == "ssl_scoring_full_v2"
        from ..config import file_sha256
        return TerrainFusionSeg(
            visual,
            n_visual=len(S2_10_BANDS),
            terrain_chans=2,
            terrain_pretrained=False,
            terrain_weights=str(resolve_path(weights)) if weights and not joint else None,
            joint_weights=str(resolve_path(weights)) if joint else None,
            expected_stats_sha256=file_sha256(resolve_path(cfg_segformer["data"]["stats"])) if joint else None,
            expected_pool_sha256=file_sha256(resolve_path(cfg_segformer["data"]["manifest"])) if joint else None,
            load_head=bool(head_weights),
            head_weights=str(resolve_path(head_weights)) if head_weights else None,
            alpha=float(t.get("alpha", 0.01)),
            alpha_learnable=bool(t.get("alpha_learnable", True)),
        )

    if typ == "single":
        if not bool(m.get("stem", True)):
            raise ValueError("model.type=single 需要 model.stem=true（细节分支）")
        pd = pretrained_dir if pretrained_dir is not None else m.get("pretrained_dir")
        return SingleHeadSeg(
            adapter=str(m.get("adapter", "direct10")),
            backbone=str(m.get("backbone", "b0")),
            stem=True,
            fusion=str(m.get("fusion", "sum")),
            width=int(m.get("fusion_width", 64)),
            pretrained_dir=str(resolve_path(pd)),
        )

    u = m.get("unet", {}) or {}
    from ..data.segformer_dataset import S2_10_BANDS
    u_in = int(m.get("in_chans", len(S2_10_BANDS)))
    u_bands = list(m.get("band_order", S2_10_BANDS))
    if len(u_bands) != u_in:
        raise ValueError(f"[unet] model.in_chans={u_in} 与 band_order 长度 {len(u_bands)} 不一致")
    return UNetSeg(
        in_chans=u_in,
        encoder_name=str(u.get("encoder_name", "resnet18")),
        encoder_weights=u.get("encoder_weights", "imagenet"),
    )


def model_type(cfg_segformer: dict) -> str:
    return str(cfg_segformer["model"].get("type", "segformer"))


def model_identity(cfg_segformer: dict, model: nn.Module) -> dict:
    """把构造时的加载/适配信息写入 run_meta / checkpoint。"""
    info = {
        "type": model_type(cfg_segformer),
        "adapter": getattr(model, "adapter_type", None),
        "backbone": getattr(model, "backbone", None),
        "stem": getattr(model, "use_stem", None),
        "refine": getattr(model, "refine", None),
        "fusion": getattr(model, "fusion_mode", None),
        "exchange": getattr(model, "exchange_mode", None),
    }
    if getattr(model, "exchange_report", None):
        info["exchange_report"] = model.exchange_report
    if hasattr(model, "load_info"):
        info["load_info"] = model.load_info
    if hasattr(model, "adapter_report"):
        info["adapter_report"] = model.adapter_report
    if hasattr(model, "probe_report"):
        info["probe"] = model.probe_report
    if hasattr(model, "terrain_report"):
        info["terrain"] = model.terrain_report()
    if hasattr(model, "spectral_report"):
        info["spectral"] = model.spectral_report()
    return info
