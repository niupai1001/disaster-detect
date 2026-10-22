"""SegFormer 二分类薄封装（MiT-B0/B2 + 官方 SegFormer 解码器）。

只做四件事，不复制任何模型算法：
1. **输入适配**（执行计划 §3）
   - ``direct10``：把第一层 patch embedding 的 3 通道卷积换成 10 通道，
     R/G/B 权重原样拷到 B04/B03/B02 通道，其余 7 个通道零初始化（可训练）；
   - ``project10to3``：前置一个可训练 ``Conv2d(10,3,1)``，初始为单位选择器
     （B04/B03/B02），其余系数与 bias 为 0。
   两条路径在初始化时都**严格等价于**"把对应 RGB 通道送给原三通道模型"，
   该等价性由 ``tests/test_segformer_adapter.py`` 验证。
2. 把 ADE 的 150 类分类器替换成单 logit 分类器（二分类，背景=0/滑坡=1）。
3. 取官方解码器分类器之前的特征 ``D``（受控 forward hook），供 Stem 融合使用。
4. 输出统一恢复到输入尺寸 ``[B,1,H,W]``。

不做的事：不启用图像处理器的 resize/rescale=1/255，不 reduce_labels，
不使用多类损失，不为 512 权重把 128 影像放大。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral_stem import CoarseGuidedRefiner, ResidualFusionHead, SpectralStem
from .exchange import EXCHANGE_MODES, ChannelwiseResidualExchange

#: RGB→本项目 10 波段的位置映射：R=B04(2), G=B03(1), B=B02(0)
RGB_TO_BAND_INDEX = {"R": 2, "G": 1, "B": 0}

#: 骨干规格：解码器隐藏维必须与加载到的权重一致。
#: ``backbone`` 是配置项，必须真正参与校验，否则 ``b2`` 会静默加载成 b0。
BACKBONE_DECODER_HIDDEN = {"b0": 256, "b2": 768}


def _load_pretrained(pretrained_dir: str | Path):
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    p = Path(pretrained_dir)
    if not p.is_dir():
        raise FileNotFoundError(
            f"SegFormer 预训练权重目录不存在: {p}\n"
            "执行计划要求显式获取并锁定权重；请先下载 "
            "nvidia/segformer-b0-finetuned-ade-512-512。"
        )
    model, info = SegformerForSemanticSegmentation.from_pretrained(
        str(p), output_loading_info=True
    )
    return model, info


def _replace_classifier(model, decoder_hidden: int) -> dict:
    """150 类 → 单 logit。返回替换记录（用于权重加载报告）。"""
    old = model.decode_head.classifier
    new = nn.Conv2d(decoder_hidden, 1, kernel_size=1, bias=True)
    nn.init.normal_(new.weight, std=0.01)
    nn.init.zeros_(new.bias)
    model.decode_head.classifier = new
    cfg = model.config
    cfg.num_labels = 1
    cfg.id2label = {0: "background"}
    cfg.label2id = {"background": 0}
    return {
        "replaced": "decode_head.classifier",
        "old": f"{type(old).__name__}({old.in_channels}->{old.out_channels})",
        "new": f"Conv2d({decoder_hidden}->1)",
        "note": "ADE 150 类分类器被替换；其余解码器与编码器权重全部复用",
    }


def adapt_first_conv_direct10(model, in_chans: int, band_order: list[str]) -> dict:
    """把 patch_embeddings[0].proj 换成 in_chans 通道，并精确拷贝 RGB 权重。"""
    emb = model.segformer.encoder.patch_embeddings[0]
    old: nn.Conv2d = emb.proj
    assert isinstance(old, nn.Conv2d) and old.in_channels == 3, old
    new = nn.Conv2d(
        in_chans, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None
    )
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, RGB_TO_BAND_INDEX["R"]] = old.weight[:, 0]   # R → B04
        new.weight[:, RGB_TO_BAND_INDEX["G"]] = old.weight[:, 1]   # G → B03
        new.weight[:, RGB_TO_BAND_INDEX["B"]] = old.weight[:, 2]   # B → B02
        if old.bias is not None:
            new.bias.copy_(old.bias)
    emb.proj = new
    model.config.num_channels = in_chans   # 保存/重载时配置与权重一致
    zeros = [i for i in range(in_chans) if i not in RGB_TO_BAND_INDEX.values()]
    return {
        "type": "direct10",
        "old": f"Conv2d(3,{old.out_channels},k={tuple(old.kernel_size)},s={tuple(old.stride)})",
        "new": f"Conv2d({in_chans},{old.out_channels})",
        "rgb_mapping": {"R": band_order[2], "G": band_order[1], "B": band_order[0]},
        "zero_init_channels": [band_order[i] for i in zeros],
        "note": "新增 7 通道零初始化 → 初始响应严格等于原 RGB 模型",
    }


class SpectralProjection(nn.Module):
    """``Conv2d(10,3,1)``：初始为 B04/B03/B02 选择器，全部参数可训练。"""

    def __init__(self, in_chans: int = 10, band_order: list[str] | None = None) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, 3, kernel_size=1, bias=True)
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()
            self.proj.weight[0, RGB_TO_BAND_INDEX["R"], 0, 0] = 1.0
            self.proj.weight[1, RGB_TO_BAND_INDEX["G"], 0, 0] = 1.0
            self.proj.weight[2, RGB_TO_BAND_INDEX["B"], 0, 0] = 1.0
        self.band_order = band_order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    def describe(self) -> dict:
        w = self.proj.weight.detach().cpu().squeeze(-1).squeeze(-1)   # [3,10]
        return {
            "type": "project10to3",
            "shape": "Conv2d(10,3,1)",
            "rgb_mapping": {"R": 2, "G": 1, "B": 0},
            "initial_R_row": w[0].tolist(),
            "initial_G_row": w[1].tolist(),
            "initial_B_row": w[2].tolist(),
            "note": "非 RGB 系数初始为 0，训练后可变为非零；导出时报告其变化",
        }


class SegFormerSeg(nn.Module):
    """10 波段单期输入 → 单 logit 滑坡概率。"""

    def __init__(
        self,
        *,
        adapter: str = "direct10",
        backbone: str = "b0",
        stem: bool = False,
        refine: str = "none",
        pretrained_dir: str = "weights/segformer/segformer-b0-finetuned-ade-512-512",
        in_chans: int = 10,
        band_order: list[str] | None = None,
        stem_width: int = 32,
        exchange: str = "none",
        exchange_stage: int = 2,
    ) -> None:
        super().__init__()
        from landslide_seg.data.segformer_dataset import S2_10_BANDS

        self.band_order = list(band_order or S2_10_BANDS)
        assert len(self.band_order) == in_chans, self.band_order
        if adapter not in ("direct10", "project10to3"):
            raise ValueError(f"adapter 必须是 direct10/project10to3，得到 {adapter!r}")
        if backbone not in BACKBONE_DECODER_HIDDEN:
            raise ValueError(
                f"backbone 必须是 {sorted(BACKBONE_DECODER_HIDDEN)}，得到 {backbone!r}"
            )

        base, load_info = _load_pretrained(pretrained_dir)
        expected_hidden = BACKBONE_DECODER_HIDDEN[backbone]
        got_hidden = int(base.config.decoder_hidden_size)
        if got_hidden != expected_hidden:
            raise ValueError(
                f"backbone={backbone!r} 期望解码器 {expected_hidden} 维，"
                f"但 {pretrained_dir} 是 {got_hidden} 维；请检查 model.pretrained_dir 与 "
                "model.backbone 是否匹配（B2 必须配 B2 权重）"
            )
        self.backbone = backbone
        self.in_chans = in_chans
        self.adapter_type = adapter
        self.use_stem = bool(stem)
        if refine not in ("none", "guided", "zero_cond"):
            raise ValueError(f"refine 必须是 none/guided/zero_cond，得到 {refine!r}")
        if refine != "none" and not self.use_stem:
            raise ValueError("refine != none 需要 stem=True（精分头要拼接原网格特征）")
        self.refine = refine
        decoder_hidden = base.config.decoder_hidden_size

        self.cls_report = _replace_classifier(base, decoder_hidden)
        self.model = base

        self.projection: SpectralProjection | None = None
        if adapter == "direct10":
            self.adapter_report = adapt_first_conv_direct10(base, in_chans, self.band_order)
        else:
            self.projection = SpectralProjection(in_chans, self.band_order)
            self.adapter_report = self.projection.describe()

        if self.use_stem:
            self.stem = SpectralStem(in_chans, stem_width)
            self.fusion = ResidualFusionHead(decoder_hidden, stem_width)
        else:
            self.stem = None
            self.fusion = None

        # v4r4：编码阶段单次通道残差交换（self 为容量对照，cross 为真交换）
        if exchange not in EXCHANGE_MODES:
            raise ValueError(f"model.exchange 必须是 {EXCHANGE_MODES}，得到 {exchange!r}")
        if exchange != "none" and not self.use_stem:
            raise ValueError("model.exchange != none 需要 model.stem=true（交换的 D 即原网格 Stem）")
        if exchange != "none" and int(exchange_stage) != 2:
            raise ValueError("本阶段只实现 exchange_stage=2（阶段 2 之后、阶段 3 之前）")
        self.exchange_mode = exchange
        self.exchange_stage = int(exchange_stage)
        self._stage_shapes: dict | None = None
        if exchange != "none":
            self.exchange = ChannelwiseResidualExchange(
                e_ch=int(base.config.hidden_sizes[1]),
                d_ch=int(stem_width),
                mode=exchange,
            )
            self.exchange_report = {
                **self.exchange.describe(),
                "position": "after encoder stage 2 (before stage 3)",
                "consumes": "stage-2 feature replaced for both stage-3 input and decoder",
            }
        else:
            self.exchange = None
            self.exchange_report = None

        if refine != "none":
            self.refiner = CoarseGuidedRefiner(decoder_hidden, stem_width)
        else:
            self.refiner = None

        self.load_info = {
            "missing_keys": list(load_info.get("missing_keys", [])),
            "unexpected_keys": list(load_info.get("unexpected_keys", [])),
            "mismatched_keys": [str(k) for k in load_info.get("mismatched_keys", [])],
            "error_msgs": list(load_info.get("error_msgs", [])),
        }
        self._decoder_feat: torch.Tensor | None = None
        # 受控特征捕获：分类器之前的那一层输出就是官方解码器特征 D
        self.model.decode_head.classifier.register_forward_pre_hook(self._capture)

    def _capture(self, _module, args):
        self._decoder_feat = args[0]

    # ------------------------------------------------------------------
    def _forward_staged(self, x: torch.Tensor, stem_feat: torch.Tensor):
        """手工逐阶段跑编码器，在阶段 2 之后插入一次交换。

        与 ``SegformerEncoder.forward`` 的差别只有两处，其余逐行照抄官方实现：
        阶段 2 的输出先过交换块，``E_new`` 既作为阶段 3 的输入，也占据
        ``hidden_states[1]`` 交给解码器；阶段 1/3/4 的输出原样传入。
        """
        enc = self.model.segformer.encoder
        cfg = self.model.config
        n_stage = len(enc.patch_embeddings)
        if n_stage != 4:
            raise RuntimeError(f"本实现假定 4 个编码阶段，得到 {n_stage}")
        batch = x.shape[0]
        hidden = x
        stages: list[torch.Tensor] = []
        shapes: dict[str, list[int]] = {"input": list(x.shape)}
        for idx in range(n_stage):
            hidden, height, width = enc.patch_embeddings[idx](hidden)
            for layer in enc.block[idx]:
                hidden = layer(hidden, height, width, False)[0]
            hidden = enc.layer_norm[idx](hidden)
            if idx != n_stage - 1 or cfg.reshape_last_stage:
                hidden = (
                    hidden.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )
            if idx == self.exchange_stage - 1:
                e_before, d_before = list(hidden.shape), list(stem_feat.shape)
                hidden, stem_feat = self.exchange(hidden, stem_feat)
                shapes["E_in"], shapes["E_out"] = e_before, list(hidden.shape)
                shapes["D_in"], shapes["D_out"] = d_before, list(stem_feat.shape)
            shapes[f"stage{idx + 1}"] = list(hidden.shape)
            stages.append(hidden)
        self._stage_shapes = shapes
        logits = self.model.decode_head(tuple(stages))
        return logits, stem_feat, shapes

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        *,
        return_features: bool = False,
        return_delta: bool = False,
        cond_override: torch.Tensor | None = None,
    ) -> dict:
        if x.ndim != 4 or x.shape[1] != self.in_chans:
            raise ValueError(f"期望 [B,{self.in_chans},H,W]，得到 {tuple(x.shape)}")
        h, w = x.shape[-2:]

        self._decoder_feat = None

        if self.exchange is not None:
            # 交换路径必须先算出 D，才能在阶段 2 之后使用
            stem_feat = self.stem(x) if self.use_stem else None
            out_logits, stem_feat, _ = self._forward_staged(x, stem_feat)
        else:
            if self.projection is not None:
                out_logits = self.model(pixel_values=self.projection(x)).logits
            else:
                out_logits = self.model(pixel_values=x).logits
            # 保持 v4r1 的算子顺序（stem 在主干之后）：顺序一变，cuDNN 选到的
            # 卷积实现会变，同一 checkpoint 的 DEV 会差 1e-5 量级。
            # 历史 B0 结果只有在完全相同的顺序下才能逐位复现。
            stem_feat = self.stem(x) if self.use_stem else None

        feat = self._decoder_feat
        if feat is None:
            raise RuntimeError("未能捕获解码器特征（transformers 版本结构可能变化）")

        base_logits = F.interpolate(out_logits, size=(h, w), mode="bilinear", align_corners=False)
        logits = base_logits
        delta = None
        if self.use_stem:
            if self.refine == "none":
                delta = self.fusion(feat, stem_feat, (h, w))
            else:
                if self.refine == "zero_cond":
                    cond = torch.zeros_like(base_logits)
                elif cond_override is not None:
                    cond = cond_override
                else:
                    cond = torch.sigmoid(base_logits)
                delta = self.refiner(feat, stem_feat, cond, (h, w))
            logits = base_logits + delta

        res = {"logits": logits, "base_logits": base_logits}
        if self.refine != "none":
            res["coarse_prob"] = torch.sigmoid(base_logits)
        if return_delta:
            res["delta_logits"] = delta
        if return_features:
            res["decoder_feat"] = feat
        return res

    # ------------------------------------------------------------------
    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        """编码器 / 解码器 / 新增层 三分组；四组实验共用这一规则。"""
        enc, dec, new = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("model.segformer."):
                enc.append(p)
            elif name.startswith("model.decode_head."):
                dec.append(p)
            else:
                new.append(p)
        return {"encoder": enc, "decoder": dec, "new": new}

    def count_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        g = self.param_groups()
        return {
            "total": total,
            "trainable": trainable,
            "encoder": sum(p.numel() for p in g["encoder"]),
            "decoder": sum(p.numel() for p in g["decoder"]),
            "new": sum(p.numel() for p in g["new"]),
        }

    def projection_change(self) -> dict | None:
        """B 组专用：报告初始选择器之外系数被更新的幅度。

        注意区分两类"非选择器"系数：
        - ``non_rgb_bands``：输入列 3–9（B05/B06/B07/B08/B8A/B11/B12），
          这才是"七个额外波段是否参与学习"的依据；
        - ``outside_selector``：除三个 RGB 选择器位置外的全部系数，
          其中还包含 RGB 通道之间的交叉混合，不能当作"非 RGB 波段"。
        """
        if self.projection is None:
            return None
        w = self.projection.proj.weight.detach().cpu().squeeze(-1).squeeze(-1)
        base = torch.zeros(3, self.in_chans)
        base[0, 2] = base[1, 1] = base[2, 0] = 1.0
        delta = (w - base).abs()
        mask = torch.ones_like(delta, dtype=torch.bool)
        mask[0, 2] = mask[1, 1] = mask[2, 0] = False
        non_rgb_cols = [i for i in range(self.in_chans) if i not in RGB_TO_BAND_INDEX.values()]
        band_mask = torch.zeros_like(delta, dtype=torch.bool)
        band_mask[:, non_rgb_cols] = True
        return {
            "max_abs_change_non_rgb_bands": float(delta[band_mask].max()),
            "mean_abs_change_non_rgb_bands": float(delta[band_mask].mean()),
            "max_abs_change_outside_selector": float(delta[mask].max()),
            "mean_abs_change_outside_selector": float(delta[mask].mean()),
            "bias_abs": self.projection.proj.bias.detach().cpu().abs().tolist(),
            "weight": w.tolist(),
        }
