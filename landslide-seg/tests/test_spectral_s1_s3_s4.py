"""S1 新增通道统计/首层同初始化、S3 五组关系、S4 独立 F_spec 与五融合的行为验收。

这些测试只证明接口、初始化与门控参与计算；不证明物理正确或研究收益。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from landslide_seg.config import load_segformer_config
from landslide_seg.data.segformer_dataset import S2_10_BANDS
from landslide_seg.models.factory import build_model
from landslide_seg.models.segformer import SegFormerSeg
from landslide_seg.models.spectral_gate import SpectralGateSeg
from landslide_seg.models.spectral_ladder import SpectralLadder
from landslide_seg.models.spectral_relation import BandRelationFeature
from landslide_seg.models.spectral_temporal import BandRelationBlock

ROOT = Path(__file__).resolve().parents[1]
STATS = {"mean": [500.0] * 10, "std": [300.0] * 10}
EXTRA = {"mean": [0.61, 0.17, 0.40, 0.31], "std": [0.23, 0.18, 0.22, 0.16]}


def test_s1_treatment_applies_extra_stats_and_flags_historical_unnormalized():
    legacy = SpectralLadder("S1", "treatment", STATS).eval()
    assert legacy.report()["extra_stats"] is None
    assert legacy.report()["mechanism"] == "stored_dn_indices_unnormalized_historical"
    ladder = SpectralLadder("S1", "treatment", STATS, extra_stats=EXTRA,
                            radiometry="harmonized_dn_v1").eval()
    x = torch.randn(2, 10, 16, 16)
    with torch.no_grad():
        out = ladder(x)
        raw = ladder._indices(x)
        want = (raw - torch.tensor(EXTRA["mean"])[None, :, None, None]) / torch.tensor(EXTRA["std"])[None, :, None, None]
    assert out.shape == (2, 14, 16, 16)
    assert torch.allclose(out[:, 10:], want, atol=1e-6)
    assert ladder.report()["extra_stats"]["mean"] == pytest.approx(EXTRA["mean"])
    assert ladder.report()["mechanism"] == "harmonized_indices_normalized"


def test_s1_control_uses_dataset_normalized_raw_bands():
    ladder = SpectralLadder("S1", "control", STATS).eval()
    x = torch.randn(2, 10, 16, 16)
    with torch.no_grad():
        out = ladder(x)
    idx = [S2_10_BANDS.index(k) for k in ("B04", "B08", "B8A", "B11")]
    assert torch.equal(out[:, 10:], x[:, idx])


def test_s3_channel_identity_has_same_params_as_wavelength_identity():
    kwargs = dict(width=16, heads=4, layers=1, stats=STATS, relation_mode="none")
    wave = BandRelationBlock(identity="wavelength", **kwargs)
    chan = BandRelationBlock(identity="channel", **kwargs)
    assert sum(p.numel() for p in wave.parameters()) == sum(p.numel() for p in chan.parameters())
    z, gh, gw = chan.tokens(torch.randn(2, 10, 8, 8))
    assert z.shape == (2, 4, 10, 16) and (gh, gw) == (2, 2)


def test_s3_global_bias_is_shared_across_patches_local_is_not():
    torch.manual_seed(5)
    kwargs = dict(width=16, heads=4, layers=1, stats=STATS, identity="wavelength")
    with torch.no_grad():
        local = BandRelationBlock(relation_mode="measured", **kwargs)
        glob = BandRelationBlock(relation_mode="global", **kwargs)
        glob.load_state_dict(local.state_dict())
        for block in (local, glob):
            torch.nn.init.normal_(block.relation[-1].weight, std=0.1)
            torch.nn.init.normal_(block.relation[-1].bias, std=0.1)
        x = torch.randn(1, 10, 8, 8)
        bl = local.relation_bias(local.reflectance(x))
        bg = glob.relation_bias(glob.reflectance(x))
    assert bl.shape == bg.shape == (1, 4, 4, 10, 10)
    assert (bl[0, 0] - bl[0, 1]).abs().max() > 1e-6
    assert (bg[0, 0] - bg[0, 1]).abs().max() < 1e-6


def test_s4_feature_shape_and_joint_band_permutation_invariance():
    torch.manual_seed(7)
    feat = BandRelationFeature(out_ch=8, width=16, heads=4, layers=1, stats=STATS,
                               relation_mode="none", identity="wavelength")
    x = torch.randn(2, 10, 8, 8)
    with torch.no_grad():
        a0 = feat(x)
        assert a0.shape == (2, 8, 2, 2) and a0.abs().sum() > 0
        perm = torch.tensor([4, 2, 6, 8, 0, 9, 3, 1, 7, 5])
        permuted = copy.deepcopy(feat)
        for name in ("waves", "dn_mean", "dn_std"):
            getattr(permuted.block, name).copy_(getattr(feat.block, name)[perm])
        a = feat(x)
        b = permuted(x[:, perm])
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)


def _gate_arm(fusion: str, identity: str = "wavelength", relation: str = "measured"):
    torch.manual_seed(11)
    visual = SegFormerSeg(adapter="direct10", backbone="b0", stem=True, refine="none",
                          in_chans=10, capture_decoder=False,
                          pretrained_dir=str(ROOT / "weights/segformer/segformer-b0-finetuned-ade-512-512"))
    return SpectralGateSeg(visual, STATS, fusion=fusion, patch=4, width=16, heads=4, layers=1,
                           relation_mode=relation, identity=identity)


def test_s4_initial_fusion_is_exactly_the_visual_model():
    model = _gate_arm("learn").eval()
    x = torch.randn(2, 10, 128, 128)
    hook = model.visual.model.decode_head.classifier.register_forward_pre_hook(model.visual._capture)
    with torch.no_grad():
        out = model(x)
        base = model.visual(x)["logits"]
    hook.remove()
    assert out["spectral_info"]["alpha"] == 0.0
    assert out["spectral_info"]["corr_abs_mean"] == 0.0
    assert torch.equal(out["logits"], base)


def test_s4_fusion_modes_and_gate_logging():
    torch.manual_seed(3)
    x = torch.randn(2, 10, 128, 128)
    add = _gate_arm("add").eval()
    fixed = _gate_arm("fixed").eval()
    learn = _gate_arm("learn").eval()
    with torch.no_grad():
        # 让支路离开零初始化，比较融合方式而不是初始化
        for m in (add, fixed, learn):
            m.feature.proj.weight.normal_(std=0.02)
            m.alpha.fill_(1.0)
        learn.gate.weight.normal_(std=0.01)
        learn.gate.bias.fill_(0.0)
        a, f, l = add(x), fixed(x), learn(x)
    assert add._last_gate is None
    assert torch.allclose(fixed._last_gate, torch.full_like(fixed._last_gate, 0.5))
    assert learn._last_gate.mean() > 0.0 and learn._last_gate.std() > 0.0
    assert a["spectral_info"]["gate_mean"] is None
    assert f["spectral_info"]["gate_mean"] == pytest.approx(0.5, abs=1e-6)
    assert l["spectral_info"]["gate_var"] is not None
    # 同权重下三种融合给出不同修正
    assert not torch.equal(a["logits"], f["logits"])
    assert not torch.equal(f["logits"], l["logits"])


def test_s4_capctrl_matches_learn_gate_effective_capacity():
    learn = _gate_arm("learn")
    cap = _gate_arm("learn", identity="channel", relation="none")
    nl = learn.count_params()
    nc = cap.count_params()
    assert nl["feature_branch"] == nc["feature_branch"]
    assert nl["gate"] == nc["gate"]
    assert nl["trainable"] == nc["trainable"]
    # 存储参数相同；capctrl 的关系偏置 MLP 在 none 模式下不参与前向，
    # 因此有效可训练参数正好少 132 个（约 0.08%）。
    assert nc["unused_relation_params"] == 132
    assert nl["effective_trainable"] - nc["effective_trainable"] == 132
    assert cap.feature.block.relation_mode == "none"


def test_configs_build_and_declare_expected_mechanisms():
    expected = {
        "s1_indices": ("spectral_ladder", {"level": "S1", "variant": "treatment",
                                           "radiometry": "harmonized_dn_v1"}),
        "s1_control": ("spectral_ladder", {"level": "S1", "variant": "control"}),
        "s3_a_linear": ("spectral_ladder", {"level": "S3", "variant": "control"}),
        "s3_b_idattn": ("spectral_relation", {"identity": "channel", "relation_mode": "none"}),
        "s3_c_wavelength_attn": ("spectral_relation", {"identity": "wavelength", "relation_mode": "none"}),
        "s3_d_local_bias": ("spectral_relation", {"identity": "wavelength", "relation_mode": "measured"}),
        "s3_e_global_bias": ("spectral_relation", {"identity": "wavelength", "relation_mode": "global"}),
        "s4_add": ("spectral_gate", {"fusion": "add"}),
        "s4_fixed_gate": ("spectral_gate", {"fusion": "fixed", "fixed_gate": 0.5}),
        "s4_learn_gate": ("spectral_gate", {"fusion": "learn"}),
        "s4_capctrl": ("spectral_gate", {"identity": "channel", "relation_mode": "none",
                                         "fusion": "learn"}),
    }
    for name, (mtype, fields) in expected.items():
        cfg = load_segformer_config(ROOT / "configs/segformer/v4r2_spectral" / f"{name}.yaml")
        model_cfg = cfg["segformer"]["model"]
        assert model_cfg["type"] == mtype
        spec = model_cfg.get("spectral_ladder") or model_cfg.get("spectral_relation") \
            or model_cfg.get("spectral_gate") or {}
        for key, value in fields.items():
            assert spec[key] == value, (name, key, spec.get(key))
        assert cfg["segformer"]["train"]["stop_updates"] == 2000
        assert cfg["segformer"]["train"]["schedule_updates"] == 3600
        assert cfg["segformer"]["data"]["stats"] == "manifests/v4r1/stats_core25.json"
    assert "extra_stats" in load_segformer_config(
        ROOT / "configs/segformer/v4r2_spectral/s1_indices.yaml")["segformer"]["model"]["spectral_ladder"]
