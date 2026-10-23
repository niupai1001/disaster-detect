"""地形有效区目标与评分阶段参数成对迁移的回归验收。"""
import pytest
import torch
from torch import nn

from landslide_seg.models.terrain import (
    TerrainEncoder, TerrainScoreHead, TerrainFusionSeg, valid_block_fraction,
)


def test_fraction_excludes_invalid_and_empty_blocks():
    mask = torch.ones(1, 1, 4, 4)
    valid = torch.zeros_like(mask, dtype=torch.bool)
    valid[:, :, 0, 0] = True
    mask[:, :, 1, 1] = float("nan")
    target, weight = valid_block_fraction(mask, valid, 2)
    assert target[0, 0, 0, 0] == 1
    assert weight[0, 0, 0, 0] == .25
    assert torch.isfinite(target).all()
    assert target.sum() == 1 and weight.sum() == .25


@pytest.fixture
def scoring_checkpoint(tmp_path):
    torch.manual_seed(12)
    encoder, head = TerrainEncoder(2, pretrained=False), TerrainScoreHead()
    with torch.no_grad():
        head.out.weight.fill_(.123)
        encoder.conv1.weight.fill_(.456)
    payload = dict(encoder=encoder.state_dict(), head=head.state_dict(),
                   task="scoring", target_version=2, stats_sha256="stats", pool_sha256="pool")
    path = tmp_path / "scoring.pt"
    torch.save(payload, path)
    return path, payload


def test_full_transfer_loads_same_encoder_and_head(scoring_checkpoint):
    path, payload = scoring_checkpoint
    model = TerrainFusionSeg(nn.Identity(), terrain_pretrained=False, joint_weights=path,
                             expected_stats_sha256="stats", expected_pool_sha256="pool")
    for module, key in ((model.terrain, "encoder"), (model.head, "head")):
        assert all(torch.equal(v, payload[key][k]) for k, v in module.state_dict().items())
    assert model.terrain_init["kind"] == "scoring_full_v2"


@pytest.mark.parametrize("change", ["stats", "pool", "version", "keys"])
def test_full_transfer_rejects_mismatch(scoring_checkpoint, change):
    path, payload = scoring_checkpoint
    if change == "version":
        payload["target_version"] = 1
    elif change == "keys":
        del payload["encoder"]["conv1.weight"]
    torch.save(payload, path)
    with pytest.raises((ValueError, RuntimeError)):
        TerrainFusionSeg(nn.Identity(), terrain_pretrained=False, joint_weights=path,
                         expected_stats_sha256="wrong" if change == "stats" else "stats",
                         expected_pool_sha256="wrong" if change == "pool" else "pool")
