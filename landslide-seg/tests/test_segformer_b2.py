"""B2 骨干回归测试（权重存在时才运行）。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
B2 = ROOT / "weights" / "segformer" / "segformer-b2-finetuned-ade-512-512"

pytestmark = pytest.mark.skipif(not (B2 / "config.json").is_file(), reason="B2 权重未下载")


def test_b2_decoder_hidden_and_forward():
    from landslide_seg.models.segformer import SegFormerSeg

    model = SegFormerSeg(
        backbone="b2", adapter="direct10", stem=False, pretrained_dir=str(B2)
    ).eval()
    assert int(model.model.config.decoder_hidden_size) == 768
    with torch.no_grad():
        out = model(torch.randn(1, 10, 128, 128))
    assert tuple(out["logits"].shape) == (1, 1, 128, 128)
    assert not model.load_info["mismatched_keys"]


def test_b2_with_stem_and_refiner_builds():
    from landslide_seg.models.segformer import SegFormerSeg

    model = SegFormerSeg(
        backbone="b2", adapter="direct10", stem=True, refine="guided",
        pretrained_dir=str(B2),
    ).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 10, 128, 128), return_delta=True)
    assert float(out["delta_logits"].abs().max()) == 0.0
    assert tuple(out["logits"].shape) == (1, 1, 128, 128)


def test_b2_rejects_b0_directory():
    from landslide_seg.models.segformer import SegFormerSeg

    b0 = ROOT / "weights" / "segformer" / "segformer-b0-finetuned-ade-512-512"
    if not b0.is_dir():
        pytest.skip("B0 权重不存在")
    with pytest.raises(ValueError, match="768"):
        SegFormerSeg(backbone="b2", adapter="direct10", pretrained_dir=str(b0))
