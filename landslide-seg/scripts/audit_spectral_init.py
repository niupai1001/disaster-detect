"""S1 首层同初始化审计：处理组、对照组与十波段基线逐张量对照。

检查三件事，全部只读：

1. 处理组与对照组的完整 ``visual`` 状态字典逐键、逐值相等（同 seed、同形状）；
2. 相对十波段基线，编码器/解码器所有预训练参数逐值相等，唯一例外是
   ``patch_embeddings.0.proj.weight``；
3. 该首层卷积的 B04/B03/B02 列与基线相同，新增四列严格为零初始化。

输出 ``outputs/segformer/v4r2_spectral/s1_init_audit.json``，不训练、不写权重。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from landslide_seg.config import load_segformer_config  # noqa: E402
from landslide_seg.models.factory import build_model  # noqa: E402
from landslide_seg.models.segformer import SegFormerSeg  # noqa: E402

OUT = ROOT / "outputs" / "segformer" / "v4r2_spectral" / "s1_init_audit.json"
FIRST_KEY = "model.segformer.encoder.patch_embeddings.0.proj.weight"


def build_baseline():
    torch.manual_seed(1234)
    return SegFormerSeg(adapter="direct10", backbone="b0", stem=True, refine="none",
                        in_chans=10, pretrained_dir=str(ROOT / "weights/segformer/segformer-b0-finetuned-ade-512-512"))


def build_arm(name: str):
    torch.manual_seed(1234)
    cfg = load_segformer_config(ROOT / "configs/segformer/v4r2_spectral" / f"{name}.yaml")
    return build_model(cfg["segformer"])


def digest(state: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def compare_state(a: dict, b: dict, ignore: set[str]) -> dict:
    keys_a, keys_b = set(a), set(b)
    if keys_a != keys_b:
        raise SystemExit(f"状态字典键不同: {sorted(keys_a ^ keys_b)[:5]}")
    mismatched = []
    for k in sorted(keys_a - ignore):
        if a[k].shape != b[k].shape or not torch.equal(a[k], b[k]):
            mismatched.append({"key": k, "shape_a": list(a[k].shape), "shape_b": list(b[k].shape)})
    return {"n_keys": len(keys_a), "ignored": sorted(ignore), "mismatched": mismatched,
            "max_abs_diff": max((float((a[k] - b[k]).abs().max()) for k in keys_a - ignore
                                 if a[k].shape == b[k].shape), default=0.0)}


def main() -> None:
    baseline = build_baseline()
    treat = build_arm("s1_indices")
    control = build_arm("s1_control")
    base_state = baseline.state_dict()
    a = treat.visual.state_dict()
    b = control.visual.state_dict()
    wa, wb = a[FIRST_KEY], base_state[FIRST_KEY]
    # 只比较预训练主干/解码器；stem 与 fusion 是随机初始化细节分支，且 stem 首层
    # 通道数随输入变化（14 vs 10），不能要求逐值相同。
    pretrained = {k for k in base_state
                  if k.startswith(("model.segformer.", "model.decode_head."))}
    baseline_ignore = (set(base_state) - pretrained) | {FIRST_KEY}
    bridge = {
        "treatment_vs_control": compare_state(a, b, set()),
        "treatment_vs_baseline": compare_state(a, base_state, baseline_ignore),
        "random_init_components": ["stem", "fusion"],
        "pretrained_keys_compared": len(pretrained) - 1,
        "first_layer": {
            "shape_treatment": list(wa.shape),
            "shape_baseline": list(wb.shape),
            "base_columns_equal": bool(torch.equal(wa[:, :10], wb)),
            "extra_columns_abs_sum": float(wa[:, 10:].abs().sum()),
            "zero_init_channels": ["NDVI", "NDMI", "NBR", "REND"],
            "rgb_mapping": {"R": 2, "G": 1, "B": 0},
        },
        "digests": {"treatment_visual": digest(a), "control_visual": digest(b),
                    "baseline": digest(base_state)},
        "note": ("处理组与对照组 visual 完全相同；相对基线只有 14 通道首层卷积不同，"
                 "其预训练三列保持不变、新增四列零初始化。stem 首层通道数随输入变化，"
                 "属于随机初始化细节分支，不是预训练主干；本审计不比较它。"),
    }
    for key, value in (("treatment_vs_control", bridge["treatment_vs_control"]),
                       ("treatment_vs_baseline", bridge["treatment_vs_baseline"])):
        if value["mismatched"]:
            raise SystemExit(f"{key} 不一致: {value['mismatched'][:3]}")
    if not bridge["first_layer"]["base_columns_equal"]:
        raise SystemExit("首层预训练三列与基线不一致")
    if bridge["first_layer"]["extra_columns_abs_sum"] != 0.0:
        raise SystemExit("首层新增四列不是零初始化")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(bridge, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "treatment_vs_control": bridge["treatment_vs_control"],
                      "treatment_vs_baseline": bridge["treatment_vs_baseline"],
                      "first_layer": bridge["first_layer"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

