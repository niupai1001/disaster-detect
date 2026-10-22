"""P2-M1 / B2：权重可得性核查 + 容量规格核算（不训练）。

本脚本只做两件事：

1. **核查 B2 官方权重是否可得**：逐个尝试直连、代理、镜像、本机缓存，
   记录每种路径的真实错误。这是任务二能否启动的前置条件。
2. **核算 B2 接入 LateDetail 后的确切规格**：在没有权重的前提下，
   用 `SegformerConfig` 构造同结构的骨架来数参数（不训练、不冒充预训练）。
   先用 b0 验证这套核算能精确复现已知的 3,753,602，再把同一算法用到 b2。

注意：本脚本**不产生任何可训练模型**。B2 正式训练必须在拿到官方权重后才进行；
计划明确禁止用随机初始化代替。
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch  # noqa: E402

from transformers import SegformerConfig, SegformerForSemanticSegmentation  # noqa: E402

import landslide_seg.models.segformer as seg  # noqa: E402
from landslide_seg.models.segformer import SegFormerSeg  # noqa: E402

B0_DIR = ROOT / "weights" / "segformer" / "segformer-b0-finetuned-ade-512-512"
B2_DIR = ROOT / "weights" / "segformer" / "segformer-b2-finetuned-ade-768-768"
KNOWN_B0_TOTAL = 3_753_602

B2_ARCH = {
    "depths": [3, 4, 6, 3],
    "hidden_sizes": [64, 128, 320, 512],
    "num_attention_heads": [1, 2, 5, 8],
    "sr_ratios": [8, 4, 2, 1],
    "mlp_ratios": [4, 4, 4, 4],
    "patch_sizes": [7, 3, 3, 3],
    "strides": [4, 2, 2, 2],
    "decoder_hidden_size": 768,
}


def probe_download() -> dict:
    """记录每种获取路径的真实结果。"""
    api = "https://huggingface.co/api/models/nvidia/segformer-b2-finetuned-ade-768-768"
    mirror = "https://hf-mirror.com/api/models/nvidia/segformer-b2-finetuned-ade-768-768"
    resolve = "https://hf-mirror.com/nvidia/segformer-b2-finetuned-ade-768-768/resolve/main/config.json"
    out = {"target": "nvidia/segformer-b2-finetuned-ade-768-768",
           "local_dir": str(B2_DIR).replace("\\", "/"),
           "local_exists": B2_DIR.is_dir(),
           "attempts": []}

    def try_url(tag: str, url: str, drop_proxy: bool = False) -> None:
        saved = {k: os.environ.pop(k, None) for k in
                 ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")} if drop_proxy else {}
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            r = urllib.request.urlopen(req, timeout=15)
            out["attempts"].append({"path": tag, "url": url, "result": "OK", "status": r.status})
        except Exception as exc:  # noqa: BLE001
            out["attempts"].append({"path": tag, "url": url,
                                    "result": "FAIL", "error": f"{type(exc).__name__}: {exc}"[:240]})
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    try_url("直接 HTTPS（走环境代理）", api)
    try_url("直接 HTTPS（绕过代理）", api, drop_proxy=True)
    try_url("HF 镜像 api", mirror, drop_proxy=True)
    try_url("HF 镜像 resolve config.json", resolve, drop_proxy=True)

    from huggingface_hub import HfApi
    try:
        info = HfApi().model_info("nvidia/segformer-b2-finetuned-ade-768-768")
        out["attempts"].append({"path": "huggingface_hub.HfApi", "result": "OK", "sha": info.sha})
    except Exception as exc:  # noqa: BLE001
        out["attempts"].append({"path": "huggingface_hub.HfApi", "result": "FAIL",
                                "error": f"{type(exc).__name__}: {exc}"[:240]})

    out["available"] = out["local_exists"] or any(a.get("result") == "OK" for a in out["attempts"])
    return out


def build_from_config(cfg: SegformerConfig) -> SegFormerSeg:
    """绕过权重加载，用同结构骨架装配 LateDetail，仅用于数参数。"""
    holder = {}

    def fake_load(_dir):
        model = SegformerForSemanticSegmentation(cfg)
        holder["model"] = model
        return model, {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}

    original = seg._load_pretrained
    seg._load_pretrained = fake_load
    try:
        return SegFormerSeg(adapter="direct10", backbone="b0" if cfg.decoder_hidden_size == 256 else "b2",
                            stem=True, refine="none", pretrained_dir=str(B0_DIR), exchange="none")
    finally:
        seg._load_pretrained = original


def main() -> None:
    dl = probe_download()
    print("[download]", json.dumps(dl, ensure_ascii=False, indent=1))

    base = SegformerConfig.from_pretrained(str(B0_DIR))
    report: dict = {"download": dl, "b0_dir": str(B0_DIR).replace("\\", "/")}

    # ---- 用 b0 校准核算方法 ----
    m0 = build_from_config(base)
    c0 = m0.count_params()
    report["calibration_b0"] = {"params": c0, "matches_known": c0["total"] == KNOWN_B0_TOTAL,
                               "known": KNOWN_B0_TOTAL}
    print(f"[calibrate] b0 total={c0['total']:,} known={KNOWN_B0_TOTAL:,} "
          f"match={c0['total'] == KNOWN_B0_TOTAL}")

    # ---- 同法核算 b2 ----
    b2 = SegformerConfig(**{**base.to_dict(), **B2_ARCH, "num_labels": 1, "num_channels": 3})
    m2 = build_from_config(b2)
    c2 = m2.count_params()
    report["b2_spec"] = {
        "decoder_hidden_size": int(b2.decoder_hidden_size),
        "hidden_sizes": list(b2.hidden_sizes),
        "depths": list(b2.depths),
        "params": c2,
        "delta_vs_b0": c2["total"] - c0["total"],
        "ratio_vs_b0": c2["total"] / c0["total"],
        "imagenet_init_used": False,
        "note": "仅数参数；未加载任何预训练权重，不构成可训练模型",
    }

    # 形状核对：B2 在 128px 下的阶段输出
    with torch.no_grad():
        x = torch.zeros(1, 10, 128, 128)
        out = m2(x)
        shapes = {k: list(v) for k, v in (m2._stage_shapes or {}).items()} if m2._stage_shapes else {}
    report["b2_shapes_at_128"] = shapes or {"logits": list(out["logits"].shape)}
    report["b2_forward_ok"] = list(out["logits"].shape) == [1, 1, 128, 128]
    print("[b2]", json.dumps(report["b2_spec"], ensure_ascii=False))

    out_path = ROOT / "reports" / "p2m1" / "b2_availability_and_spec.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
