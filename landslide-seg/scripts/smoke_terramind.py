"""TerraMind V1 Small + TiM 接口验证（真实官方权重）。

验证内容：
1. 官方权重能否严格加载；
2. 完整 12 波段输入能否前向；
3. 特征形状是否符合预期（token 网格 = H/16）；
4. **TiM ON / OFF 在同一输入上是否真的产生不同输出**（证明 TiM 真被执行）。

注意：输入是**合成 12 通道张量**（见 pipeline.make_synthetic_batch），
只能证明接口与权重可用，不能证明 TiM 对真实滑坡有效。

用法:
    python scripts/smoke_terramind.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from landslide_seg.data.pipeline import make_synthetic_batch  # noqa: E402
from landslide_seg.encoders.terramind_tim import (  # noqa: E402
    TerraMindTiMAdapter,
    normalize_s2l2a_dn,
)


def main() -> None:
    out_dir = ROOT / "outputs" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}")

    adapter = TerraMindTiMAdapter(tim_enabled=True, freeze=True).to(device)
    print("[model]", json.dumps(adapter.describe(), ensure_ascii=False, indent=2))

    batch = make_synthetic_batch(batch_size=1, size=128, seed=0)
    x12 = normalize_s2l2a_dn(batch.x12.to(device))

    res_on = adapter(x12, tim=True)
    print(f"[TiM ON ] tokens={[tuple(t.shape) for t in res_on['tokens']]} "
          f"grid={res_on['grid']} latency={res_on['info'].tim_latency_s:.2f}s")

    res_off = adapter(x12, tim=False)
    print(f"[TiM OFF] tokens={[tuple(t.shape) for t in res_off['tokens']]} "
          f"latency={res_off['info'].tim_latency_s:.2f}s")

    a = res_on["tokens"][-1]
    b = res_off["tokens"][-1]
    diff = (a - b).abs().mean().item()
    rel = diff / (b.abs().mean().item() + 1e-9)
    print(f"[TiM check] mean|Δ|={diff:.6f}  relative={rel:.4%}")

    feat = adapter.tokens_to_map(a)
    print(f"[map] {tuple(feat.shape)} stride_m={128 * 10 / res_on['grid'][0]:.1f}")

    report = {
        "device": device,
        "describe": adapter.describe(),
        "input": {"shape": list(x12.shape), "synthetic": True,
                  "bands": "S2L2A 12 (B01..B12 incl. B8A)"},
        "tim_on": {"shapes": [list(t.shape) for t in res_on["tokens"]],
                   "latency_s": res_on["info"].tim_latency_s},
        "tim_off": {"shapes": [list(t.shape) for t in res_off["tokens"]],
                    "latency_s": res_off["info"].tim_latency_s},
        "tim_output_delta_mean_abs": diff,
        "tim_output_delta_relative": rel,
        "tim_effectively_executed": bool(rel > 1e-6),
    }
    path = out_dir / "smoke_terramind.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {path}")
    if not report["tim_effectively_executed"]:
        print("[WARN] TiM ON/OFF 输出一致，TiM 可能没有真正执行！")


if __name__ == "__main__":
    main()
