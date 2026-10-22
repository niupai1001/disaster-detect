"""验收门槛 #2：旧 checkpoint 在固定样本上的回归输出保持一致（FP32 最大绝对误差 ≤1e-5）。

做法：取 v4r4 Self 组当时代码产出的 ``predictions_best_val.npz``（固定样本、固定顺序的概率图），
用**当前代码**重建同一 checkpoint、同一份统计、同一批样本，逐像素比较 sigmoid 概率。
若能复现，说明这段时间对数据集/损失/工厂的改动没有改变原有前向行为。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402
import torch  # noqa: E402

from landslide_seg.config import load_segformer_config, resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import S2SegDataset, collate, load_stats  # noqa: E402
from landslide_seg.training.segformer_trainer import load_checkpoint  # noqa: E402

RUN = "outputs/segformer/s2_10band_v4r4/segformer_b0_detail_early_self_residual_n100_u3600_s1234/1234"
CFG = "configs/segformer/v4r4/segformer_b0_detail_early_self_residual_n100_u3600_s1234.yaml"
NPZ = f"{RUN}/eval/predictions_best_val.npz"
THRESH = 1e-5


def main() -> None:
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    z = np.load(resolve_path(NPZ), allow_pickle=True)
    ids = [str(s) for s in z["sample_ids"].tolist()]
    ref = z["probability"].astype(np.float32)
    print(f"参考预测：{len(ids)} 块，{ref.shape[1]}x{ref.shape[2]}")

    cfg = load_segformer_config(CFG)
    s = cfg["segformer"]
    stats = load_stats(resolve_path(s["data"]["stats"]))
    recs = {r["sample_id"]: r for r in
            (json.loads(l) for l in resolve_path(s["data"]["manifest"]).read_text(encoding="utf-8").splitlines() if l.strip())}
    missing = [i for i in ids if i not in recs]
    if missing:
        raise SystemExit(f"清单缺少 {len(missing)} 个样本")
    ordered = [recs[i] for i in ids]

    ds = S2SegDataset(ordered, stats, augment=False, preload=True,
                      extra=str(s["model"].get("extra_channel", "none")))
    model, payload = load_checkpoint(resolve_path(f"{RUN}/best.pt"), device=str(dev))
    model.eval()

    got = []
    with torch.no_grad():
        for k in range(0, len(ds), 32):
            batch = collate([ds[i] for i in range(k, min(k + 32, len(ds)))])
            logits = model(batch["image"].to(dev))["logits"].float()
            got.append(torch.sigmoid(logits[:, 0]).cpu().numpy())
    got = np.concatenate(got, axis=0)

    diff = np.abs(got - ref)
    mx = float(diff.max())
    mean = float(diff.mean())
    # 阈值处的标签一致性（更贴近实际影响）
    thr = float(s["eval"]["threshold"])
    a, b = got >= thr, ref >= thr
    agree = float((a == b).mean())
    rep = {"checkpoint": f"{RUN}/best.pt", "n": len(ids), "max_abs_diff": mx, "mean_abs_diff": mean,
           "threshold": thr, "pixel_agreement": agree, "gate": THRESH,
           "pass": mx <= THRESH, "selection_metric": payload.get("extra", {}).get("selection_metric")}
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    out = ROOT / "manifests" / "event_cv_v5" / "regression_check.json"
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    if not rep["pass"]:
        print(f"[FAIL] 最大绝对误差 {mx:.3e} > {THRESH:.0e}")
        sys.exit(1)
    print(f"[PASS] 最大绝对误差 {mx:.3e} ≤ {THRESH:.0e}；阈值处像素一致率 {agree:.6f}")


if __name__ == "__main__":
    main()
