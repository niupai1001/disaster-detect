"""三架构固定案例图（v4r4 §7）。

从已保存的概率图直接渲染，不重复推理；三个架构共用同一批瓦片，
因此同一行的差异只来自模型。

选例规则：沿用v4r1 detailres已经固定的8例，不按当前模型输出或图块GT总面积重选。
误差图的TP/FP/FN均限于有效区；图像可见性和标签时相未人工确认。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from landslide_seg.config import resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import read_manifest  # noqa: E402

PRED = {
    "LateDetail": "outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_detailres_n100_u3600_s1234/1234/eval/predictions_best_val.npz",
    "Self": "outputs/segformer/s2_10band_v4r4/segformer_b0_detail_early_self_residual_n100_u3600_s1234/1234/eval/predictions_best_val.npz",
    "Cross": "outputs/segformer/s2_10band_v4r4/segformer_b0_detail_early_cross_exchange_n100_u3600_s1234/1234/eval/predictions_best_val.npz",
}
OUT = ROOT / "reports" / "v4r4_results"


def rgb(x10: np.ndarray) -> np.ndarray:
    img = np.stack([x10[2], x10[1], x10[0]], axis=-1).astype(np.float32)
    lo, hi = np.percentile(img, [2, 98], axis=(0, 1))
    return np.clip((img - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="reports/v4r4_results")
    args = parser.parse_args()
    output = ROOT / args.out_dir
    arms = {}
    for k, v in PRED.items():
        z = np.load(resolve_path(v), allow_pickle=True)
        arms[k] = {"prob": z["probability"], "ids": [str(s) for s in z["sample_ids"].tolist()]}
    ids = arms["LateDetail"]["ids"]
    for k, v in arms.items():
        assert v["ids"] == ids, k

    gt = np.load(resolve_path(PRED["LateDetail"]), allow_pickle=True)["gt"].astype(bool)
    valid = np.load(resolve_path(PRED["LateDetail"]), allow_pickle=True)["valid"].astype(bool)

    # Retain the predeclared v4r1 detail-model cases rather than selecting
    # different tiles from the current results or total foreground area.
    gt_px = gt.reshape(len(ids), -1).sum(1)
    selection_path = ROOT / "reports/v4r1_results/error_atlas/selection.json"
    selected = [c for c in json.loads(selection_path.read_text(encoding="utf-8"))
                if c["experiment"] == "segformer_b0_direct10_detailres_n100_u3600_s1234" and c["available"]]
    indices = {sid: i for i, sid in enumerate(ids)}
    picks = [indices[c["sample_id"]] for c in selected]
    assert len(picks) == 8 and len(set(picks)) == 8

    records = {r["sample_id"]: r for r in read_manifest(resolve_path("manifests/v4r1/pool_100.jsonl"))}

    n = len(picks)
    fig, axes = plt.subplots(n, 5, figsize=(15, 2.9 * n), squeeze=False)
    for row, i in enumerate(picks):
        rec = records[ids[i]]
        x10 = np.load(resolve_path(rec["cache"]))["x10"].astype(np.float32)
        panels = [(rgb(x10), f"{ids[i]} ({rec['event_id']})", None), (gt[i], "GT", "gray")]
        for k in ("LateDetail", "Self", "Cross"):
            p = arms[k]["prob"][i]
            err = np.zeros((*gt[i].shape, 3), dtype=np.float32)
            pr = (p >= 0.5) & valid[i]
            err[..., 1] = (gt[i] & pr).astype(np.float32)
            err[..., 0] = ((~gt[i]) & pr).astype(np.float32)
            err[..., 2] = (gt[i] & (~pr) & valid[i]).astype(np.float32)
            panels.append((err, f"{k}", None))
        for col, (img, title, cmap) in enumerate(panels):
            ax = axes[row][col]
            if cmap:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            else:
                ax.imshow(np.clip(img, 0, 1))
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        axes[row][0].set_ylabel(f"gt={int(gt_px[i])}px", fontsize=8)
    fig.suptitle("v4r4 fixed cases: GT / LateDetail / Self / Cross  (green=TP red=FP blue=FN)", fontsize=11, y=.995)
    fig.tight_layout(rect=(0, 0, 1, .975))
    output.mkdir(parents=True, exist_ok=True)
    png = output / "cases_best_val.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)
    (output / "cases_best_val.json").write_text(
        json.dumps({"selection_source": str(selection_path), "visibility_reviewed": False,
                    "picks": [{"sample_id": ids[i], "event_id": records[ids[i]]["event_id"],
                               "gt_px": int(gt_px[i])} for i in picks]}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    print(f"[saved] {png}")
    for i in picks:
        print(f"  {ids[i]:28s} {records[ids[i]]['event_id']:14s} gt_px={int(gt_px[i])}")


if __name__ == "__main__":
    main()
