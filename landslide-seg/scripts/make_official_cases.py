"""BENCH_OFFICIAL 单期主实验的固定错误案例（TEST 侧）。

选例规则（确定性，不参与选模）：
- 只看官方 TEST 的瓦片；
- 用「GT 对象中与预测完全无重叠的个数」作为难度指标（向量化，避免 O(n²) 匹配）；
- 按 (零重叠对象数降序, 其中 ≤19px 的个数降序, sample_id) 取前 8 例，并覆盖多个事件。

列：RGB | GT | 预测误差（绿=TP 红=FP 蓝=FN）。
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

from landslide_seg.config import resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import read_manifest  # noqa: E402
from landslide_seg.training.segformer_metrics import label_objects  # noqa: E402

PRED = "outputs/segformer/official_single/SegFormer-LateDetail-OfficialSplit-SingleDate/1234/eval/predictions_best_test.npz"
OUT = ROOT / "reports" / "official_single" / "cases_test.png"
N_CASES = 8
THR = 0.5


def rgb(x10: np.ndarray) -> np.ndarray:
    img = np.stack([x10[2], x10[1], x10[0]], axis=-1).astype(np.float32)
    lo, hi = np.percentile(img, [2, 98], axis=(0, 1))
    return np.clip((img - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


def main() -> None:
    z = np.load(resolve_path(PRED), allow_pickle=True)
    ids = [str(s) for s in z["sample_ids"].tolist()]
    prob, gt, valid = z["probability"], z["gt"].astype(bool), z["valid"].astype(bool)
    records = {r["sample_id"]: r for r in read_manifest(resolve_path("manifests/official_single/test.jsonl"))}

    stats = []
    for i, sid in enumerate(ids):
        pred = (prob[i] >= THR) & valid[i]
        if not gt[i].any():
            continue
        lab, n = label_objects(gt[i] & valid[i], 2)
        if not n:
            continue
        areas = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        hit = np.bincount(lab[pred].ravel(), minlength=n + 1)[1:]
        missed = [j for j in range(n) if hit[j] == 0]
        tiny = sum(1 for j in missed if areas[j] <= 19)
        stats.append({"i": i, "sid": sid, "event": records[sid]["event_id"],
                      "n_obj": int(n), "missed": len(missed), "tiny_missed": tiny,
                      "gt_px": int(gt[i].sum())})
    stats.sort(key=lambda r: (-r["missed"], -r["tiny_missed"], r["sid"]))

    picks, by_ev = [], {}
    for r in stats:
        if len(picks) >= N_CASES:
            break
        if by_ev.get(r["event"], 0) >= max(2, N_CASES // 4):
            continue
        picks.append(r)
        by_ev[r["event"]] = by_ev.get(r["event"], 0) + 1
    for r in stats:
        if len(picks) >= N_CASES:
            break
        if r not in picks:
            picks.append(r)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(picks)
    fig, axes = plt.subplots(n, 3, figsize=(9, 2.9 * n), squeeze=False)
    for row, r in enumerate(picks):
        i = r["i"]
        x10 = np.load(resolve_path(records[r["sid"]]["cache"]))["x10"].astype(np.float32)
        pred = (prob[i] >= THR) & valid[i]
        err = np.zeros((*gt[i].shape, 3), dtype=np.float32)
        err[..., 1] = (gt[i] & pred).astype(np.float32)
        err[..., 0] = ((~gt[i]) & pred).astype(np.float32)
        err[..., 2] = (gt[i] & (~pred)).astype(np.float32)
        for c, (img, title, cmap) in enumerate([(rgb(x10), f"{r['sid']} ({r['event']})", None),
                                                (gt[i], "GT", "gray"), (err, "best prediction", None)]):
            ax = axes[row][c]
            if cmap:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            else:
                ax.imshow(np.clip(img, 0, 1))
            if row == 0:
                ax.set_title(title, fontsize=9)
            ax.axis("off")
        axes[row][0].set_ylabel(f"miss {r['missed']}/{r['n_obj']}", fontsize=7)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110)
    plt.close(fig)
    (OUT.with_suffix(".json")).write_text(json.dumps({"picks": picks}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {OUT}")
    for r in picks:
        print(f"  {r['sid']:26s} {r['event']:14s} missed {r['missed']}/{r['n_obj']} (tiny {r['tiny_missed']})")


if __name__ == "__main__":
    main()
