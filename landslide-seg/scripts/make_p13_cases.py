"""P1-3 固定错误案例图。

选例规则（确定性、只用 GT 与两组预测，不参与选模）：
- 只在 DEV 的 669 块里取，且该块至少被两组之一漏掉一个可评估对象；
- 按「先难后易」排序：优先 1–4 px 对象被漏的块，其次 5–19 px，再次 20–49 px；
- 同一档内按 (漏检数降序, sample_id) 取，覆盖两个事件。

列：RGB | GT | FewEvents 误差 | AllEvents 误差（绿=TP 红=FP 蓝=FN）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
from landslide_seg.training.segformer_metrics import label_objects, match_objects, object_table  # noqa: E402

BINS = [("1_4", 1, 4), ("5_9", 5, 9), ("10_19", 10, 19), ("20_49", 20, 49)]
THR, IOU_THR = 0.5, 0.25


def bin_of(a: int) -> str:
    for n, lo, hi in BINS:
        if lo <= a <= hi:
            return n
    return "big"


def rgb(x10: np.ndarray) -> np.ndarray:
    img = np.stack([x10[2], x10[1], x10[0]], axis=-1).astype(np.float32)
    lo, hi = np.percentile(img, [2, 98], axis=(0, 1))
    return np.clip((img - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--few", required=True)
    ap.add_argument("--all", dest="all_", required=True)
    ap.add_argument("--n-cases", type=int, default=8)
    ap.add_argument("--out", default="reports/p13/cases_dev.png")
    args = ap.parse_args()

    z = {k: np.load(resolve_path(p), allow_pickle=True) for k, p in (("few", args.few), ("all_", args.all_))}
    ids = [str(s) for s in z["few"]["sample_ids"].tolist()]
    assert ids == [str(s) for s in z["all_"]["sample_ids"].tolist()]
    gt = z["few"]["gt"].astype(bool)
    valid = z["few"]["valid"].astype(bool)

    records = {r["sample_id"]: r for r in read_manifest(resolve_path("manifests/v4r1/pool_100.jsonl"))}

    rows = []
    for i, sid in enumerate(ids):
        gl, ng = label_objects(gt[i], 2)
        if not ng:
            continue
        gobjs = object_table(gl, ng, valid[i])
        per = {}
        for k in ("few", "all_"):
            pred = (z[k]["probability"][i] >= THR) & valid[i]
            pl, npd = label_objects(pred, 2)
            pobjs = object_table(pl, npd, valid[i])
            per[k] = match_objects(gl, gobjs, pl, pobjs, IOU_THR)
        miss = {"few": [], "all_": []}
        for o in gobjs:
            if o["truncated"]:
                continue
            for k in ("few", "all_"):
                if o["id"] not in per[k]:
                    miss[k].append(o["area"])
        if not miss["few"] and not miss["all_"]:
            continue
        tiny = lambda lst: sum(1 for a in lst if a <= 4)
        small = lambda lst: sum(1 for a in lst if 5 <= a <= 19)
        mid = lambda lst: sum(1 for a in lst if 20 <= a <= 49)
        allm = miss["few"] + miss["all_"]
        rows.append({"i": i, "sid": sid, "event": records[sid]["event_id"],
                     "n_miss_few": len(miss["few"]), "n_miss_all": len(miss["all_"]),
                     "tiny": tiny(allm), "small": small(allm), "mid": mid(allm),
                     "gt_px": int(gt[i].sum())})

    def key(r):
        if r["tiny"] > 0:
            tier = 0
        elif r["small"] > 0:
            tier = 1
        elif r["mid"] > 0:
            tier = 2
        else:
            tier = 3
        return (tier, -(r["n_miss_few"] + r["n_miss_all"]), r["sid"])

    rows.sort(key=key)
    # 覆盖两个事件：先按事件轮流取
    picks, by_ev = [], Counter()
    for r in rows:
        if len(picks) >= args.n_cases:
            break
        if by_ev[r["event"]] >= args.n_cases // 2 and by_ev[r["event"]] > 0:
            continue
        picks.append(r)
        by_ev[r["event"]] += 1
    for r in rows:
        if len(picks) >= args.n_cases:
            break
        if r not in picks:
            picks.append(r)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(picks)
    fig, axes = plt.subplots(n, 4, figsize=(12, 2.8 * n), squeeze=False)
    for row, r in enumerate(picks):
        i = r["i"]
        x10 = np.load(resolve_path(records[r["sid"]]["cache"]))["x10"].astype(np.float32)
        panels = [(rgb(x10), f"{r['sid']} ({r['event']})"), (gt[i], "GT")]
        for k, name in (("few", "FewEvents"), ("all_", "AllEvents")):
            pr = (z[k]["probability"][i] >= THR) & valid[i]
            err = np.zeros((*gt[i].shape, 3), dtype=np.float32)
            err[..., 1] = (gt[i] & pr).astype(np.float32)
            err[..., 0] = ((~gt[i]) & pr).astype(np.float32)
            err[..., 2] = (gt[i] & (~pr)).astype(np.float32)
            panels.append((err, name))
        for c, (img, title) in enumerate(panels):
            ax = axes[row][c]
            if c == 1:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(np.clip(img, 0, 1))
            if row == 0:
                ax.set_title(title, fontsize=10)
            ax.axis("off")
        axes[row][0].set_ylabel(f"miss\nfew={r['n_miss_few']}\nall={r['n_miss_all']}", fontsize=7)
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    (out.with_suffix(".json")).write_text(json.dumps({"picks": picks}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {out}")
    for r in picks:
        print(f"  {r['sid']:26s} {r['event']:14s} miss few={r['n_miss_few']:2d} all={r['n_miss_all']:2d} "
              f"tiny={r['tiny']} 5-19={r['small']} 20-49={r['mid']}")


if __name__ == "__main__":
    main()
