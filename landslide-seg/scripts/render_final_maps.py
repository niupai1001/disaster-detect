"""把 DOFA / AnySat 两套系统的最终输出（滑坡概率图）与输入、真值并排画出来。

只读已有产物：概率来自 `outputs/use_next_v2/audit/p1_predictions/<route>/best/*.npz`
（P1 冻结读出的最终输出，best 权重，0 次更新重放），输入与真值来自
`data/s2_10band_v4_p20`。不训练、不改权重。

选例规则（写死在 EXAMPLES 里，可核查）：
  1) thrissur_s2_1037 —— 两条路线都命中最完整的瓦片（DOFA 0.68 / AnySat 0.79）；
  2) hiroshima_s2_1074 —— 目标面积大、AnySat 明显强于 DOFA 的瓦片；
  3) thrissur_s2_420 —— 反过来 DOFA 明显强于 AnySat 的瓦片。
不是挑最好看的，是刻意让两条路线各赢一次，避免单边选例。
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "outputs" / "use_next_v2" / "audit" / "p1_predictions"
OUT = ROOT / "outputs" / "final_maps"
ROUTES = ("dofa", "anysat")

# (sample_id, 这一行为什么选它)
EXAMPLES = (
    ("thrissur_s2_1037", "两条路线都命中：滑坡主体连成片"),
    ("hiroshima_s2_1074", "同一处滑坡，AnySat 收得住、DOFA 基本不响应"),
    ("thrissur_s2_420", "反过来：DOFA 命中，AnySat 漏掉"),
)


def load_routes() -> dict[str, dict[str, dict]]:
    """一次扫完 669×2 个 npz，按 sample_id 归档。"""
    box: dict[str, dict[str, dict]] = {}
    for f in glob.glob(str(PRED / "*" / "best" / "*.npz")):
        route = Path(f).parent.parent.name
        d = np.load(f)
        box.setdefault(str(d["sample_id"]), {})[route] = {
            "prob": d["prob"].astype(np.float32),
            "mask": d["mask"].astype(bool),
            "valid": d["valid"].astype(bool),
        }
    return box


def rgb_of(sid: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(ROOT / "data" / "s2_10band_v4_p20" / f"{sid}.npz")
    x = d["x10"].astype(np.float32) / 10000.0
    valid = d["valid"].astype(bool)
    out = np.zeros((128, 128, 3), np.float32)
    for i, b in enumerate((2, 1, 0)):  # B04 / B03 / B02
        lo, hi = np.percentile(x[b][valid], [2, 98])
        out[..., i] = np.clip((x[b] - lo) / max(hi - lo, 1e-6), 0, 1)
    out[~valid] = 0.35
    return out, valid


def iou(prob: np.ndarray, mask: np.ndarray, valid: np.ndarray, thr: float = 0.5) -> float:
    p = (prob > thr) & valid
    g = mask & valid
    u = int((p | g).sum())
    return float((p & g).sum() / u) if u else float("nan")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    OUT.mkdir(parents=True, exist_ok=True)

    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("0.58")  # 无效像元涂灰，不能留白（留白会被读成高概率）

    box = load_routes()
    fig, axes = plt.subplots(len(EXAMPLES), 4, figsize=(17.0, 5.0 * len(EXAMPLES)))
    stats = []
    for r, (sid, why) in enumerate(EXAMPLES):
        assert sid in box, f"没有 {sid} 的预测"
        rgb, valid = rgb_of(sid)
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(f"输入  {sid}\nwhy：{why}", fontsize=10.5, linespacing=1.5)

        rec, mask_ref = {}, None
        for c, route in enumerate(ROUTES, start=1):
            d = box[sid][route]
            prob, mask, v = d["prob"], d["mask"], d["valid"]
            mask_ref = mask
            rec[route] = iou(prob, mask, v)
            axes[r, c].imshow(
                np.ma.masked_where(~v, prob), cmap=cmap, vmin=0.0, vmax=1.0,
                interpolation="nearest",
            )
            axes[r, c].set_title(
                f"{route.upper()} 的最终输出：滑坡概率\nIoU@0.5 = {rec[route]:.3f}",
                fontsize=10.5, linespacing=1.5,
            )

        lab = np.zeros((128, 128, 3), np.float32)
        lab[valid] = 0.42
        lab[mask_ref & valid] = (0.95, 0.75, 0.10)
        axes[r, 3].imshow(lab)
        axes[r, 3].set_title(
            f"真值（黄色=滑坡，{int(mask_ref.sum())} 像元）\n仅作对照，不参与评分",
            fontsize=10.5, linespacing=1.5,
        )

        for c in range(4):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            for s in axes[r, c].spines.values():
                s.set_color("0.80")
        stats.append({
            "sample_id": sid,
            "label_px": int(mask_ref.sum()),
            "iou_dofa": round(rec["dofa"], 4),
            "iou_anysat": round(rec["anysat"], 4),
        })

    fig.suptitle(
        "DOFA 与 AnySat 两套系统对同一块 Sentinel-2 的最终输出（P1 已训练读出，只读重放，0 次更新）\n"
        "全部 669 块上的官方口径：事件等权 IoU  DOFA 0.107 ｜ AnySat 0.179（同预算 3000 次更新）",
        fontsize=12, linespacing=1.7,
    )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.88, bottom=0.02, wspace=0.05, hspace=0.22)
    fig.savefig(OUT / "final_maps.png", dpi=150)
    plt.close(fig)
    (OUT / "final_maps.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
