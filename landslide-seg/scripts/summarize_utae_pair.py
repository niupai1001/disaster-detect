"""U-TAE 时序配对汇总：单期 vs 15 期。

按计划给三样东西：
- best / last × VAL / TEST × valid / 全像素 两套口径；
- 逐事件 IoU/AP 与六档尺度召回；
- 与单期组的差值（这是"时间观测是否有增量"的直接读数）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BINS = ["1_4", "5_9", "10_19", "20_49", "50_199", "ge200"]
OUT = ROOT / "reports" / "event_cv_v5" / "utae_pair_summary.md"


def load(mode: str) -> dict:
    p = ROOT / "outputs" / "utae_pair" / f"utae_{mode}" / "1234" / "eval_summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def f4(v, n=4):
    return "-" if v is None else f"{v:.{n}f}"


def main() -> None:
    data = {m: load(m) for m in ("single", "multi")}
    L = ["# U-TAE 时序配对：单期 vs 15 期 + DEM\n",
         "同一 3,012/758/945 子集、同一 valid、同一 normalizer、同一初始状态；"
         "官方 U-TAE + AdamW 1e-3 / wd 0.01 / pos_weight 5 / BCE+Dice 各 0.5 / 75 epoch / batch 16。\n",
         "## 1. 总体（两套像素口径）\n",
         "| 组 | 检查点 | split | valid IoU | 全像素 IoU | AP | 事件均值 |",
         "|---|---|---|---:|---:|---:|---:|"]
    for mode, label in (("single", "单期+DEM"), ("multi", "15期+DEM")):
        for ck in ("best", "last"):
            for split in ("val", "test"):
                r = (data[mode].get(ck) or {}).get(split)
                if not r:
                    continue
                L.append(f"| {label} | {ck} | {split} | {f4(r['valid_scope']['iou'])} | "
                         f"{f4(r['all_pixels']['iou'])} | {f4(r['valid_scope'].get('ap'))} | "
                         f"{f4(r['event_mean_iou'])} |")

    L += ["\n## 2. 差值（15 期 − 单期）\n",
          "| 检查点 | split | Δvalid IoU | Δ全像素 IoU | ΔAP | Δ事件均值 |",
          "|---|---|---:|---:|---:|---:|"]
    for ck in ("best", "last"):
        for split in ("val", "test"):
            a = (data["single"].get(ck) or {}).get(split)
            b = (data["multi"].get(ck) or {}).get(split)
            if not a or not b:
                continue
            L.append(f"| {ck} | {split} | {b['valid_scope']['iou']-a['valid_scope']['iou']:+.4f} | "
                     f"{b['all_pixels']['iou']-a['all_pixels']['iou']:+.4f} | "
                     f"{(b['valid_scope'].get('ap') or 0)-(a['valid_scope'].get('ap') or 0):+.4f} | "
                     f"{(b['event_mean_iou'] or 0)-(a['event_mean_iou'] or 0):+.4f} |")

    L += ["\n## 3. 逐事件 IoU（TEST，best）\n",
          "| 事件 | 单期 | 15 期 | 差值 |", "|---|---:|---:|---:|"]
    sa = ((data["single"].get("best") or {}).get("test") or {}).get("by_event") or {}
    sb = ((data["multi"].get("best") or {}).get("test") or {}).get("by_event") or {}
    for ev in sorted(set(sa) | set(sb)):
        x = (sa.get(ev) or {}).get("iou")
        y = (sb.get(ev) or {}).get("iou")
        d = f"{y-x:+.4f}" if (x is not None and y is not None) else "-"
        L.append(f"| {ev} | {f4(x)} | {f4(y)} | {d} |")

    L += ["\n## 4. 尺度召回（TEST，best，对象级）\n",
          "| 组 | " + " | ".join(f"{b} px" for b in BINS) + " |",
          "|---|" + "---:|" * len(BINS)]
    for mode, label in (("single", "单期+DEM"), ("multi", "15期+DEM")):
        r = ((data[mode].get("best") or {}).get("test") or {}).get("scale_recall") or {}
        if not r:
            continue
        L.append(f"| {label} | " + " | ".join(f4(r.get(b), 3) for b in BINS) + " |")

    L += ["\n## 5. 该对照能说明与不能说明的\n",
          "**能说明**：在完全相同的架构、数据子集、评价范围、初始化与优化设置下，"
          "把输入从单期扩展到 15 期（含灾前历史帧）带来的增量。\n",
          "**不能说明**：\n",
          "1. 不是「同一预警时刻」的收益——15 期里包含**晚于单期帧**的观测，属回顾性增量；",
          "2. 不是纯「时间信息」的贡献——两组都带 DEM，且 15 期天然包含更多云-free 机会；",
          "3. 未复现官方 998 块 TEST 与三 seed 均值（单 seed、可用子集 945 块）。\n",
          "## 6. 若两组都未完成\n",
          "下面的表为空即表示该组尚未跑完或未评测；不得用中途读数代替。"]
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
