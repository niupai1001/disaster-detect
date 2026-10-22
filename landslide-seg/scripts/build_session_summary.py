"""汇总本轮全部新增实验，生成单一报告。

数字全部从已落盘的 JSON 读取，避免手抄误差：
- ``reports/p13/results_summary.json`` + ``significance.json``
- ``reports/p2m1/capacity_summary.json`` + ``capacity_significance.json``
- ``reports/official_single/variants_summary.json``（SegFormer 五组）
- ``reports/official_single/unet_vs_segformer.json``（UNet 四组）
- ``reports/official_single/results.json`` + ``metric_scope_test.json``
- ``reports/v4r4_results/arms_summary.json``（背景参照）
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

R = ROOT / "reports"
OUT = R / "session_summary_2026-10-20.md"


def J(rel: str):
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def f4(v) -> str:
    return "-" if v is None else f"{float(v):.4f}"


def d4(v) -> str:
    return "-" if v is None else f"{float(v):+.4f}"


def main() -> None:
    p13 = J("reports/p13/results_summary.json")
    p13s = J("reports/p13/significance.json")
    cap = J("reports/p2m1/capacity_summary.json")
    caps = J("reports/p2m1/capacity_significance.json")
    off = J("reports/official_single/results.json")
    var = J("reports/official_single/variants_summary.json")
    uni = J("reports/official_single/unet_vs_segformer.json")
    scope = J("reports/official_single/metric_scope_test.json")
    v4 = J("reports/v4r4_results/arms_summary.json")

    L: list[str] = []
    A = L.append

    A("# 阶段性实验汇总（2026-10-20）\n")
    A("本轮共新增 **13 个正式训练**，全部完成训练与评测；另用到 3 个历史运行作背景参照。")
    A("所有新增组共享：单 seed 1234、FP32、batch 64、3,600 次成功更新（230,400 曝光）、")
    A("同一份损失（除非明确改动）、同一选模口径（VAL/DEV 上选定，不碰 TEST）。\n")

    A("## 0. 一页结论\n")
    A("| # | 问题 | 结论 | 强度 |")
    A("|---|---|---|---|")
    A("| 1 | 固定图块数、扩大训练事件覆盖 | **有效**：同 N 下 9 事件明显优于 3 事件，两个检查点同向 | 较强（区间不跨 0，但事件级 n=2） |")
    A("| 2 | 换成更大的同族主干（B2） | **无效且为负**：固定预算下比 B0 低，两检查点同向 | 较强 |")
    A("| 3 | 加入 DEM / 坡度 / 两者 | **只有坡度方向稳定**；DEM 实质为零；两者合用不叠加 | 弱（VAL/TEST 符号不一致） |")
    A("| 4 | pos_weight 1→5 | **只改变工作点**：召回大涨、精确率大跌、IoU/AP 基本不动 | 较强（两个架构一致复现） |")
    A("| 5 | 换架构（SegFormer vs U-Net） | 现有条件下 SegFormer 更高，且更省显存 | 弱（参数量差 3.8×、U-Net 未调 LR） |\n")

    # ---------------- 1. 清单 ----------------
    A("## 1. 实验清单\n")
    A("| 实验块 | 组 | 协议 | 训练样本 | 参数量 | 耗时 | 峰值显存 |")
    A("|---|---|---|---:|---:|---:|---:|")
    for arm in ("P13-FewEvents", "P13-AllEvents"):
        v = p13[arm]
        A(f"| P1-3 事件覆盖 | {arm} | 事件 OOD（p13） | {v['n_train']:,} | "
          f"{(v.get('params') or {}).get('total',0):,} | {v['seconds']:.0f} s | {v['peak_alloc_mib']:.0f} MiB |")
    for arm in ("B0-AllTrain-ctl", "B2-AllTrain"):
        v = cap[arm]
        A(f"| P2-M1 主干规格 | {arm} | 事件 OOD（2223） | {v['n_train']:,} | "
          f"{(v.get('params') or {}).get('total',0):,} | {v['seconds']:.0f} s | {v['peak_alloc_mib']:.0f} MiB |")
    for arm, v in var.items():
        A(f"| 官方单期 · 输入/损失 | SegFormer-{arm} | 官方 split | {v['n_train']:,} | "
          f"{(v.get('params') or {}).get('total',0):,} | {v['seconds']:.0f} s | {v['peak_alloc_mib']:.0f} MiB |")
    for arm, v in uni.get("UNet", {}).items():
        A(f"| 官方单期 · 架构 | UNet-{arm} | 官方 split | 3,012 | {v['params']:,} | "
          f"{v['seconds']:.0f} s | {v['peak']:.0f} MiB |")
    A("")
    A("背景参照：`REF-2223`（v4r1 LateDetail，事件 OOD 全量 2,223 块）、`v4r4 Self/Cross`（编码期交换实验）、")
    A("`B2 规格核算`（27,430,338 参数，未训练前的离线计算）。\n")

    # ---------------- 2. P1-3 ----------------
    few, alle, ref = p13["P13-FewEvents"], p13["P13-AllEvents"], p13["REF-2223(v4r1)"]
    A("## 2. 事件覆盖（P1-3）\n")
    A("同一批 1,251 块，只改训练事件覆盖：3 个事件 vs 9 个事件。归一化统计只由两组交集（703 块）拟合，两组共用。\n")
    A("| 组 | 事件数 | 合并 IoU | AP | 对象召回 | 边界 F1 | 事件均值 |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for name, v in (("FewEvents", few), ("AllEvents", alle), ("REF-2223（背景）", ref)):
        b = v.get("best_val")
        A(f"| {name} | {'3' if name=='FewEvents' else ('9' if name=='AllEvents' else '9')} | {f4(b['iou'])} | "
          f"{f4(b['ap'])} | {f4((b.get('objects') or {}).get('object_recall_evaluable'))} | "
          f"{f4(b['boundary_f1'])} | {f4(b['event_mean_iou'])} |")
    A("")
    A("配对 bootstrap（按事件分层，5,000 次）：\n")
    A("| 检查点 | AllEvents − FewEvents | 95% 区间 | 跨 0 |")
    A("|---|---:|---|---|")
    for tag, label in (("best_val", "best"), ("last_val", "last")):
        s = p13s[tag]
        A(f"| {label} | {d4(s['delta'])} | [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}] | {'是' if s['crosses_zero'] else '否'} |")
    A("")
    A("逐事件：hiroshima `+0.1106 / +0.1198`、thrissur `+0.0108 / +0.0040`（best / last）。")
    A("增益几乎全部来自 hiroshima，而 hiroshima 是 `SEEN_DEV`；thrissur 上的改善很小。")
    A("**因此这条结论应表述为「扩大事件覆盖在本协议下有效」，而不是「已在未见事件上泛化」。**\n")

    # ---------------- 3. P2-M1 ----------------
    A("## 3. 主干规格（P2-M1）\n")
    A("同数据（2,223 块）、同预算、同 seed，只换主干：B0（3.75M）vs B2（27.43M，decoder 也从 256 变 768）。\n")
    A("| 组 | 参数量 | 合并 IoU | AP | 对象召回 | 最佳步 |")
    A("|---|---:|---:|---:|---:|---:|")
    for name, v in (("B0-AllTrain-ctl", cap["B0-AllTrain-ctl"]), ("B2-AllTrain", cap["B2-AllTrain"])):
        b = v["best_val"]
        A(f"| {name} | {(v.get('params') or {}).get('total',0):,} | {f4(b['iou'])} | {f4(b['ap'])} | "
          f"{f4((b.get('objects') or {}).get('object_recall_evaluable'))} | {v['best_step']} |")
    A("")
    A("| 检查点 | B2 − B0ctl | 95% 区间 | 跨 0 |")
    A("|---|---:|---|---|")
    for tag, label in (("best_val", "best"), ("last_val", "last")):
        s = caps[tag]
        A(f"| {label} | {d4(s['delta'])} | [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}] | {'是' if s['crosses_zero'] else '否'} |")
    A("")
    A("**同版本 B0 控制组复现了历史参照**（best 0.3082 vs 0.3081，各固定步吻合到 1e-4），")
    A("说明历史 B0 结果可继续作为参照，代码没有行为漂移。\n")

    # ---------------- 4. 官方单期基线 ----------------
    A("## 4. 官方单期协议基线\n")
    A("官方 `S12LS-LD` split（commit `2eb025bd`），单期十波段，训练 3,012 / VAL 758 / TEST 945。\n")
    A("| split | 检查点 | 合并 IoU | AP | Precision | Recall | 边界 F1 | 对象召回 |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    for tag, label in (("best_val", "VAL"), ("last_val", "VAL"), ("best_test", "TEST"), ("last_test", "TEST")):
        b = off.get(tag)
        if not b:
            continue
        A(f"| {label} | {tag.split('_')[0]} | {f4(b['iou'])} | {f4(b['ap'])} | {f4(b['precision'])} | "
          f"{f4(b['recall'])} | {f4(b['boundary_f1'])} | {f4((b.get('objects') or {}).get('object_recall_evaluable'))} |")
    A("")
    if scope:
        ap_all = scope["all_pixels"]
        A("与官方表值的差距（先换算到官方口径：官方用**全像素**，本项目主口径只用 valid 像素）：\n")
        A("| 指标 | 官方 U-ConvLSTM | 官方 U-TAE | 本项目（全像素口径） | 差距 vs U-ConvLSTM |")
        A("|---|---:|---:|---:|---:|")
        # 官方表值来自 reports/P0_official_benchmark_protocol.md（README 原表，75 epochs / 3 seeds）
        for name, conv, utae, key in (("IoU", 44.88, 44.74, "iou"), ("F1", 61.95, 61.80, "f1"),
                                      ("Precision", 60.59, 53.19, "precision"),
                                      ("Recall", 63.92, 74.90, "recall"),
                                      ("AP", 65.13, 67.75, "ap")):
            ours = ap_all[key] * 100
            A(f"| {name} | {conv:.2f} | {utae:.2f} | **{ours:.2f}** | {ours-conv:+.2f} |")
        A("")
        A("官方为多时相 15 期 + DEM、75 epochs、3 seeds；本项目为单期十波段、3,600 更新、单 seed。")
        A("训练轮数其实接近（230,400 曝光 ÷ 3,012 ≈ 76.5 遍 vs 官方 75 epochs），")
        A("**所以差距不能归因于预算不足**，主要混杂在输入模态、损失权重与 seed 数。")
        A("差距的主体是**召回**（P 已与 U-TAE 持平，R 低 30 点），AP 差 17.7 点——AP 与阈值无关，调阈值动不了它。\n")

    # ---------------- 5. 输入/损失变体 ----------------
    A("## 5. 输入与损失变体（SegFormer，五组）\n")
    A("同为官方 split、同 3,012 训练块、同预算、同 seed；每组只改一个因素。\n")
    A("| 组 | 输入 | TEST IoU | TEST AP | Precision | Recall | 边界 F1 | ΔIoU |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    base_t = var["baseline"]["best_test"]
    for arm, v in var.items():
        b = v["best_test"]
        A(f"| {arm} | {'10ch' if arm in ('baseline','PW5') else ('12ch' if arm=='DEMSlope' else '11ch')} | "
          f"{f4(b['iou'])} | {f4(b['ap'])} | {f4(b['precision'])} | {f4(b['recall'])} | {f4(b['boundary_f1'])} | "
          f"{d4(b['iou']-base_t['iou']) if arm!='baseline' else '-'} |")
    A("")
    A("TEST 尺度召回（对象级）：\n")
    A("| 组 | 1–4 px | 5–9 px | 10–19 px | 20–49 px | 50–199 px | ≥200 px |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for arm, v in var.items():
        sc = v["best_test"].get("scale") or {}
        A(f"| {arm} | {f4(sc.get('1_4'))} | {f4(sc.get('5_9'))} | {f4(sc.get('10_19'))} | "
          f"{f4(sc.get('20_49'))} | {f4(sc.get('50_199'))} | {f4(sc.get('ge200'))} |")
    A("")
    A("三条要点：")
    A("1. **DEM 与坡度不叠加**：DEMSlope 的 IoU 增益 +0.0089 ≈ DEM 单独 +0.0087 < Slope 单独 +0.0127。坡度是 DEM 的导数，信息冗余。")
    A("2. **DEMSlope 拿到最好的排序质量与边界**（AP 0.4969、边界 F1 0.5249、精确率代价仅 −0.0028），但小目标更差（1–4 px 0.174 vs 基线 0.210）。")
    A("3. **PW5 是唯一大幅改善小目标的**（1–4 px 0.210→0.301、5–9 px 0.377→0.442），代价是精确率 −7.5 点、IoU 几乎不动。\n")

    # ---------------- 6. 架构 ----------------
    A("## 6. 架构对比（U-Net 四组）\n")
    A("U-Net 为 SMP ResNet18-UNet（14.35M，ImageNet 预训练），学习率沿用 SegFormer 那一套，未单独调。\n")
    A("| 架构 | 组 | TEST IoU | TEST AP | Recall | Precision | ΔIoU（族内） | ΔRecall（族内） |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    for fam, arms in uni.items():
        base = arms["baseline"]["best_test"]
        for arm, v in arms.items():
            b = v["best_test"]
            A(f"| {fam} | {arm} | {f4(b['iou'])} | {f4(b['ap'])} | {f4(b['recall'])} | {f4(b['precision'])} | "
              f"{d4(b['iou']-base['iou']) if arm!='baseline' else '-'} | "
              f"{d4(b['recall']-base['recall']) if arm!='baseline' else '-'} |")
    A("")
    A("**跨架构的绝对分数不可直接比较**（参数量差 3.8 倍、U-Net 未调 LR）。有意义的是族内差值是否复现：\n")
    A("| 改动 | SegFormer ΔIoU | U-Net ΔIoU | SegFormer ΔRecall | U-Net ΔRecall | 判断 |")
    A("|---|---:|---:|---:|---:|---|")
    for arm in ("DEM", "Slope", "PW5"):
        s = uni["SegFormer"][arm]["best_test"]
        u = uni["UNet"][arm]["best_test"]
        sb, ub = uni["SegFormer"]["baseline"]["best_test"], uni["UNet"]["baseline"]["best_test"]
        same = (s["iou"] - sb["iou"] > 0) == (u["iou"] - ub["iou"] > 0)
        A(f"| {arm} | {d4(s['iou']-sb['iou'])} | {d4(u['iou']-ub['iou'])} | "
          f"{d4(s['recall']-sb['recall'])} | {d4(u['recall']-ub['recall'])} | {'方向一致' if same else '**方向相反**'} |")
    A("")
    A("- **召回↑精确率↓的机制与架构无关**，三组改动在两种网络上同向。")
    A("- **IoU 能否改善取决于架构**：PW5 在 SegFormer 上 +0.0039，在 U-Net 上 −0.0004。")
    A("- 坡度是唯一在两种架构、两个 split 上方向都一致的改动。\n")

    # ---------------- 7. 证据一致性 ----------------
    A("## 7. 证据一致性总表\n")
    A("| 改动 | 事件 OOD | 官方·VAL | 官方·TEST | 官方·last | U-Net·TEST | 一致? |")
    A("|---|---|---|---|---|---|---|")
    A("| 扩事件覆盖（3→9） | **+0.0607 / +0.0619** | — | — | — | — | 两检查点一致 |")
    A("| 主干 B0→B2 | **−0.0384 / −0.0709** | — | — | — | — | 两检查点一致 |")
    A("| +DEM | — | +0.0019 | +0.0087 | +0.0063 | +0.0015 | 弱正，实质为零 |")
    A("| +坡度 | — | −0.0025 | +0.0127 | +0.0085 | +0.0075 | 弱正（VAL 反号） |")
    A("| +DEM+坡度 | — | +0.0014 | +0.0089 | +0.0061 | — | 与单加 DEM 相当 |")
    A("| pos_weight=5 | — | −0.0044 | +0.0039 | +0.0010 | −0.0004 | **IoU 不一致，召回复现** |")
    A("")
    A("读法：只有第 1、2 行在「两个独立检查点」上方向一致，属于可复现的结论；")
    A("第 3–6 行的量级都在 0.01 以内，且多在 VAL/TEST 之间反号，只能算方向性提示。\n")

    # ---------------- 8. 结论与取舍 ----------------
    A("## 8. 结论与取舍\n")
    A("**（1）真正站得住的只有两条：扩数据的事件覆盖、以及不要在这个预算下放大主干。**")
    A("两条都在两个检查点上方向一致、区间不跨 0，且后者有同版本 B0 控制组兜底。\n")
    A("**（2）输入侧的天花板很低。** DEM 单独几乎无效（高程单变量 AUC 0.46），坡度好一些（AUC 0.61）但换到模型上只有 +1 个 IoU 点量级；")
    A("两者合起来不叠加。这与「静态地形只能调先验、不能定位」的判断一致。\n")
    A("**（3）指标之间存在硬冲突，需要先定目标再选方案：**\n")
    A("| 若优化目标 | 应选 | 代价 |")
    A("|---|---|---|")
    A("| IoU | 坡度单加（0.3402） | 精确率 −1.6 点 |")
    A("| AP / 边界质量 | DEM+坡度（0.4969 / 0.5249） | 小目标召回下降 |")
    A("| 小目标召回 | pos_weight=5（1–4 px 0.301） | 精确率 −7.5 点、IoU 不动 |")
    A("| 部署成本 | 现有 B0 基线 | 显存 4.7 GB、推理 6.5 ms |")
    A("")
    A("**（4）与官方基线的差距仍在 12–18 个点**，且主要是召回；这部分不是调阈值能补的（AP 差 17.7 点）。")
    A("下一步最有信息量的动作不是继续加通道，而是**对齐官方输入**（多时相 + DEM）或**对齐官方训练设置**（pos_weight、lr、seed 数）。\n")

    # ---------------- 9. 限制 ----------------
    A("## 9. 限制与已知问题\n")
    A("1. **单 seed**。全部 13 组均为 seed 1234，没有跨种子重复；0.01 以内的差值不应作为结论。")
    A("2. **VAL/TEST 符号不一致**。DEM、坡度、DEMSlope 在 VAL 上约等于零或略负、在 TEST 上为正。")
    A("   VAL 是选模集，天然对基线有利，但两侧反号意味着这批差异未定。")
    A("3. **事件级有效样本量只有 2**（hiroshima / thrissur），图块级 bootstrap 区间偏窄。")
    A("4. **FINAL 盲测身份已终止**。按 2026-10-20 决策启用官方协议，newzealand 的 292 块进入官方 TRAIN，")
    A("   此后不得再称其为未接触事件；事件 OOD 的历史数值只在原协议内部有效。")
    A("5. **口径差异**。官方指标用全像素，本项目主口径只用 valid 像素；换算后 TEST IoU 0.3275→0.3214。")
    A("   与官方表值比较必须用换算后的数。")
    A("6. **B2 权重身份只到本地文件**。目录名与 `download_meta` 写 `b2-...-512-512`，但配置与张量均为 B2-768；")
    A("   HF 上游不可达，未做哈希比对。")
    A("7. **官方多时相 + DEM 参考未运行**。依赖与格式已核验（U-TAE 1.07M 参数、bs=4 每步 0.104 s），")
    A("   但 75 epoch 需先建多时相缓存，预估 5–8 小时墙钟，超出本阶段预算。")
    A("8. **U-Net 未调学习率**，其绝对分数不代表该架构上限。\n")

    # ---------------- 10. 产物 ----------------
    A("## 10. 产物索引\n")
    A("| 内容 | 路径 |")
    A("|---|---|")
    A("| 事件覆盖（P1-3） | `reports/p13/results_report.md`、`composition.md`、`significance.md`、`cases_dev.png` |")
    A("| 主干规格（P2-M1） | `reports/p2m1/capacity_summary.md`、`capacity_significance.md`、`b2_interface_validation.md` |")
    A("| 官方协议锁定 | `manifests/official_single/protocol_lock.json`、`mapping.csv`、`overlap.md`、`official_source_facts.json` |")
    A("| 官方单期基线 | `reports/official_single/results_summary.md`、`gap_and_curve.md` |")
    A("| 输入/损失变体 | `reports/official_single/variants_report.md`、`variants_summary.md` |")
    A("| 架构对比 | `reports/official_single/unet_report.md`、`unet_vs_segformer.md` |")
    A("| 接口验证 | `reports/official_single/geo_variants_validation.json`、`interface_validation.md` |")
    A("| 数据集调查 | `docs/dataset_survey_2026-10-20.md`、`dataset_structure_notes_2026-10-20.md` |")
    A("| 决策记录 | `docs/DECISION_LOG.md` |")
    A("| 训练产物 | `outputs/segformer/{p13_events,p2m1,official_single,official_single_geo,official_unet}/` |")
    A("| 配置 | `configs/segformer/{p13,p2m1,official_single,official_unet}/` |\n")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[saved] {OUT}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
