"""汇总 event_ood_v4r1 六种地形方案（对照 v4r1 基线）。

输出 ``outputs/segformer/s2_10band_v4r1_p20_geo/summary_terrain.json`` 与 markdown 表。
所有数字都是 DEV（Hiroshima448+Thrissur221）上的，DEV 是开发集，不是盲测。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "segformer" / "s2_10band_v4r1_geo_raw"
BASELINE = ROOT / "outputs" / "segformer" / "s2_10band_v4r1_p20" / "segformer_b0_direct10_detailres_n100_u3600_s1234" / "1234"

ARMS = [
    ("当前参照：十波段影像＋细节分支", "segformer_b0_direct10_detailres_n100_u3600_s1234", "s2_10band_v4r1_p20"),
    ("12通道共同输入：影像＋高程＋坡度", "SegFormer-B0-LateDetail-OODv4-DEMSlopeRaw", None),
    ("独立地形分支：直接随分割训练", "SegFormer-B0-LateDetail-OODv4-TerrainCapControl", None),
    ("独立地形分支：先做地形自监督", "SegFormer-B0-LateDetail-OODv4-TerrainSSL", None),
    ("独立地形分支：加载标注训练的评分头", "SegFormer-B0-LateDetail-OODv4-TerrainSSLSup", None),
    ("独立地形分支：地形自监督预算加倍", "SegFormer-B0-LateDetail-OODv4-TerrainSSLBudget", None),
    ("独立地形分支：固定融合系数", "SegFormer-B0-LateDetail-OODv4-TerrainFrozenAlpha", None),
]


def run_dir(exp: str, version: str | None) -> Path:
    return (BASE if version is None else ROOT / "outputs" / "segformer" / version) / exp / "1234"


def read(exp: str, version: str | None) -> dict:
    d = run_dir(exp, version)
    out = {"dir": str(d.relative_to(ROOT)).replace("\\", "/"), "state": "missing"}
    ts = d / "train_summary.json"
    ev = d / "eval" / "eval_val.json"
    if not ev.is_file():
        ev = d / "eval" / "eval_best_val.json"
    if ts.is_file():
        j = json.loads(ts.read_text(encoding="utf-8"))
        out.update(state="trained", best_val=j.get("best_val_iou"), best_step=j.get("best_step"),
                   seconds=j.get("seconds"), peak=j.get("peak_alloc_mib"),
                   params=j.get("params", {}))
    if ev.is_file():
        e = json.loads(ev.read_text(encoding="utf-8"))
        px = e["overall"]["pixel"]
        out.update(state="done", iou=px["iou"], ap=px["average_precision"], dice=px["dice"],
                   precision=px["precision"], recall=px["recall"],
                   object_recall=e["overall"]["objects"]["object_recall_evaluable"],
                   boundary_f1=e["overall"]["boundary"]["f1"],
                   per_event={k: v["pixel"]["iou"] for k, v in e["by_event"].items()})
    ck = d / "best.pt"
    if ck.is_file():
        sd = torch.load(ck, map_location="cpu", weights_only=False)["model"]
        if "alpha" in sd:
            out["alpha"] = float(sd["alpha"])
    return out


def f(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def main() -> None:
    rows = []
    for label, exp, version in ARMS:
        r = read(exp, version)
        r.update(label=label, exp=exp)
        rows.append(r)
    base = rows[0]
    lines = ["# 高程与坡度对照（广岛与特里苏尔验证669块，随机种子1234）\n",
             "验证事件平均IoU先分别计算两事件再等权平均；像素汇总IoU将有效像素合并统计。系数α来自最佳模型，不是训练末态。\n",
             "| 方案 | 验证事件平均 IoU | 相对基线 | 广岛 IoU | 特里苏尔 IoU | 像素汇总 IoU | AP | Dice | Precision | Recall | 对象召回 | 边界 F1 | 最佳模型α | 峰值显存 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        d = None if (r.get("best_val") is None or base.get("best_val") is None) else r["best_val"] - base["best_val"]
        lines.append(
            f"| {r['label']} | {f(r.get('best_val'))} | {'—' if d is None else f'{d:+.4f}'} | "
            f"{f((r.get('per_event') or {}).get('hiroshima'))} | {f((r.get('per_event') or {}).get('thrissur'))} | "
            f"{f(r.get('iou'))} | {f(r.get('ap'))} | {f(r.get('dice'))} | {f(r.get('precision'))} | {f(r.get('recall'))} | "
            f"{f(r.get('object_recall'))} | {f(r.get('boundary_f1'))} | "
            f"{'—' if r.get('alpha') is None else f'{r[chr(97) + chr(108) + chr(112) + chr(104) + chr(97)]:.4f}'} | "
            f"{'—' if r.get('peak') is None else f'{r[chr(112) + chr(101) + chr(97) + chr(107)]:.0f} MiB'} |")
    md = "\n".join(lines) + "\n"
    (BASE / "summary_terrain.md").write_text(md, encoding="utf-8")
    (BASE / "summary_terrain.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
