"""汇总 S1/S3/S4 光谱批次的 best/last/step1000/step1500 评测与参照。

只读已保存的 run、配置与 eval JSON；不重训、不改权重。输出
``outputs/segformer/v4r2_spectral/summary_spectral.json`` 并打印合并表。
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "segformer" / "v4r2_spectral"
OUT = ROOT / "outputs" / "segformer" / "v4r2_spectral" / "summary_spectral.json"

ARMS = [
    ("s1_indices", "S1 指数", "十波段＋四个指数，14 通道，指数用 TRAIN 统计标准化"),
    ("s1_control", "S1 容量对照", "十波段＋四个原始波段，14 通道，dataset z-score"),
    ("s3_a_linear", "S3-A 线性混合", "逐像元满秩 10×10 线性混合（非注意力）"),
    ("s3_b_idattn", "S3-B ID 注意力", "band token 自注意力＋自由可学习 band ID，无关系偏置"),
    ("s3_c_wavelength_attn", "S3-C 波长注意力", "同一注意力＋真实波长嵌入，无关系偏置"),
    ("s3_d_local_bias", "S3-D 局部偏置", "波长注意力＋每 4×4 块局部物理关系偏置"),
    ("s3_e_global_bias", "S3-E 全局偏置", "波长注意力＋整幅图共享关系偏置"),
    ("s4_add", "S4-ADD", "独立 F_spec 在解码特征处同尺度相加，无门控"),
    ("s4_fixed_gate", "S4-FIXEDGATE", "同一 F_spec，常数门 0.5"),
    ("s4_learn_gate", "S4-LEARNGATE", "同一 F_spec，输入依赖空间学习门"),
    ("s4_capctrl", "S4-CAPCTRL", "同容量非物理分支：自由 ID、无关系偏置、学习门"),
]

REFERENCES = {
    "baseline_v4r1_3600_best500": {
        "display": "十波段单期参照（3600 步日程，best@500，在 2000 窗口内）",
        "run_dir": "outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_detailres_n100_u3600_s1234/1234",
    },
    "s2_wavelength_map": {
        "display": "S2 波长条件映射（2000 步）",
        "run_dir": "outputs/segformer/s2_10band_v4r2_spectral_mapping/SegFormer-B0-LateDetail-OODv4-S2WavelengthMap/1234",
    },
    "s2_free_map": {
        "display": "S2 自由通道混合（2000 步）",
        "run_dir": "outputs/segformer/s2_10band_v4r2_spectral_mapping/SegFormer-B0-LateDetail-OODv4-S2FreeMap/1234",
    },
}


def cfg_of(arm: str) -> dict:
    return yaml.safe_load((CFG / f"{arm}.yaml").read_text(encoding="utf-8-sig"))["segformer"]


def run_dir(arm: str) -> Path:
    c = cfg_of(arm)
    return ROOT / c["output_dir"] / c["data"]["version"] / c["experiment"] / str(c["seed"])


def compact(path: Path) -> dict | None:
    if not path.is_file():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    ov = d["overall"]
    px, obj, bd = ov["pixel"], ov["objects"], ov["boundary"]
    tiny = ov.get("objects", {}).get("fine_bins", {}).get("1_4", {})
    return {
        "checkpoint_sha256": d.get("checkpoint_sha256"),
        "event_mean_iou": d.get("event_mean_iou"),
        "pooled_iou": px["iou"],
        "dice": px["dice"],
        "precision": px["precision"],
        "recall": px["recall"],
        "average_precision": px.get("average_precision"),
        "object_recall_evaluable": obj.get("object_recall_evaluable"),
        "boundary_f1": bd.get("f1"),
        "tp": px["tp"], "fp": px["fp"], "fn": px["fn"],
        "by_event_iou": {k: v["pixel"]["iou"] for k, v in d["by_event"].items()},
        "tiny_objects": {"matched_evaluable": tiny.get("n_matched_evaluable"),
                          "evaluable": tiny.get("n_objects_evaluable")},
        "n_samples": d.get("n_samples"),
    }


def branch_activity(run: Path) -> dict | None:
    """读取 best checkpoint 的新增分支权重幅度，确认零初始化后确实离开零。"""
    ck = run / "best.pt"
    if not ck.is_file():
        return None
    try:
        import torch
        state = torch.load(ck, map_location="cpu", weights_only=False)["model"]
    except Exception:
        return None
    keys = ("relation.readout.weight", "relation.relation.2.weight",
            "feature.proj.weight", "feature.block.relation.2.weight",
            "gate.weight", "alpha")
    return {k: float(v.abs().sum()) for k, v in state.items()
            if any(k.endswith(s) or k == s for s in keys)}


def arm_record(arm: str) -> dict:
    rd = run_dir(arm)
    summary = json.loads((rd / "train_summary.json").read_text(encoding="utf-8")) if (rd / "train_summary.json").is_file() else {}
    meta = json.loads((rd / "run_meta.json").read_text(encoding="utf-8")) if (rd / "run_meta.json").is_file() else {}
    evals = {}
    for tag, ckpt in (("best", "best.pt"), ("last", "last.pt"),
                      ("step1000", "step1000.pt"), ("step1500", "step1500.pt")):
        evals[tag] = compact(rd / "eval" / f"eval_{tag}_val.json")
        if evals[tag] is not None:
            evals[tag]["step"] = summary.get("best_step") if tag == "best" else (
                2000 if tag == "last" else int(tag.replace("step", "")))
    params = summary.get("params") or meta.get("params") or {}
    return {
        "arm": arm,
        "experiment": cfg_of(arm)["experiment"],
        "config": f"configs/segformer/v4r2_spectral/{arm}.yaml",
        "run_dir": rd.relative_to(ROOT).as_posix(),
        "status": ("COMPLETED_VALID_FOR_AUDITED_SCOPE" if summary.get("complete") and evals["best"]
                   else "TRAINED_AWAITING_EVALUATION" if summary.get("complete")
                   else "INCOMPLETE"),
        "global_step": summary.get("global_step"),
        "best_step": summary.get("best_step"),
        "seconds": summary.get("seconds"),
        "params": params,
        "spectral": meta.get("model_identity", {}).get("spectral"),
        "branch_activity": branch_activity(rd),
        "evals": evals,
    }


def reference(name: str) -> dict:
    rd = ROOT / REFERENCES[name]["run_dir"]
    rec = arm_record_like(rd)
    rec["display"] = REFERENCES[name]["display"]
    rec["run_dir"] = REFERENCES[name]["run_dir"]
    return rec


def arm_record_like(rd: Path) -> dict:
    summary = json.loads((rd / "train_summary.json").read_text(encoding="utf-8")) if (rd / "train_summary.json").is_file() else {}
    return {
        "best": compact(rd / "eval" / "eval_best_val.json"),
        "best_step": summary.get("best_step"),
        "params": summary.get("params"),
    }


def main() -> None:
    arms = []
    for arm, display, mechanism in ARMS:
        rec = arm_record(arm)
        rec["display"] = display
        rec["mechanism"] = mechanism
        arms.append(rec)
    refs = {name: reference(name) for name in REFERENCES}
    baseline = refs["baseline_v4r1_3600_best500"]["best"]
    s2w = refs["s2_wavelength_map"]["best"]
    s2f = refs["s2_free_map"]["best"]
    for rec in arms:
        best = rec["evals"]["best"]
        if best and baseline:
            rec["delta_baseline"] = best["event_mean_iou"] - baseline["event_mean_iou"]
        if best and s2w and s2f:
            rec["delta_s2_reference"] = best["event_mean_iou"] - max(
                s2w["event_mean_iou"], s2f["event_mean_iou"])
    out = {
        "batch": "v4r2_spectral",
        "protocol": "event_ood_v4r1",
        "data_split_display": "我们的数据划分（留事件验证）",
        "evaluation": "广岛448＋特里苏尔221共669块DEV；事件等权IoU选模；阈值0.5；单seed1234",
        "budget": {"stop_updates": 2000, "schedule_updates": 3600, "batch_size": 64,
                   "note": "单期批次；best 在 0–2000 步内选，step2000 与 last 同一权重"},
        "arms": arms,
        "references": refs,
        "notes": [
            "S3-D 同时是 S4-INPUTRES：入口前 x+Δx，不重复训练。",
            "S4-ADD/FIXEDGATE/LEARNGATE 共享同一关系编码器与 F_spec 投影，只改融合算子。",
            "S4-CAPCTRL 与 LEARNGATE 存储参数相同；其关系偏置 MLP 在 none 模式下不用，有效参数少 132（约 0.08%）。",
            "2000 步结果只与同日程前缀的旧运行比较；单 seed 不宣称显著性。",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print(f"{'arm/display':30s} {'best@':>6s} {'IoU':>7s} {'AP':>7s} {'广岛':>7s} {'特里苏尔':>7s} {'微小':>7s} {'last':>7s}")
    for rec in arms:
        b = rec["evals"]["best"]
        last = rec["evals"]["last"]
        if not b:
            print(f"{rec['display']:30s}  missing")
            continue
        tiny = b["tiny_objects"]
        print(f"{rec['display']:30s} {str(rec['best_step']):>6s} {b['event_mean_iou']:7.4f} "
              f"{(b['average_precision'] or 0):7.4f} {b['by_event_iou'].get('hiroshima', 0):7.4f} "
              f"{b['by_event_iou'].get('thrissur', 0):7.4f} "
              f"{tiny['matched_evaluable']}/{tiny['evaluable']!s:>3s} "
              f"{(last['event_mean_iou'] if last else 0):7.4f}")
    print(f"{'[参照] 十波段 3600':30s} {str(refs['baseline_v4r1_3600_best500']['best_step']):>6s} "
          f"{baseline['event_mean_iou']:7.4f} {baseline['average_precision']:7.4f} "
          f"{baseline['by_event_iou']['hiroshima']:7.4f} {baseline['by_event_iou']['thrissur']:7.4f} "
          f"{baseline['tiny_objects']['matched_evaluable']}/{baseline['tiny_objects']['evaluable']:3d}")
    for name in ("s2_wavelength_map", "s2_free_map"):
        b = refs[name]["best"]
        print(f"{'[参照] ' + REFERENCES[name]['display']:30s} {str(refs[name]['best_step']):>6s} "
              f"{b['event_mean_iou']:7.4f} {b['average_precision']:7.4f} "
              f"{b['by_event_iou']['hiroshima']:7.4f} {b['by_event_iou']['thrissur']:7.4f} "
              f"{b['tiny_objects']['matched_evaluable']}/{b['tiny_objects']['evaluable']:3d}")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
