"""Summarize actual completed probes and compare identical preselected DEV cases."""
from __future__ import annotations
import csv
import json
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/segformer/plot_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from landslide_seg.config import file_sha256, load_segformer_config
from landslide_seg.training.segformer_metrics import label_objects, object_table, match_objects

OUT = ROOT / "reports/v4r2_probe_results"
SOURCE = ROOT / "outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_n100_u3600_s1234/1234"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rgb(x):
    image = np.moveaxis(x[[2, 1, 0]], 0, -1).astype(float)
    lo, hi = np.percentile(image, [2, 98], axis=(0, 1))
    return np.clip((image-lo) / np.maximum(hi-lo, 1), 0, 1)


def error_mask(prob, gt, valid):
    pred = (prob >= .5) & valid
    return np.stack((pred & ~gt & valid, pred & gt & valid, ~pred & gt & valid), -1).astype(float)


def main():
    status = load(ROOT / "outputs/segformer/v4r2_probe_batch/status.json")
    assert status["state"] == "complete" and len(status["experiments"]) == 2
    OUT.mkdir(exist_ok=True)
    rows, all_evaluations, curves, audits = [], [], [], []
    runs = [("joint_finetune", SOURCE)] + [("frozen_" + entry["source"], Path(entry["run_dir"])) for entry in status["experiments"]]
    for label, run in runs:
        summary = load(run / "train_summary.json")
        assert summary["global_step"] == 3600 and summary["samples_exposed"] == 230400
        curves.extend(dict(condition=label, **row) for row in summary["history"])
        meta = load(run / "run_meta.json")
        if label != "joint_finetune":
            assert len(summary["history"]) == 36 and summary["skipped_batches"] == 0
            assert [h["global_step"] for h in summary["history"]] == list(range(100, 3601, 100))
            for history in summary["history"]:
                assert history["probe_integrity"]["encoder_unchanged"] and history["probe_integrity"]["decoder_changed"]
                assert not history["probe_integrity"]["encoder_has_grad"] and not history["probe_integrity"]["encoder_training"]
            for relative, digest in meta["source_sha256"].items():
                assert file_sha256(ROOT / relative) == digest
            assert meta["model_identity"]["probe"]["decoder_initial_sha256"] == status["common_decoder_initial_sha256"]
            assert summary["params"]["trainable"] == 395009
            maximum = max(summary["history"], key=lambda h: h["val_iou"])
            assert maximum["global_step"] == summary["best_step"]
            assert maximum["val_iou"] == summary["best_val_iou"]
            exposures = load(run / "sample_exposures.json")
            assert len(exposures) == 2223 and sum(exposures.values()) == 230400
            assert max(exposures.values()) - min(exposures.values()) <= 1
            integrity = load(run / "probe_integrity.json")
            audits.append({"condition": label, "source": meta["model_identity"]["probe"], "saved": integrity,
                           "updates": 3600, "dev_checks": 36, "samples_exposed": 230400,
                           "source_sha256": meta["source_sha256"]})
        evaluations = {tag: load(run / "eval" / f"eval_{tag}.json") for tag in ("best_val", "last_val", "best_train", "last_train")}
        for tag, evaluation in evaluations.items():
            checkpoint, split = tag.split("_")
            assert evaluation["checkpoint_sha256"] == file_sha256(run / f"{checkpoint}.pt")
            assert evaluation["stats_sha256"] == meta["stats_sha256"]
            assert evaluation["manifest_sha256"] == file_sha256(ROOT / "manifests/v4r1/pool_100.jsonl")
            assert evaluation["role"] == ("TRAIN" if split == "train" else "DEV")
        all_evaluations.append({"condition": label, "evaluations": evaluations})
        for checkpoint in ("best", "last"):
            evaluation = evaluations[f"{checkpoint}_val"]
            assert evaluation["checkpoint_sha256"] == file_sha256(run / f"{checkpoint}.pt")
            assert evaluation["stats_sha256"] == meta["stats_sha256"]
            assert set(evaluation["by_event"]) == {"hiroshima", "thrissur"}
            pixel = evaluation["overall"]["pixel"]
            assert abs(pixel["iou"] - pixel["tp"] / (pixel["tp"] + pixel["fp"] + pixel["fn"])) < 1e-12
            mean = np.mean([event["pixel"]["iou"] for event in evaluation["by_event"].values()])
            assert abs(mean - evaluation["event_mean_iou"]) < 1e-12
            objects = evaluation["overall"]["objects"]
            row = {"condition": label, "checkpoint": checkpoint,
                   "step": summary["best_step"] if checkpoint == "best" else 3600,
                   "train_iou": evaluations[f"{checkpoint}_train"]["overall"]["pixel"]["iou"],
                   "dev_mean_iou": float(mean), "pooled_iou": pixel["iou"], "dice": pixel["dice"],
                   "precision": pixel["precision"], "recall": pixel["recall"], "ap": pixel["average_precision"],
                   "hiroshima_iou": evaluation["by_event"]["hiroshima"]["pixel"]["iou"],
                   "thrissur_iou": evaluation["by_event"]["thrissur"]["pixel"]["iou"],
                   "object_recall": objects["object_recall_evaluable"],
                   "boundary_f1": evaluation["overall"]["boundary"]["f1"],
                   "params": summary["params"]["total"], "trainable_params": summary["params"]["trainable"],
                   "peak_alloc_mib": summary["peak_alloc_mib"], "training_seconds": summary["seconds"],
                   "latency_b1_ms": evaluations["best_val"]["efficiency"]["latency_b1"]["mean_ms"],
                   "checkpoint_sha256": file_sha256(run / f"{checkpoint}.pt"),
                   "negative_image_fpr": None, "final_evaluated": False}
            for name, values in objects["fine_bins"].items():
                row[f"matched_{name}"] = values["n_matched_evaluable"]
                row[f"objects_{name}"] = values["n_objects_evaluable"]
                row[f"recall_{name}"] = values["object_recall_evaluable"]
            rows.append(row)
    assert audits[0]["source_sha256"] == audits[1]["source_sha256"]
    assert audits[0]["source"]["source_stats_sha256"] == audits[1]["source"]["source_stats_sha256"]
    assert audits[0]["source"]["source_manifest_sha256"] == audits[1]["source"]["source_manifest_sha256"]
    assert audits[0]["source"]["encoder_initial_sha256"] != audits[1]["source"]["encoder_initial_sha256"]
    write_csv(OUT / "probe_comparison.csv", rows)
    (OUT / "full_evaluations.json").write_text(json.dumps(all_evaluations, indent=1), encoding="utf-8")
    (OUT / "curves.json").write_text(json.dumps(curves, indent=1), encoding="utf-8")
    (OUT / "completion_audit.json").write_text(json.dumps({"status": "PASS", "seed": 1234, "n_new_runs": 2,
        "final_evaluated": False, "common_decoder_initial_sha256": status["common_decoder_initial_sha256"], "runs": audits}, indent=1), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for label, _ in runs:
        history = [row for row in curves if row["condition"] == label]
        steps = [r["global_step"] for r in history]
        for ax, values in zip(axes, ([r["train_loss"] for r in history], [r["train_clean"]["pooled"]["iou"] for r in history], [r["val_iou"] for r in history])):
            ax.plot(steps, values, "--" if label == "joint_finetune" else "-", label=label)
    for ax, title in zip(axes, ("Training loss", "Clean TRAIN IoU", "DEV event mean IoU")):
        ax.set(title=title, xlabel="Optimizer updates")
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "probe_curves.png", dpi=160)
    plt.close(fig)
    # Fixed before seeing probe results: use original standard model's DEV selections.
    source_name = "segformer_b0_direct10_n100_u3600_s1234"
    cases = [case for case in load(ROOT / "reports/v4r1_results/error_atlas/selection.json") if case["experiment"] == source_name and case["available"]]
    records = {r["sample_id"]: r for r in [json.loads(line) for line in (ROOT / "manifests/v4r1/pool_100.jsonl").read_text(encoding="utf-8").splitlines()]}
    arrays = {}
    for label, run in runs:
        with np.load(run / "eval/predictions_best_val.npz") as prediction:
            arrays[label] = {key: prediction[key].copy() for key in prediction.files}
    reference = arrays["joint_finetune"]
    for data in arrays.values():
        assert np.array_equal(data["sample_ids"], reference["sample_ids"])
        assert np.array_equal(data["gt"], reference["gt"]) and np.array_equal(data["valid"], reference["valid"])
    indices = {str(sid): i for i, sid in enumerate(reference["sample_ids"])}
    # Same original GT object identities for all models; reuse v2 matching exactly.
    paired_objects = []
    for sid, idx in indices.items():
        gt, valid = reference["gt"][idx], reference["valid"][idx]
        labels, n = label_objects(gt)
        objects = [obj for obj in object_table(labels, n, valid) if not obj["truncated"]]
        hits = {}
        for label, data in arrays.items():
            predicted, count = label_objects((data["probability"][idx] >= .5) & valid)
            hits[label] = match_objects(labels, objects, predicted, object_table(predicted, count, None), .25)
        for obj in objects:
            area = obj["area"]
            size = "1_4" if area < 5 else "5_9" if area < 10 else "10_19" if area < 20 else "20_49" if area < 50 else "50_199" if area < 200 else "ge200"
            paired_objects.append({"sample_id": sid, "event": records[sid]["event_id"], "gt_object_id": obj["id"],
                                   "area_px": area, "size_bin": size, "bbox": json.dumps(obj["bbox"]),
                                   **{label + "_hit": int(obj["id"] in matched) for label, matched in hits.items()}})
    for label, _ in runs:
        evaluation = next(item for item in all_evaluations if item["condition"] == label)["evaluations"]["best_val"]
        assert sum(o[label + "_hit"] for o in paired_objects) == evaluation["overall"]["objects"]["n_matched_evaluable"]
        assert len(paired_objects) == evaluation["overall"]["objects"]["n_gt_evaluable"]
    write_csv(OUT / "paired_dev_objects.csv", paired_objects)
    overlap = []
    for left, right in (("joint_finetune", "frozen_best"), ("joint_finetune", "frozen_last"), ("frozen_best", "frozen_last")):
        for size in ("all", "1_4", "5_9", "10_19", "20_49", "50_199", "ge200"):
            subset = [o for o in paired_objects if size == "all" or o["size_bin"] == size]
            overlap.append({"left": left, "right": right, "size_bin": size, "n_objects": len(subset),
                            "both_hit": sum(o[left+"_hit"] and o[right+"_hit"] for o in subset),
                            "left_only": sum(o[left+"_hit"] and not o[right+"_hit"] for o in subset),
                            "right_only": sum(not o[left+"_hit"] and o[right+"_hit"] for o in subset),
                            "both_missed": sum(not o[left+"_hit"] and not o[right+"_hit"] for o in subset)})
    write_csv(OUT / "paired_object_overlap.csv", overlap)
    case_rows = []
    fig, axes = plt.subplots(len(cases), 7, figsize=(21, 3 * len(cases)), squeeze=False)
    for i, case in enumerate(cases):
        sid = case["sample_id"]
        idx, record = indices[sid], records[sid]
        gt, valid = reference["gt"][idx], reference["valid"][idx]
        with np.load(record["cache"]) as cache:
            color = rgb(cache["x10"])
        panels = [(color, f"{case['event']} {case['category']}\n{sid} {record['date']}"), (gt, "GT (visibility unreviewed)")]
        for label in ("joint_finetune", "frozen_best", "frozen_last"):
            probability = arrays[label]["probability"][idx]
            pred = probability >= .5
            tp, fp, fn = int((pred & gt & valid).sum()), int((pred & ~gt & valid).sum()), int((~pred & gt & valid).sum())
            case_rows.append({"sample_id": sid, "event": case["event"], "category": case["category"], "condition": label,
                              "tp": tp, "fp": fp, "fn": fn, "iou": tp / (tp+fp+fn) if tp+fp+fn else None,
                              "selection": "fixed original standard model DEV numerical cases", "visibility_reviewed": False})
            if label != "joint_finetune":
                panels.append((probability, label + " probability"))
            panels.append((error_mask(probability, gt, valid), label + " redFP greenTP blueFN"))
        for ax, (arr, title) in zip(axes[i], panels):
            ax.imshow(arr, **({} if arr.ndim == 3 else {"cmap": "gray", "vmin": 0, "vmax": 1}))
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fixed_dev_cases.png", dpi=110)
    plt.close(fig)
    write_csv(OUT / "fixed_dev_cases.csv", case_rows)
    (OUT / "fixed_case_selection.json").write_text(json.dumps(cases, indent=1), encoding="utf-8")
    lines = ["# v4r2：冻结编码器、共同解码器诊断结果", "", "两个正式probe均为真实训练：2223 TRAIN / 669 DEV、seed1234、batch64、3600更新、每100步双事件均值IoU选模；不评FINAL。", "",
             "| 条件 | checkpoint/步数 | TRAIN IoU | DEV均值IoU | Hiroshima | Thrissur | AP | 对象召回 | tiny命中 | 边界F1 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['condition']} | {r['checkpoint']}/{r['step']} | {r['train_iou']:.4f} | {r['dev_mean_iou']:.4f} | {r['hiroshima_iou']:.4f} | {r['thrissur_iou']:.4f} | {r['ap']:.4f} | {r['object_recall']:.4f} | {r['matched_1_4']}/{r['objects_1_4']} | {r['boundary_f1']:.4f} |")
    lines += ["", "## 成本", "", "| 条件 | 总参数 | 可训练参数 | 训练秒数（含周期评测） | 峰值allocated MiB | batch1 ms |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["checkpoint"] == "best":
            lines.append(f"| {r['condition']} | {r['params']} | {r['trainable_params']} | {r['training_seconds']:.1f} | {r['peak_alloc_mib']:.1f} | {r['latency_b1_ms']:.2f} |")
    lines += ["", "joint_finetune是上一阶段标准模型的历史参照；两个probe只加载其对应状态的编码器，decoder均从同一官方权重与相同二值头初始化重新训练。冻结编码器共3330368参数；可训练decoder共395009参数。", "",
              "## 已验证与解释边界", "", "- 25项定点测试；真实GPU13步、冻结模式与梯度检查、输出尺寸和重载误差0.0。",
              "- 两正式组各36次检查，编码器参数/缓冲区哈希不变、无梯度，decoder确实改变；初始decoder哈希两组相同。best/last保存状态再次核验。",
              "- 这是非线性decoder probe；冻结encoder处于eval，区别于历史联合微调中的训练随机层。不同表征上的head优化难度可能不同，不作遗忘或因果证明。",
              "- 来源best已使用本DEV选择；本轮继续使用相同DEV，所有结论为开发诊断，不能称独立泛化验证。",
              "- 各系统均维持0.5阈值；可靠负影像FPR仍为N/A。单seed不声称统计显著。",
              "- 固定案例沿用此前标准模型的选例ID，GT与valid已逐像素对齐；没有按新模型结果重新挑例，标签可见性仍未人工确认。",
              "- 训练成本包含完整TRAIN/DEV周期推理与checkpoint保存，不等于纯反传时间。", "",
              "## 产物", "", "`reports/v4r2_probe_results/`含完整CSV/JSON、训练曲线、固定案例、完成核验；`configs/segformer/v4r2_probe/`含两组独立配置；`reports/v4r2_methods/mechanism_review.md`为官方方法归纳。"]
    interpretation = OUT / "interpretation.md"
    if interpretation.exists():
        lines += ["", interpretation.read_text(encoding="utf-8")]
    report = ROOT / "reports/v4r2_probe_execution_report_2026-10-19.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)
    for row in rows:
        print(row["condition"], row["checkpoint"], "DEV", row["dev_mean_iou"], "AP", row["ap"], "tiny", row["matched_1_4"])


if __name__ == "__main__":
    main()
