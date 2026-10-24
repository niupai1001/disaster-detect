"""核验固定基线及时序控制的真实权重、哈希、预算与评价；不训练。"""
from __future__ import annotations

import json
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from landslide_seg.config import file_sha256, resolve_path
from landslide_seg.models.factory import build_model
from landslide_seg.models.spectral_temporal import build_temporal_model
from landslide_seg.data.segformer_dataset import load_stats
from landslide_seg.training.budget import resolve_update_budget


def validate_temporal_evaluation(ev, digest, ids, tile_rows):
    """检查实际评价成员与权重身份；人数相同不能代替成员相同。"""
    assert ev.get("checkpoint_sha256") == digest, "评价权重哈希缺失或不一致"
    assert ev["val"]["n_samples"] == len(ids), "评价成员数错误"
    assert set(ev["val"]["by_event"]) == {"hiroshima", "thrissur"}, "评价事件错误"
    observed = [r["sample_id"] for r in tile_rows]
    assert len(observed) == len(ids) and set(observed) == ids, "逐图块评价成员重复或与冻结验证集不一致"
    expected = sum(v["pixel"]["iou"] for v in ev["val"]["by_event"].values()) / 2
    assert abs(ev["val_event_mean_iou"] - expected) < 1e-6, "事件等权IoU与逐事件结果不一致"


def main():
    out = ROOT / "docs/assets/research/execution_audit.json"
    runs = [ROOT / "outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_detailres_n100_u3600_s1234/1234"]
    runs += sorted((ROOT / "outputs/segformer/v4r1_temporal_control").glob("SegFormer*/1234"))
    runs += sorted((ROOT / "outputs/segformer/v4r1_temporal_frame_count").glob("SegFormer*/1234"))
    runs += sorted((ROOT / "outputs/segformer/s2_10band_v4r1_geo_scoring_v2").glob("SegFormer*/1234"))
    # 派生共同子集的两个新批次；工程 smoke 目录不计正式成绩
    for batch in ("v4r2_temporal_cutoff", "v4r2_temporal_window", "v4r3_composition",
                  "s2_10band_v4r2_spectral_mapping", "v4r2_spectral"):
        runs += sorted(p for p in (ROOT / "outputs/segformer" / batch).glob("SegFormer*/1234")
                       if "Smoke" not in p.parent.name)
    result = {"as_of": "2026-10-23", "scope": "原固定单期基线、时间控制与新补全队列；其他历史运行未在此重验权重",
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "runs": []}
    for run in runs:
        record = {"run_dir": run.relative_to(ROOT).as_posix(), "errors": [], "checkpoints": {}}
        if not (run / "train_summary.json").is_file():
            record["status"] = "PARTIAL_NO_COMPLETION_SUMMARY"
            result["runs"].append(record)
            print(record["run_dir"], record["status"], flush=True)
            continue
        try:
            summary = json.loads((run / "train_summary.json").read_text(encoding="utf-8"))
            cfg_path = run / "resolved_config.json"
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))["segformer"]
            tc = cfg["train"]
            updates = str(tc.get("budget_mode")) == "updates" or "stop_updates" in tc or "schedule_updates" in tc
            budget = resolve_update_budget(tc) if updates else None
            target = budget.stop_updates if budget else int(tc.get("max_steps") or summary.get("global_step") or 0)
            record.update(config_sha256=file_sha256(cfg_path), seed=cfg["seed"],
                          global_step=summary.get("global_step"), target_updates=target,
                          schedule_updates=budget.schedule_updates if budget else None,
                          report_steps=list(tc.get("report_steps") or ()),
                          selection=cfg["eval"]["selection_metric"])
            record["files"] = {key: {"path": cfg["data"][key], "sha256": file_sha256(resolve_path(cfg["data"][key]))}
                               for key in ("manifest", "stats", "protocol")}
            extra_stats = (cfg.get("model", {}).get("spectral_ladder", {}) or {}).get("extra_stats")
            if extra_stats:
                ex_path = resolve_path(extra_stats)
                record["extra_stats"] = {"path": str(ex_path.relative_to(ROOT)).replace("\\", "/"),
                                         "sha256": file_sha256(ex_path)}
            assert summary.get("global_step") == target, "更新数不足"
            assert summary.get("complete"), "没有完成标记"
            stats = load_stats(resolve_path(cfg["data"]["stats"]))
            temporal = cfg["model"]["type"] == "temporal_spectral"
            # 预定 checkpoint 一律来自该 run 自己的 resolved_config，不硬编码 1200/2400/3600
            steps = [int(x) for x in tc.get("report_steps") or ()]
            tags = ("best", "last", *(f"step{s}" for s in steps)) if temporal else ("best", "last")
            paths = {tag: run / "eval" / (f"eval_{tag}.json" if temporal else f"eval_{tag}_val.json") for tag in tags}
            missing = [str(path.relative_to(ROOT)) for path in paths.values() if not path.is_file()]
            if missing:
                record.update(status="TRAINED_AWAITING_EVALUATION", missing_evaluations=missing)
                result["runs"].append(record)
                print(record["run_dir"], record["status"], flush=True)
                continue
            ids = {r["sample_id"] for r in map(json.loads, resolve_path(cfg["data"]["manifest"]).read_text(encoding="utf-8-sig").splitlines()) if r["split"] == "val"}
            model = build_temporal_model(cfg, stats) if temporal else build_model(cfg)
            record["actual_model_class"] = type(model).__name__
            if extra_stats:
                ex = json.loads(resolve_path(extra_stats).read_text(encoding="utf-8"))
                assert model.ladder.extra_normalized, "S1 新增通道未走 TRAIN 统计归一化"
                assert torch.allclose(model.ladder.extra_mean.cpu(), torch.tensor(ex["mean"], dtype=torch.float32))
                assert torch.allclose(model.ladder.extra_std.cpu(), torch.tensor(ex["std"], dtype=torch.float32))
                record["extra_stats_normalization"] = "VERIFIED_IN_MODEL_BUFFERS"
            for tag in tags:
                path = run / f"{tag}.pt"
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                assert ckpt["config"]["segformer"] == cfg, f"{tag} checkpoint配置与resolved_config不同"
                model.load_state_dict(ckpt["model"], strict=True)
                digest = file_sha256(path)
                ep = paths[tag]
                ev = json.loads(ep.read_text(encoding="utf-8"))
                if ev.get("checkpoint_sha256"):
                    assert ev["checkpoint_sha256"] == digest, f"{tag} 评价权重哈希不一致"
                if temporal:
                    with (run / "eval" / f"per_tile_{tag}_val.csv").open(encoding="utf-8-sig", newline="") as f:
                        validate_temporal_evaluation(ev, digest, ids, list(csv.DictReader(f)))
                record["checkpoints"][tag] = {"sha256": digest, "step": ckpt.get("global_step"),
                                               "evaluation": ep.relative_to(ROOT).as_posix(),
                                               "evaluation_sha256": file_sha256(ep),
                                               "event_mean_iou": ev.get("val_event_mean_iou")}
                del ckpt
            record["status"] = "COMPLETED_VALID_FOR_AUDITED_SCOPE"
            del model
        except (OSError, ValueError, KeyError, RuntimeError, AssertionError) as exc:
            record["errors"].append(str(exc))
            record["status"] = "NEEDS_REPAIR_OR_EVIDENCE"
        result["runs"].append(record)
        print(record["run_dir"], record["status"], record["errors"], flush=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(any(r["errors"] for r in result["runs"]))


if __name__ == "__main__":
    main()
