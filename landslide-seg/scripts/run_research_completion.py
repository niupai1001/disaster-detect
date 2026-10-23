"""有限补全队列：3/5/7期、修正评分迁移；每任务失败即停，不搜索超参。"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/research_completion"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status = OUT / "status.json"
    if status.exists():
        raise FileExistsError("补全队列已有状态；先检查日志，不自动覆盖或重跑")
    quality = json.loads((ROOT / "docs/assets/research/temporal_observations.json").read_text(encoding="utf-8"))
    if quality["errors"] or quality["n_samples"] != 2892 or quality["nodata_pixels"]:
        raise ValueError("时序全量质量门禁未通过")
    jobs = []
    for n in (3, 5, 7):
        cfg = f"configs/segformer/v4r1_temporal_control/frames{n}.yaml"
        jobs.append((f"preflight_frames{n}", ["scripts/check_research_protocol.py", "--config", cfg]))
        jobs.append((f"train_frames{n}", ["scripts/train_temporal.py", "--config", cfg, "--device", "cuda:1"]))
        run = f"outputs/segformer/v4r1_temporal_frame_count/SegFormer-B0-Temporal-Date{n}/1234"
        for tag in ("best", "last", "step1200", "step2400", "step3600"):
            jobs.append((f"eval_frames{n}_{tag}", ["scripts/evaluate_temporal.py", "--checkpoint", f"{run}/{tag}.pt", "--device", "cuda:1", "--splits", "val"]))
    cfg = "configs/segformer/v4r1_terrain_raw/scoring_full_v2.yaml"
    jobs += [
        ("terrain_scoring_eval", ["scripts/evaluate_terrain_scoring.py", "--checkpoint", "weights/terrain/v4r1raw_scoring_valid_v2.pt", "--device", "cuda:1", "--out", "outputs/research_completion/terrain_scoring_valid_v2.json"]),
        ("preflight_terrain", ["scripts/check_research_protocol.py", "--config", cfg]),
        ("train_terrain", ["scripts/train_segformer.py", "--config", cfg, "--device", "cuda:1"]),
    ]
    run = "outputs/segformer/s2_10band_v4r1_geo_scoring_v2/SegFormer-B0-LateDetail-OODv4-TerrainScoringFullV2/1234"
    for tag in ("best", "last"):
        jobs.append((f"eval_terrain_{tag}", ["scripts/evaluate_segformer.py", "--checkpoint", f"{run}/{tag}.pt", "--device", "cuda:1", "--split", "val", "--tag", f"{tag}_val"]))
    state = {"started": time.time(), "status": "RUNNING", "jobs": [
        {"name": name, "command": [sys.executable, "-X", "utf8", *args], "status": "PENDING"} for name, args in jobs]}
    def save():
        tmp = status.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(status)
    save()
    for job in state["jobs"]:
        job.update(status="RUNNING", started=time.time())
        save()
        with (OUT / f"{job['name']}.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run(job["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        job.update(status="FINISHED_PROCESS" if proc.returncode == 0 else "FAILED", exit_code=proc.returncode, ended=time.time())
        if proc.returncode:
            state["status"] = "FAILED"
            save()
            raise SystemExit(proc.returncode)
        save()
    state.update(status="PROCESSES_FINISHED_REQUIRES_RESULT_REVIEW", ended=time.time())
    save()


if __name__ == "__main__":
    main()
