"""Run the six predeclared v4r1 experiments, TRAIN/DEV evaluation only."""
from __future__ import annotations

import contextlib
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
torch.set_num_threads(4)
from landslide_seg.config import load_segformer_config, file_sha256
from landslide_seg.training.segformer_trainer import train_one
import evaluate_segformer


def main():
    gate = json.loads((ROOT / "reports/v4r1_integration_validation.json").read_text(encoding="utf-8"))
    if gate["updates"] != 13 or gate["reload_max_error"] != 0 or gate["final_used"]:
        raise RuntimeError("Real model validation gate failed")
    registry = json.loads((ROOT / "configs/segformer/v4r1/experiments.json").read_text(encoding="utf-8"))
    batch_dir = ROOT / "outputs/segformer/v4r1_batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    status_file = batch_dir / "status.json"
    status = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "final_evaluated": False,
              "device": "cuda:1", "experiments": [], "state": "running"}
    for item in registry:
        cfg = load_segformer_config(ROOT / item["config"])
        s = cfg["segformer"]
        run_dir = ROOT / s["output_dir"] / s["data"]["version"] / s["experiment"] / str(s["seed"])
        entry = dict(item, state="training", config_sha256=file_sha256(ROOT / item["config"]))
        status["experiments"].append(entry)
        status_file.write_text(json.dumps(status, indent=1), encoding="utf-8")
        print(f"[START] {item['experiment']}", flush=True)
        # An interrupted batch can continue only through the trainer's exact resume.
        resume = (run_dir / "last.pt").is_file()
        with (batch_dir / f"{item['experiment']}.log").open("a", encoding="utf-8", buffering=1) as log_file:
            existing = run_dir / "train_summary.json"
            previous = json.loads(existing.read_text(encoding="utf-8")) if existing.is_file() else {}
            if previous.get("complete") and previous.get("global_step") == 3600:
                if json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8")) != cfg:
                    raise ValueError("Completed run configuration differs; choose a new experiment")
                summary = previous  # Preserve measured training cost and peak memory on re-evaluation.
                print("[reuse] completed training; re-evaluate saved checkpoints", file=log_file, flush=True)
            else:
                summary = train_one(cfg, device="cuda:1", resume=resume, log=lambda x: print(x, file=log_file, flush=True))
        if not summary["complete"] or summary["global_step"] != 3600 or summary["samples_exposed"] != 3600 * 64:
            raise RuntimeError("Unexpected training budget")
        entry.update(state="evaluating", best_step=summary["best_step"], best_dev_mean=summary["best_val_iou"])
        status_file.write_text(json.dumps(status, indent=1), encoding="utf-8")
        gc.collect()
        torch.cuda.empty_cache()
        for checkpoint in ("best", "last"):
            for split in ("val", "train"):
                tag = f"{checkpoint}_{split}"
                argv = ["evaluate_segformer.py", "--checkpoint", str(run_dir / f"{checkpoint}.pt"),
                        "--split", split, "--device", "cuda:1", "--tag", tag,
                        "--role", "DEV" if split == "val" else "TRAIN"]
                if split == "val" and checkpoint == "best":
                    argv += ["--save-predictions", "--figures", "4"]
                else:
                    argv += ["--skip-efficiency"]
                saved_argv = sys.argv
                try:
                    sys.argv = argv
                    with (batch_dir / f"{item['experiment']}_{tag}.log").open("w", encoding="utf-8") as log_file:
                        with contextlib.redirect_stdout(log_file):
                            evaluate_segformer.main()
                finally:
                    sys.argv = saved_argv
                gc.collect()
                torch.cuda.empty_cache()
        # Evaluate identical reference tiles for curve comparison; unseen reference
        # members in smaller pools are explicitly not called training fit.
        saved_argv = sys.argv
        try:
            sys.argv = ["evaluate_segformer.py", "--checkpoint", str(run_dir / "best.pt"), "--split", "train",
                        "--manifest", "manifests/v4r1/pool_100.jsonl", "--device", "cuda:1", "--tag", "best_common_train_reference",
                        "--role", "COMMON_TRAIN_REFERENCE_includes_unseen_tiles_for_smaller_pools", "--skip-efficiency"]
            with (batch_dir / f"{item['experiment']}_reference.log").open("w", encoding="utf-8") as log_file:
                with contextlib.redirect_stdout(log_file):
                    evaluate_segformer.main()
        finally:
            sys.argv = saved_argv
        entry.update(state="complete", training_seconds=summary["seconds"], run_dir=str(run_dir))
        status_file.write_text(json.dumps(status, indent=1), encoding="utf-8")
        print(f"[COMPLETE] {item['experiment']} dev_mean={summary['best_val_iou']:.5f} best_step={summary['best_step']}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    status.update(state="complete", ended=time.strftime("%Y-%m-%d %H:%M:%S"))
    status_file.write_text(json.dumps(status, indent=1), encoding="utf-8")
    print("[ALL COMPLETE] six experiments; FINAL was never evaluated", flush=True)


if __name__ == "__main__":
    main()
