"""One declared pretrained DOFA system run using the existing exact-update trainer."""
import contextlib
import gc
import json
import sys
from pathlib import Path
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
torch.set_num_threads(4)
from landslide_seg.config import load_segformer_config, file_sha256
from landslide_seg.training.segformer_trainer import train_one
import evaluate_segformer

CONFIG = ROOT / "configs/segformer/v4r3/dofa_base_upernet_pyramid_binary_n100_u3600_s1234.yaml"
BATCH = ROOT / "outputs/segformer/v4r3_dofa_batch"


if __name__ == "__main__":
    gate = json.loads((ROOT / "reports/v4r3_dofa_interface_validation.json").read_text(encoding="utf-8"))
    assert gate["status"] == "PASS" and gate["batch64"]["passed"] and gate["reload_max_error"] == 0
    assert gate["config_sha256"] == file_sha256(CONFIG)
    for path, digest in gate["source_sha256"].items():
        assert file_sha256(ROOT / path) == digest
    cfg = load_segformer_config(CONFIG)
    s = cfg["segformer"]
    BATCH.mkdir(exist_ok=True)
    snapshot = BATCH / "training_sources.zip"
    if not snapshot.exists():
        with zipfile.ZipFile(snapshot, "x", zipfile.ZIP_DEFLATED) as archive:
            for path in gate["source_sha256"]:
                archive.write(ROOT / path, path)
            for path in s["model"]["code_sha256"]:
                archive.write(ROOT / path, path)
            for path in (CONFIG, Path(__file__), ROOT / "scripts/evaluate_segformer.py"):
                archive.write(path, path.relative_to(ROOT).as_posix())
    run = ROOT / s["output_dir"] / s["data"]["version"] / s["experiment"] / "1234"
    status = {"state": "training", "run_dir": str(run), "final_evaluated": False, "fresh_pretrained_initialization": True}
    status_path = BATCH / "status.json"
    status_path.write_text(json.dumps(status, indent=1), encoding="utf-8")
    with (BATCH / "train.log").open("a", encoding="utf-8", buffering=1) as log:
        previous = json.loads((run / "train_summary.json").read_text(encoding="utf-8")) if (run / "train_summary.json").exists() else {}
        if previous.get("complete"):
            assert json.loads((run / "resolved_config.json").read_text(encoding="utf-8")) == cfg
            summary = previous
        else:
            summary = train_one(cfg, device="cuda:1", resume=(run / "last.pt").exists(), log=lambda x: print(x, file=log, flush=True))
    assert summary["global_step"] == 3600 and summary["samples_exposed"] == 230400
    assert len(summary["history"]) == 36 and summary["skipped_batches"] == 0
    print(f"[TRAIN COMPLETE] DEV={summary['best_val_iou']}@{summary['best_step']}", flush=True)
    status.update(state="evaluating", best_step=summary["best_step"], best_dev_mean=summary["best_val_iou"])
    status_path.write_text(json.dumps(status, indent=1), encoding="utf-8")
    gc.collect()
    torch.cuda.empty_cache()
    for checkpoint in ("best", "last"):
        for split in ("train", "val"):
            tag = f"{checkpoint}_{split}"
            args = ["evaluate_segformer.py", "--checkpoint", str(run / f"{checkpoint}.pt"), "--split", split,
                    "--tag", tag, "--device", "cuda:1", "--role", "TRAIN" if split == "train" else "DEV"]
            args += ["--save-predictions", "--figures", "4"] if tag == "best_val" else ["--skip-efficiency"]
            saved = sys.argv
            try:
                sys.argv = args
                with (BATCH / f"{tag}.log").open("w", encoding="utf-8") as log:
                    with contextlib.redirect_stdout(log):
                        evaluate_segformer.main()
            finally:
                sys.argv = saved
            gc.collect()
            torch.cuda.empty_cache()
    status.update(state="complete", summary=summary, config_sha256=file_sha256(CONFIG))
    status_path.write_text(json.dumps(status, indent=1), encoding="utf-8")
    print("[ALL COMPLETE] one DOFA full-data run, TRAIN/DEV only", flush=True)
