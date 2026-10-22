"""Validate and run the two fixed v4r2 decoder probes, TRAIN/DEV only."""
from __future__ import annotations
import argparse
import contextlib
import copy
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
from landslide_seg.config import load_segformer_config, file_sha256, resolve_path, set_seed
from landslide_seg.data.segformer_dataset import S2SegDataset, read_manifest, load_stats, collate
from landslide_seg.models.factory import build_model
from landslide_seg.models.segformer_probe import state_digest
from landslide_seg.training.segformer_trainer import train_one, load_checkpoint, _make_optimizer, _loss
import evaluate_segformer

BATCH = ROOT / "outputs/segformer/v4r2_probe_batch"
REGISTRY = ROOT / "configs/segformer/v4r2_probe/experiments.json"
GATE = ROOT / "reports/v4r2_probe_integration_validation.json"


def validate(device, experiment="validation_frozen_best_decodeprobe_u13"):
    cfg = load_segformer_config(ROOT / json.loads(REGISTRY.read_text(encoding="utf-8"))[0]["config"])
    cfg["segformer"]["experiment"] = experiment
    cfg["segformer"]["train"].update(max_steps=13, batch_size=4, eval_every_updates=5)
    summary = train_one(cfg, device=device)
    assert summary["global_step"] == 13 and summary["params"]["trainable"] == 395009
    assert all(r["probe_integrity"]["encoder_unchanged"] and r["probe_integrity"]["decoder_changed"] for r in summary["history"])
    model, ckpt = load_checkpoint(summary["best_checkpoint"], device=device)
    rows = [r for r in read_manifest(resolve_path(cfg["segformer"]["data"]["manifest"])) if r["split"] == "train"][:4]
    stats = load_stats(resolve_path(cfg["segformer"]["data"]["stats"]))
    ds = S2SegDataset(rows, stats, augment=False)
    batch = collate([ds[i] for i in range(4)])
    image = batch["image"].to(device)
    with torch.no_grad():
        before = model(image)["logits"].clone()
    reloaded, _ = load_checkpoint(summary["best_checkpoint"], device=device)
    with torch.no_grad():
        error = float((before - reloaded(image)["logits"]).abs().max())
    assert error == 0 and tuple(before.shape) == (4, 1, 128, 128)
    optimizer = _make_optimizer(model, cfg)
    model.train()
    assert not model.model.segformer.training and model.model.decode_head.training
    optimizer.zero_grad(set_to_none=True)
    logits = model(image)["logits"]
    loss, _ = _loss(cfg, logits, batch["mask"].to(device), batch["valid"].to(device))
    loss.backward()
    decoder = list(model.model.decode_head.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in decoder)
    assert all(p.grad is None for p in model.model.segformer.parameters())
    gradient = sum(float(p.grad.abs().sum()) for p in decoder)
    assert gradient > 0
    optimizer.step()
    integrity = model.probe_integrity()
    assert integrity["encoder_unchanged"] and integrity["decoder_changed"]
    report = {"status": "PASS", "purpose": "integration only, not a performance result", "final_evaluated": False,
              "updates": 13, "output_shape": list(before.shape), "reload_max_error": error,
              "decoder_abs_grad_sum": gradient, "integrity": integrity, "params": summary["params"],
              "registry_sha256": file_sha256(REGISTRY),
              "source_sha256": ckpt["meta"]["source_sha256"], "summary": summary}
    GATE.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("[PASS] actual frozen SegFormer forward/backward/train-mode/reload", flush=True)


def inspect_run(run_dir):
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    original = meta["model_identity"]["probe"]
    results = {}
    for tag in ("best", "last"):
        path = run_dir / f"{tag}.pt"
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        prefix = "model.segformer."
        enc = state_digest({k[len(prefix):]: v for k, v in ckpt["model"].items() if k.startswith(prefix)})
        prefix = "model.decode_head."
        dec = state_digest({k[len(prefix):]: v for k, v in ckpt["model"].items() if k.startswith(prefix)})
        assert enc == original["encoder_initial_sha256"] and dec != original["decoder_initial_sha256"]
        results[tag] = {"global_step": ckpt["global_step"], "checkpoint_sha256": file_sha256(path),
                        "encoder_unchanged": True, "encoder_sha256": enc, "decoder_changed": True,
                        "decoder_sha256": dec}
    (run_dir / "probe_integrity.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--validation-experiment", default="validation_frozen_best_decodeprobe_u13")
    args = parser.parse_args()
    BATCH.mkdir(exist_ok=True)
    if args.validate_only:
        validate(args.device, args.validation_experiment)
        return
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    assert gate["status"] == "PASS" and gate["registry_sha256"] == file_sha256(REGISTRY)
    for relative, digest in gate["source_sha256"].items():
        assert file_sha256(ROOT / relative) == digest, f"Source changed after validation: {relative}"
    status_path = BATCH / "status.json"
    status = {"state": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
              "device": args.device, "final_evaluated": False, "experiments": []}
    common_decoder = None
    for entry in json.loads(REGISTRY.read_text(encoding="utf-8")):
        cfg = load_segformer_config(ROOT / entry["config"])
        s = cfg["segformer"]
        run = ROOT / s["output_dir"] / s["data"]["version"] / s["experiment"] / str(s["seed"])
        current = dict(entry, state="training", run_dir=str(run))
        status["experiments"].append(current)
        status_path.write_text(json.dumps(status, indent=1), encoding="utf-8")
        print("[START] " + entry["experiment"], flush=True)
        with (BATCH / f"{entry['source']}_train.log").open("a", encoding="utf-8", buffering=1) as log:
            previous = json.loads((run / "train_summary.json").read_text(encoding="utf-8")) if (run / "train_summary.json").is_file() else {}
            if previous.get("complete"):
                assert json.loads((run / "resolved_config.json").read_text(encoding="utf-8")) == cfg
                summary = previous
            else:
                summary = train_one(cfg, device=args.device, resume=(run / "last.pt").is_file(),
                                    log=lambda x: print(x, file=log, flush=True))
        assert summary["global_step"] == 3600 and summary["samples_exposed"] == 230400
        assert len(summary["history"]) == 36 and summary["params"]["trainable"] == 395009
        assert all(r["probe_integrity"]["encoder_unchanged"] and r["probe_integrity"]["decoder_changed"]
                   and not r["probe_integrity"]["encoder_has_grad"] for r in summary["history"])
        original = inspect_run(run)
        if common_decoder is None:
            common_decoder = original["decoder_initial_sha256"]
        assert original["decoder_initial_sha256"] == common_decoder
        current.update(state="evaluating", best_step=summary["best_step"], best_dev_mean=summary["best_val_iou"])
        status_path.write_text(json.dumps(status, indent=1), encoding="utf-8")
        for checkpoint in ("best", "last"):
            for split in ("train", "val"):
                argv = ["evaluate_segformer.py", "--checkpoint", str(run / f"{checkpoint}.pt"),
                        "--split", split, "--device", args.device, "--tag", f"{checkpoint}_{split}",
                        "--role", "TRAIN" if split == "train" else "DEV"]
                if checkpoint == "best" and split == "val":
                    argv += ["--save-predictions", "--figures", "4"]
                else:
                    argv += ["--skip-efficiency"]
                saved = sys.argv
                try:
                    sys.argv = argv
                    with (BATCH / f"{entry['source']}_{checkpoint}_{split}.log").open("w", encoding="utf-8") as log:
                        with contextlib.redirect_stdout(log):
                            evaluate_segformer.main()
                finally:
                    sys.argv = saved
                gc.collect()
                torch.cuda.empty_cache()
        current.update(state="complete", training_seconds=summary["seconds"])
        status_path.write_text(json.dumps(status, indent=1), encoding="utf-8")
        print(f"[COMPLETE] {entry['source']} dev={summary['best_val_iou']:.5f}@{summary['best_step']}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    status.update(state="complete", ended=time.strftime("%Y-%m-%d %H:%M:%S"), common_decoder_initial_sha256=common_decoder)
    status_path.write_text(json.dumps(status, indent=1), encoding="utf-8")
    print("[ALL COMPLETE] two frozen encoder probes, TRAIN/DEV only", flush=True)


if __name__ == "__main__":
    main()
