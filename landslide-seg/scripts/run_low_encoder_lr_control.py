"""One predeclared LR control; existing model/data/trainer/evaluator reused."""
from __future__ import annotations
import argparse
import contextlib
import copy
import gc
import json
import sys
from pathlib import Path
import zipfile
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
torch.set_num_threads(4)
from landslide_seg.config import load_segformer_config, file_sha256, resolve_path, set_seed
from landslide_seg.models.factory import build_model
from landslide_seg.training.segformer_trainer import train_one, load_checkpoint, _make_optimizer
from landslide_seg.data.segformer_dataset import read_manifest, load_stats, S2SegDataset, collate
import evaluate_segformer

EXPERIMENT = "segformer_b0_direct10_detailres_pretrained_encoder_low_lr_n100_u3600_s1234"
CONFIG = ROOT / f"configs/segformer/v4r3/{EXPERIMENT}.yaml"
BASE = ROOT / "configs/segformer/v4r1/segformer_b0_direct10_detailres_n100_u3600_s1234.yaml"
BATCH = ROOT / "outputs/segformer/v4r3_low_encoder_lr_batch"
GATE = ROOT / "reports/v4r3_low_encoder_lr_validation.json"


def prepare():
    cfg = load_segformer_config(BASE)
    s = cfg["segformer"]
    s["experiment"] = EXPERIMENT
    s["data"]["version"] = "s2_10band_v4r3_low_encoder_lr"
    s["optim"].update(encoder_lr=6e-6, first_conv_lr=6e-5)
    assert not s["model"].get("probe") and s["model"]["stem"]
    content = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
    CONFIG.parent.mkdir(exist_ok=True)
    if CONFIG.exists() and CONFIG.read_text(encoding="utf-8") != content:
        raise FileExistsError("Refuse changed control config")
    CONFIG.write_text(content, encoding="utf-8")
    return cfg


def validate(cfg, device):
    set_seed(1234, True)
    model = build_model(cfg["segformer"])
    optimizer = _make_optimizer(model, cfg)
    assert model.count_params()["total"] == model.count_params()["trainable"] == 3753602
    assert [g["lr"] for g in optimizer.param_groups] == [6e-6, 6e-5, 6e-4]
    first_ids = {id(p) for p in model.model.segformer.encoder.patch_embeddings[0].proj.parameters()}
    assert {id(p) for p in optimizer.param_groups[1]["params"]} == first_ids
    grouped = [p for g in optimizer.param_groups for p in g["params"]]
    assert len({id(p) for p in grouped}) == len(grouped)
    assert {id(p) for p in grouped} == {id(p) for p in model.parameters()}
    groups = [{"initial_lr": g["lr"], "parameters": sum(p.numel() for p in g["params"])} for g in optimizer.param_groups]
    del model, optimizer
    tiny = copy.deepcopy(cfg)
    tiny["segformer"]["experiment"] = "validation_detailres_low_encoder_lr_u3"
    tiny["segformer"]["train"].update(max_steps=3, eval_every_updates=3, batch_size=4)
    summary = train_one(tiny, device=device)
    model, checkpoint = load_checkpoint(summary["last_checkpoint"], device=device)
    records = [r for r in read_manifest(resolve_path(cfg["segformer"]["data"]["manifest"])) if r["split"] == "train"][:4]
    ds = S2SegDataset(records, load_stats(resolve_path(cfg["segformer"]["data"]["stats"])), augment=False)
    batch = collate([ds[i] for i in range(4)])
    x = batch["image"].to(device)
    with torch.no_grad():
        before = model(x)["logits"].clone()
    again, _ = load_checkpoint(summary["last_checkpoint"], device=device)
    with torch.no_grad():
        error = float((before - again(x)["logits"]).abs().max())
    assert error == 0 and tuple(before.shape) == (4, 1, 128, 128)
    report = {"status": "PASS", "purpose": "integration only", "final_used": False, "updates": 3,
              "config_sha256": file_sha256(CONFIG), "optimizer_groups": groups,
              "params": summary["params"], "reload_max_error": error, "output_shape": list(before.shape),
              "source_sha256": checkpoint["meta"]["source_sha256"], "summary": summary}
    GATE.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("[PASS] actual optimizer groups, full training, shape and reload", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    BATCH.mkdir(exist_ok=True)
    cfg = prepare()
    if args.validate_only:
        validate(cfg, args.device)
        return
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    assert gate["status"] == "PASS" and gate["config_sha256"] == file_sha256(CONFIG)
    for relative, digest in gate["source_sha256"].items():
        assert file_sha256(ROOT / relative) == digest
    snapshot = BATCH / "training_sources.zip"
    if not snapshot.exists():
        with zipfile.ZipFile(snapshot, "x", zipfile.ZIP_DEFLATED) as archive:
            for relative in gate["source_sha256"]:
                archive.write(ROOT / relative, relative)
            for path in (CONFIG, Path(__file__), ROOT / "scripts/evaluate_segformer.py", GATE):
                archive.write(path, path.relative_to(ROOT).as_posix())
    s = cfg["segformer"]
    run = ROOT / s["output_dir"] / s["data"]["version"] / s["experiment"] / "1234"
    with (BATCH / "train.log").open("a", encoding="utf-8", buffering=1) as log:
        previous = json.loads((run / "train_summary.json").read_text(encoding="utf-8")) if (run / "train_summary.json").exists() else {}
        if previous.get("complete"):
            assert json.loads((run / "resolved_config.json").read_text(encoding="utf-8")) == cfg
            summary = previous
        else:
            summary = train_one(cfg, device=args.device, resume=(run / "last.pt").exists(), log=lambda x: print(x, file=log, flush=True))
    assert summary["global_step"] == 3600 and summary["samples_exposed"] == 230400
    assert summary["params"]["trainable"] == 3753602
    print(f"[TRAIN COMPLETE] dev={summary['best_val_iou']}@{summary['best_step']}", flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    for checkpoint in ("best", "last"):
        for split in ("train", "val"):
            tag = f"{checkpoint}_{split}"
            argv = ["evaluate_segformer.py", "--checkpoint", str(run / f"{checkpoint}.pt"), "--split", split,
                    "--device", args.device, "--tag", tag, "--role", "TRAIN" if split == "train" else "DEV"]
            argv += ["--save-predictions", "--figures", "4"] if tag == "best_val" else ["--skip-efficiency"]
            saved = sys.argv
            try:
                sys.argv = argv
                with (BATCH / f"{tag}.log").open("w", encoding="utf-8") as log:
                    with contextlib.redirect_stdout(log):
                        evaluate_segformer.main()
            finally:
                sys.argv = saved
            gc.collect()
            torch.cuda.empty_cache()
    (BATCH / "status.json").write_text(json.dumps({"state": "complete", "final_used": False,
        "run_dir": str(run), "summary": summary, "config_sha256": file_sha256(CONFIG)}, indent=1), encoding="utf-8")
    print("[ALL COMPLETE] one LR control; TRAIN/DEV only", flush=True)


if __name__ == "__main__":
    main()
