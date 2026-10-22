"""Actual pretrained dense DOFA: real data gradients, reload, batch and overfit."""
from __future__ import annotations
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
from landslide_seg.config import load_segformer_config, resolve_path, file_sha256, set_seed
from landslide_seg.data.segformer_dataset import S2SegDataset, read_manifest, load_stats, collate
from landslide_seg.models.factory import build_model, model_identity
from landslide_seg.training.segformer_trainer import _make_optimizer, _make_scheduler, _loss, _amp_context, _save_checkpoint, load_checkpoint
from landslide_seg.training.protocol import validate_protocol
from landslide_seg.training.metrics import confusion, summarize

CONFIG = ROOT / "configs/segformer/v4r3/dofa_base_upernet_pyramid_binary_n100_u3600_s1234.yaml"
DEVICE = torch.device("cuda:0")


@torch.no_grad()
def score(model, batch):
    model.eval()
    with _amp_context(CFG, DEVICE):
        probability = model(batch["image"].to(DEVICE))["logits"].float()[:, 0].sigmoid().cpu().numpy()
    return summarize([confusion(probability[i] >= .5, batch["mask"][i].numpy().astype(bool), batch["valid"][i].numpy().astype(bool)) for i in range(len(probability))])


if __name__ == "__main__":
    CFG = load_segformer_config(CONFIG)
    seed_info = set_seed(1234, True)
    rows = read_manifest(resolve_path(CFG["segformer"]["data"]["manifest"]))
    stats = load_stats(resolve_path(CFG["segformer"]["data"]["stats"]))
    protocol = validate_protocol(CFG, rows, stats)
    train = [r for r in rows if r["split"] == "train"]
    ds = S2SegDataset(train[:64], stats, augment=False, preload=True)
    small = collate([ds[i] for i in range(8)])
    print("[BUILD] official DOFA + MMSeg neck + Transformers UPerNet", flush=True)
    model = build_model(CFG["segformer"]).to(DEVICE)
    optimizer = _make_optimizer(model, CFG)
    scheduler = _make_scheduler(optimizer, CFG, 100)
    initial = score(model, small)
    history, gradients = [], {}
    started = time.perf_counter()
    for step in range(100):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(CFG, DEVICE):
            output = model(small["image"].to(DEVICE), return_features=step == 0)
        loss, _ = _loss(CFG, output["logits"], small["mask"].to(DEVICE), small["valid"].to(DEVICE))
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if step == 0:
            gradients = {name: {"parameters": sum(p.numel() for p in group),
                                "with_grad": sum(p.numel() for p in group if p.grad is not None),
                                "abs_grad_sum": sum(float(p.grad.abs().sum()) for p in group if p.grad is not None)}
                         for name, group in model.param_groups().items()}
            assert all(item["abs_grad_sum"] > 0 for item in gradients.values())
            feature_shapes = [list(x.shape) for x in output["encoder_features"]]
            pyramid_shapes = [list(x.shape) for x in output["pyramid_features"]]
            assert output["logits"].shape == (8, 1, 128, 128)
            print("[GRAD PASS] " + json.dumps(gradients), flush=True)
        optimizer.step()
        scheduler.step()
        if (step+1) % 10 == 0:
            metrics = score(model, small)
            history.append({"step": step+1, "loss": float(loss.detach()), "train_iou": metrics["iou"]})
            print(json.dumps(history[-1]), flush=True)
        del output, loss
    model.eval()
    with torch.no_grad(), _amp_context(CFG, DEVICE):
        before = model(small["image"].to(DEVICE))["logits"].float().clone()
    run = ROOT / "outputs/segformer/v4r3_dofa_interface"
    run.mkdir(exist_ok=True)
    checkpoint = run / "overfit_8_tiles_u100.pt"
    meta = {"stats_sha256": file_sha256(resolve_path(CFG["segformer"]["data"]["stats"])),
            "model_identity": model_identity(CFG["segformer"], model), "protocol": protocol}
    _save_checkpoint(checkpoint, model, optimizer, scheduler, 0, history[-1]["train_iou"], 0,
                     CFG, meta, seed_info, extra={"purpose": "interface/overfit only; not a DEV performance baseline", "global_step": 100})
    reloaded, _ = load_checkpoint(checkpoint, device=str(DEVICE))
    with torch.no_grad(), _amp_context(CFG, DEVICE):
        after = reloaded(small["image"].to(DEVICE))["logits"].float()
    error = float((before-after).abs().max())
    assert error == 0
    del reloaded, after, before
    gc.collect()
    torch.cuda.empty_cache()
    optimizer.zero_grad(set_to_none=True)
    model.train()
    batch = collate([ds[i] for i in range(64)])
    torch.cuda.reset_peak_memory_stats(DEVICE)
    try:
        with _amp_context(CFG, DEVICE):
            output = model(batch["image"].to(DEVICE))
        loss, _ = _loss(CFG, output["logits"], batch["mask"].to(DEVICE), batch["valid"].to(DEVICE))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        batch64 = {"passed": True, "peak_alloc_mib": torch.cuda.max_memory_allocated(DEVICE)/2**20}
    except torch.cuda.OutOfMemoryError as exc:
        batch64 = {"passed": False, "error": str(exc)}
    report = {"status": "PASS", "purpose": "interface + 8 TRAIN tiles overfit; not a generalization result",
              "final_used": False, "n_train_tiles": 8, "sample_ids": [r["sample_id"] for r in train[:8]],
              "config_sha256": file_sha256(CONFIG), "device": str(DEVICE), "precision": "bf16 autocast, fp32 weights/optimizer/loss",
              "initial_train": initial, "history": history, "gradients": gradients, "params": model.count_params(),
              "feature_shapes": feature_shapes, "pyramid_shapes": pyramid_shapes, "output_shape": [8,1,128,128],
              "source_sha256": {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in sorted((ROOT / "src/landslide_seg").rglob("*.py"))},
              "reload_max_error": error, "batch64": batch64, "seconds": time.perf_counter()-started,
              "weight_loading": model.load_info, "input": model.adapter_report, "checkpoint_sha256": file_sha256(checkpoint)}
    (ROOT / "reports/v4r3_dofa_interface_validation.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("[PASS] actual DOFA pretrained interface, overfit and reload; batch64=" + str(batch64["passed"]), flush=True)
