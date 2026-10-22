"""Real SegFormer integration smoke on TRAIN/DEV only; never evaluates FINAL."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
torch.set_num_threads(4)
from landslide_seg.config import load_segformer_config, resolve_path, file_sha256
from landslide_seg.data.segformer_dataset import read_manifest, load_stats, S2SegDataset, collate
from landslide_seg.training.protocol import validate_protocol
from landslide_seg.training.segformer_trainer import train_one, load_checkpoint, _loss, _make_optimizer


def main():
    cfg = load_segformer_config(ROOT / "configs/segformer/v4r1/segformer_b0_direct10_detailres_n025_u3600_s1234.yaml")
    cfg["segformer"]["experiment"] = "validation_direct10_detailres_u13"
    cfg["segformer"]["train"].update(max_steps=13, batch_size=4, eval_every_updates=5)
    rows = read_manifest(resolve_path(cfg["segformer"]["data"]["manifest"]))
    stats = load_stats(resolve_path(cfg["segformer"]["data"]["stats"]))
    contract = validate_protocol(cfg, rows, stats)
    bad = json.loads(json.dumps(stats))
    bad["fit_sample_ids"].append(next(r["sample_id"] for r in rows if r["split"] == "val"))
    try:
        validate_protocol(cfg, rows, bad)
        raise AssertionError("Contaminated stats accepted")
    except ValueError:
        pass
    summary = train_one(cfg, device="cuda:1")
    assert summary["global_step"] == 13
    assert [r["global_step"] for r in summary["history"]] == [5, 10, 13]
    model, ckpt = load_checkpoint(summary["best_checkpoint"], device="cuda:1")
    ds = S2SegDataset([r for r in rows if r["split"] == "train"][:4], stats, preload=True)
    batch = collate([ds[i] for i in range(4)])
    image = batch["image"].to("cuda:1")
    model.eval()
    with torch.no_grad():
        before = model(image)["logits"].clone()
    assert before.shape == (4, 1, 128, 128) and torch.isfinite(before).all()
    reloaded, _ = load_checkpoint(summary["best_checkpoint"], device="cuda:1")
    with torch.no_grad():
        after = reloaded(image)["logits"]
    error = float((before - after).abs().max())
    assert error == 0.0
    del reloaded
    optimizer = _make_optimizer(model, cfg)
    losses = []
    model.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        out = model(image)
        loss, _ = _loss(cfg, out["logits"], batch["mask"].to("cuda:1"), batch["valid"].to("cuda:1"))
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        losses.append(float(loss.detach()))
        optimizer.step()
    gradients = {}
    for group, params in model.param_groups().items():
        gradients[group] = {"parameters": sum(p.numel() for p in params),
                            "with_grad": sum(p.numel() for p in params if p.grad is not None),
                            "abs_grad_sum": sum(float(p.grad.abs().sum()) for p in params if p.grad is not None)}
    for name, module in (("stem", model.stem), ("fusion", model.fusion)):
        gradients[name] = {"abs_grad_sum": sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)}
    assert all(v["abs_grad_sum"] > 0 for v in gradients.values())
    report = {"purpose": "integration only; not a performance baseline", "final_used": False,
              "contract": contract, "updates": 13, "dev_checks": [5, 10, 13],
              "output_shape": list(before.shape), "reload_max_error": error,
              "gradients": gradients, "three_step_probe_losses": losses,
              "checkpoint_sha256": file_sha256(summary["best_checkpoint"]), "summary": summary}
    dest = ROOT / "reports/v4r1_integration_validation.json"
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[PASS] real SegFormer gradient/reload/update validation: {dest}")


if __name__ == "__main__":
    main()
