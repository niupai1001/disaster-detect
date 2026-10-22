"""Update-budget and restart integration tests with a real small torch model."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
from torch import nn

from landslide_seg.config import load_segformer_config
from landslide_seg.training import segformer_trainer as trainer
from landslide_seg.training.update_training import FullBatchStream, evaluate_events
from landslide_seg.training.segformer_metrics import pixel_average_precision, evaluate_tile_v2, summarize_tiles_v2
from landslide_seg.data.segformer_dataset import S2SegDataset


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(10, 1, 1)
        self.dropout = nn.Dropout2d(0.2)
        self.adapter_report, self.load_info = {}, {}

    def forward(self, image):
        return {"logits": self.conv(self.dropout(image))}

    def param_groups(self):
        return {"encoder": list(self.conv.parameters()), "decoder": [], "new": []}

    def count_params(self):
        return {"total": 11, "trainable": 11}


@pytest.fixture
def setup_run(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(trainer, "build_model", lambda *a, **kw: Tiny())
    monkeypatch.setattr(trainer, "model_type", lambda *a: "unet")
    monkeypatch.setattr(trainer, "model_identity", lambda *a: {"toy": True})
    rows = []
    for i in range(5):
        path = tmp_path / f"{i}.npz"
        x = np.arange(640, dtype=np.int16).reshape(10, 8, 8) + i
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:5, 2:5] = 1
        np.savez(path, x10=x, mask=mask, valid=np.ones_like(mask, dtype=bool))
        rows.append({"sample_id": str(i), "event_id": "train" if i < 3 else ("hiroshima" if i == 3 else "thrissur"),
                     "split": "train" if i < 3 else "val", "cache": str(path)})
    manifest = tmp_path / "data.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"mean": [0] * 10, "std": [1000] * 10}))
    cfg = load_segformer_config()
    cfg["segformer"].update(experiment="toy", output_dir=str(tmp_path / "runs"), seed=1234)
    cfg["segformer"]["data"].update(manifest=str(manifest), stats=str(stats), version="toy", preload=True)
    cfg["segformer"]["train"].update(budget_mode="updates", max_steps=7, epochs=1, batch_size=4, eval_every_updates=3)
    return cfg, rows, json.loads(stats.read_text())


@pytest.mark.parametrize("target", [1, 7, 11])
def test_exact_budget_ignores_epoch_cap(setup_run, target):
    cfg, _, _ = setup_run
    cfg["segformer"]["train"]["max_steps"] = target
    summary = trainer.train_one(cfg, device="cpu", log=lambda *a: None)
    assert summary["global_step"] == target
    assert summary["samples_exposed"] == target * 4
    assert summary["n_val"] == 2
    assert set(summary["history"][0]["dev"]["by_event"]) == {"hiroshima", "thrissur"}
    assert [r["global_step"] for r in summary["history"]] == list(range(3, target, 3)) + [target]


def test_restart_exact_weights_rng_cursor_and_scheduler(setup_run):
    cfg, _, _ = setup_run
    cfg["segformer"]["experiment"] = "continuous"
    uninterrupted = trainer.train_one(cfg, device="cpu", log=lambda *a: None)
    a = torch.load(uninterrupted["last_checkpoint"], weights_only=False)
    cfg["segformer"]["experiment"] = "resumed"
    cfg["segformer"]["train"]["pause_after_updates"] = 3
    partial = trainer.train_one(cfg, device="cpu", log=lambda *a: None)
    assert partial["global_step"] == 3 and not partial["complete"]
    resumed = trainer.train_one(cfg, device="cpu", resume=True, log=lambda *a: None)
    b = torch.load(resumed["last_checkpoint"], weights_only=False)
    assert resumed["global_step"] == 7
    assert a["stream_state"] == b["stream_state"]
    assert a["scheduler"] == b["scheduler"]
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    assert len(resumed["history"]) == 3


def test_empty_valid_batches_are_not_optimizer_steps(setup_run):
    cfg, rows, _ = setup_run
    for r in rows[:3]:
        with np.load(r["cache"]) as d:
            x, mask = d["x10"].copy(), d["mask"].copy()
        np.savez(r["cache"], x10=x, mask=mask, valid=np.zeros_like(mask, dtype=bool))
    with pytest.raises(RuntimeError, match="No valid optimizer"):
        trainer.train_one(cfg, device="cpu", log=lambda *a: None)


def test_run_overwrite_and_changed_resume_are_rejected(setup_run):
    cfg, _, _ = setup_run
    cfg["segformer"]["train"]["max_steps"] = 1
    trainer.train_one(cfg, device="cpu", log=lambda *a: None)
    with pytest.raises(FileExistsError):
        trainer.train_one(cfg, device="cpu", log=lambda *a: None)
    cfg["segformer"]["seed"] = 5678
    # Experiment directory is seed-dependent; restore seed and change budget.
    cfg["segformer"]["seed"] = 1234
    cfg["segformer"]["train"]["max_steps"] = 2
    with pytest.raises(ValueError, match="Resume"):
        trainer.train_one(cfg, device="cpu", resume=True, log=lambda *a: None)


def test_stream_crosses_epochs_and_restores(setup_run):
    _, rows, stats = setup_run
    ds = S2SegDataset(rows[:3], stats, augment=True)
    stream = FullBatchStream(ds, 4, 1234)
    assert len(next(stream)["sample_id"]) == 4
    restored = FullBatchStream(ds, 4, 1234, stream.state_dict())
    a, b = next(stream), next(restored)
    assert a["sample_id"] == b["sample_id"]
    assert torch.equal(a["image"], b["image"])


def test_ap_ties_ignore_and_undefined():
    assert pixel_average_precision([0.5, 0.5], [1, 0]) == pytest.approx(0.5)
    assert pixel_average_precision([0.9, 0.1], [1, 0]) == 1.0
    assert pixel_average_precision([0.9, 0.1], [0, 1]) == 0.5
    assert pixel_average_precision([0.5], [0]) is None
    assert pixel_average_precision([0.8, 0.6, 0.4, 0.2], [1, 0, 1, 0]) == pytest.approx((1 + 2 / 3) / 2)


def test_fine_object_bins_partition():
    gt = np.zeros((128, 128), dtype=bool)
    for x, area in zip((5, 20, 35, 50, 65, 85), (4, 9, 19, 49, 199, 200)):
        # Place each vertical strip far apart; all objects are interior.
        w = 2 if area < 100 else 4
        coords = [(y, xx) for y in range(5, 65) for xx in range(x, x + w)][:area]
        for y, xx in coords:
            gt[y, xx] = True
    row = evaluate_tile_v2(gt.astype(np.float32), gt, np.ones_like(gt))
    result = summarize_tiles_v2([row])["objects"]["fine_bins"]
    assert all(v["n_objects"] == 1 for v in result.values())
    assert all(v["object_recall_evaluable"] == 1 for v in result.values())
