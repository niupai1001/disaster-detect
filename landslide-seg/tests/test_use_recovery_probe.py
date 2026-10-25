import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location(
    "run_use_recovery_probe",
    Path(__file__).resolve().parents[1] / "scripts/run_use_recovery_probe.py",
)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def test_index_stream_state_and_resume():
    first = recovery.IndexStream(10, 3, 1234)
    a = first.next()
    b = first.next()
    state = first.state_dict()
    second = recovery.IndexStream(10, 3, 1234, state)
    c = second.next()
    d = second.next()
    assert c == first.next()
    assert d == first.next()
    assert state["n"] == 10 and state["batch_size"] == 3
    assert len(a) == len(b) == 3
    assert sum(state["exposures"]) == 6


def test_full_support_mask_requires_all_pixels():
    valid = torch.ones(2, 16, 16, dtype=torch.bool)
    mask = recovery.full_support_mask(valid, 4)
    assert mask.shape == (2, 4, 4)
    assert bool(mask.all())
    valid[0, 0:4, 0:4] = False
    mask = recovery.full_support_mask(valid, 4)
    assert not bool(mask[0, 0, 0])
    assert bool(mask[1].all())


def test_feature_loss_zero_for_identical_and_empty_is_differentiable():
    torch.manual_seed(0)
    student = torch.randn(1, 4, 8, 8, requires_grad=True)
    teacher = student.detach().clone()
    norm = {"stages": {"mit.stage1": {"mean": [0.0] * 4, "std": [1.0] * 4}}}
    valid = torch.ones(1, 32, 32, dtype=torch.bool)
    loss = recovery.feature_loss(
        {"mit.stage1": student}, {"mit.stage1": teacher}, valid, norm, ("mit.stage1",)
    )
    assert abs(float(loss.detach())) < 1e-8

    empty = torch.zeros(1, 32, 32, dtype=torch.bool)
    loss_empty = recovery.feature_loss(
        {"mit.stage1": student}, {"mit.stage1": teacher}, empty, norm, ("mit.stage1",)
    )
    assert float(loss_empty.detach()) == 0.0
    assert loss_empty.requires_grad
    loss_empty.backward()
    assert student.grad is not None


def test_save_checkpoint_roundtrip(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    stream = recovery.IndexStream(4, 2, 1234)
    norm = {"stages": {"mit.stage1": {"mean": [0.0, 0.0], "std": [1.0, 1.0]}}}
    path = tmp_path / "checkpoint.pt"
    recovery.save_checkpoint(path, model, optimizer, scheduler, stream, 2, {"x": 1}, norm)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["step"] == 2
    assert payload["stream"]["n"] == 4
    assert set(payload["model"]) == set(model.state_dict())
    for key in model.state_dict():
        assert torch.equal(payload["model"][key], model.state_dict()[key])


def test_recovery_configs_are_isolated_and_preregistered():
    g1 = recovery.load_segformer_config(recovery.CONFIG_DIR / "group1.yaml")
    g2 = recovery.load_segformer_config(recovery.CONFIG_DIR / "group2.yaml")
    assert g1["segformer"]["experiment"] == "use_recovery_g1"
    assert g2["segformer"]["experiment"] == "use_recovery_g2"
    assert g1["segformer"]["recovery"]["feature_assist"] is False
    assert g2["segformer"]["recovery"]["feature_assist"] is True
    assert g2["segformer"]["recovery"]["feature"]["coefficient"] == 0.05
    assert g1["segformer"]["recovery"]["calibration"]["members"] == 512
    assert g1["segformer"]["data"]["mixed_bands"] is True
    teacher = recovery.resolve_path(g1["segformer"]["recovery"]["teacher"])
    assert recovery.sha(teacher) == g1["segformer"]["recovery"]["teacher_sha256"]


def test_formal_entry_refuses_without_execute():
    cfg = recovery.load_segformer_config(recovery.CONFIG_DIR / "group1.yaml")
    result = recovery.supervised(cfg, "g1", short=False, execute_formal=False)
    assert result["status"] == "NOT_RUN"
    assert "formal" in result["reason"].lower() or "not authorised" in result["reason"].lower()


def test_calibration_scheduler_warmup_and_end():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = recovery._calibration_scheduler(optimizer, steps=10, warmup=2)
    lrs = []
    for _ in range(10):
        lrs.append(float(optimizer.param_groups[0]["lr"]))
        optimizer.step()
        scheduler.step()
    assert lrs[0] < lrs[1]
    assert 0.0 < lrs[-1] < 0.1
