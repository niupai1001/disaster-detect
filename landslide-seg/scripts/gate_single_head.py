"""P2 门槛：单头 N1/N2 的工程检查。

- 共享初始张量一致、N2 初始 W=0.5 且输出与 N1 严格一致；
- 真实样本 5 步更新：损失有限、各模块有梯度、AFF 在若干步后获得梯度；
- 空标注、全无效 batch、batch1 eval、保存重载、优化器/调度器恢复；
- 记录初始与 5 步后的 W 统计（均值/标准差/饱和比例）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch  # noqa: E402

from landslide_seg.config import load_segformer_config, resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import (  # noqa: E402
    S2SegDataset,
    collate,
    filter_split,
    load_stats,
    read_manifest,
)
from landslide_seg.models.factory import build_model  # noqa: E402
from landslide_seg.training.segformer_trainer import (  # noqa: E402
    _loss,
    _make_optimizer,
    _make_scheduler,
    _train_epoch,
    set_seed,
)

CFG_DIR = ROOT / "configs" / "segformer" / "single_head"


def _wstats(w: torch.Tensor) -> dict:
    x = w.detach().float()
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "sat_lo": float((x < 0.05).float().mean()),
        "sat_hi": float((x > 0.95).float().mean()),
    }


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfgs = {n: load_segformer_config(CFG_DIR / f"{n}.yaml") for n in ("n1", "n2")}
    pretrained = resolve_path(cfgs["n1"]["segformer"]["model"]["pretrained_dir"])
    set_seed(1234, True)
    m1 = build_model(cfgs["n1"]["segformer"], pretrained_dir=pretrained).to(device).eval()
    set_seed(1234, True)
    m2 = build_model(cfgs["n2"]["segformer"], pretrained_dir=pretrained).to(device).eval()

    data_cfg = cfgs["n1"]["segformer"]
    stats = load_stats(resolve_path(data_cfg["data"]["stats"]))
    recs = filter_split(read_manifest(resolve_path(data_cfg["data"]["manifest"])), "train")[:8]
    ds = S2SegDataset(recs, stats, preload=True)
    batch = collate([ds[i] for i in range(4)])
    x = batch["image"].to(device)
    report = {}

    with torch.no_grad():
        o1 = m1(x)["logits"]
        o2 = m2(x, return_attn=True)["logits"]
    report["init_equivalence"] = {
        "max_abs_diff": float((o1 - o2).abs().max()),
        "bitwise_equal": bool(torch.equal(o1, o2)),
    }
    with torch.no_grad():
        w = m2(x, return_attn=True)["attn"]
    report["w_init"] = _wstats(w)

    # 5 步更新（N2）
    m2.train()
    opt = _make_optimizer(m2, cfgs["n2"])
    sched = _make_scheduler(opt, cfgs["n2"], total_steps=3600)
    mask, valid = batch["mask"].to(device), batch["valid"].to(device)
    losses = []
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        out = m2(x)
        loss, _parts = _loss(cfgs["n2"], out["logits"], mask, valid, out["base_logits"])
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
    grads = {n: float(p.grad.abs().sum()) for n, p in m2.named_parameters() if p.grad is not None}
    aff = {k: v for k, v in grads.items() if "aff." in k}
    report["train_5_steps"] = {
        "losses": [round(v, 5) for v in losses],
        "finite": all(torch.isfinite(torch.tensor(losses))),
        "nonzero_grad_params": sum(1 for v in grads.values() if v > 0),
        "total_params": len(grads),
        "aff_grads_nonzero": sum(1 for v in aff.values() if v > 0),
        "aff_params": len(aff),
    }
    with torch.no_grad():
        w2 = m2(x, return_attn=True)["attn"]
    report["w_after_5_steps"] = _wstats(w2)
    report["w_changed"] = float((w2 - w).abs().max())

    # 空标注
    m2.eval()
    with torch.no_grad():
        out = m2(x)
        empty = torch.zeros_like(mask)
        loss_e, parts = _loss(cfgs["n2"], out["logits"], empty, valid, out["base_logits"])
    report["empty_gt"] = {"loss": round(float(loss_e), 5), "finite": bool(torch.isfinite(loss_e))}

    # 全无效 batch 跳过
    m2.train()
    bad = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    bad["valid"] = torch.zeros_like(valid)
    mean_loss, skipped, steps, _ = _train_epoch(m2, [bad, batch], opt, sched, cfgs["n2"], device)
    report["all_invalid_batch"] = {
        "skipped": skipped, "optimizer_steps": steps, "loss_finite": bool(torch.isfinite(torch.tensor(mean_loss)))
    }

    # batch1 eval
    m2.eval()
    with torch.no_grad():
        one = m2(x[:1])["logits"]
    report["batch1_eval"] = {"shape": list(one.shape), "finite": bool(torch.isfinite(one).all())}

    # 重载
    state = {k: v.clone() for k, v in m2.state_dict().items()}
    set_seed(999, True)
    fresh = build_model(cfgs["n2"]["segformer"], pretrained_dir=pretrained).to(device).eval()
    fresh.load_state_dict(state, strict=True)
    with torch.no_grad():
        a = m2(x)["logits"]
        b = fresh(x)["logits"]
    report["reload"] = {"max_abs_diff": float((a - b).abs().max())}

    # 恢复
    lr_before = opt.param_groups[0]["lr"]
    ep_before = sched.last_epoch
    st = {"opt": opt.state_dict(), "sched": sched.state_dict()}
    opt2 = _make_optimizer(m2, cfgs["n2"])
    sched2 = _make_scheduler(opt2, cfgs["n2"], total_steps=3600)
    opt2.load_state_dict(st["opt"])
    sched2.load_state_dict(st["sched"])
    report["resume"] = {
        "lr_ok": abs(lr_before - opt2.param_groups[0]["lr"]) < 1e-12,
        "epoch_ok": ep_before == sched2.last_epoch,
        "lr": lr_before,
        "epoch": ep_before,
    }
    report["params"] = {"n1": m1.count_params(), "n2": m2.count_params()}

    out_dir = resolve_path("outputs/segformer/single_head")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gates.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"[saved] {out_dir / 'gates.json'}")


if __name__ == "__main__":
    main()
