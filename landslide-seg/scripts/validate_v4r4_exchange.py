"""v4r4 接口验证：正式训练前的七道门槛。

对应执行计划 §5。按 Self → Cross 顺序执行，全部通过才允许启动正式训练。
产物：
- ``reports/v4r4_interface_validation.json``：全部原始数值；
- ``reports/v4r4_interface_validation.md``：人读版，逐门 PASS/FAIL。

设计要点：
- 三架构（LateDetail / Self / Cross）用**同一 seed 构造**，但身份靠**哈希**确认，
  不靠"设了同一个 seed 就默认相同"——构造新增层会消耗随机数。
- 零残差系数会让 ``H_*`` 在第一步拿不到梯度，因此梯度检查用 20 步窗口，
  不断言首步非零。
- 诊断性干预（缩放 D、关闭 α）只在副本上做，不写入正式配置。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402
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
from landslide_seg.models.segformer_probe import state_digest  # noqa: E402
from landslide_seg.training.segformer_trainer import (  # noqa: E402
    _loss,
    _make_optimizer,
    _make_scheduler,
    load_checkpoint,
    set_seed,
)
from landslide_seg.training.metrics import confusion, summarize  # noqa: E402

CFG_SELF = "configs/segformer/v4r4/segformer_b0_detail_early_self_residual_n100_u3600_s1234.yaml"
CFG_CROSS = "configs/segformer/v4r4/segformer_b0_detail_early_cross_exchange_n100_u3600_s1234.yaml"
OUT_JSON = ROOT / "reports" / "v4r4_interface_validation.json"
OUT_MD = ROOT / "reports" / "v4r4_interface_validation.md"
CKPT_TMP = ROOT / "outputs" / "segformer" / "_v4r4_validation" / "ckpt_roundtrip.pt"
N_FIXED = 8
N_STEPS = 100


class Gate:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, gate: str, name: str, ok: bool | None, detail, *, hard: bool = True) -> None:
        self.rows.append({"gate": gate, "check": name, "ok": ok, "hard": hard, "detail": detail})
        flag = "PASS" if ok else ("FAIL" if ok is False else "INFO")
        print(f"[{flag}] {gate} · {name}: {detail}")

    def ok(self, gate: str | None = None) -> bool:
        return all(r["ok"] for r in self.rows if r["hard"] and (gate is None or r["gate"] == gate))


def _subset_hash(sd: dict, keys) -> str:
    return state_digest({k: sd[k] for k in keys})


def _forward_logits(model, x):
    model.eval()
    with torch.no_grad():
        return model(x)["logits"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=N_STEPS)
    args = ap.parse_args()

    dev = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    g = Gate()
    report: dict = {"device": str(dev), "configs": {"self": CFG_SELF, "cross": CFG_CROSS}}

    cfg_self = load_segformer_config(CFG_SELF)
    cfg_cross = load_segformer_config(CFG_CROSS)
    s = cfg_self["segformer"]
    seed = int(s["seed"])

    # ---------- 数据 ----------
    manifest = resolve_path(s["data"]["manifest"])
    stats = load_stats(resolve_path(s["data"]["stats"]))
    records = read_manifest(manifest)
    train_recs = filter_split(records, "train")
    if len(train_recs) < N_FIXED:
        raise RuntimeError("TRAIN 样本不足")
    ordered = sorted(train_recs, key=lambda r: r["sample_id"])
    picks = [ordered[int(round(i * (len(ordered) - 1) / (N_FIXED - 1)))] for i in range(N_FIXED)]
    picks = list({r["sample_id"]: r for r in picks}.values())[:N_FIXED]
    ds8 = S2SegDataset(picks, stats, augment=False, preload=True)
    batch8 = collate([ds8[i] for i in range(len(ds8))])
    x8 = batch8["image"].to(dev)
    report["fixed_samples"] = [{"sample_id": r["sample_id"], "event_id": r["event_id"]} for r in picks]
    g.add("setup", "固定 8 个 TRAIN 样本（无增强）", len(picks) == N_FIXED,
          {"n": len(picks), "events": sorted({r["event_id"] for r in picks})})

    # ---------- 构造三架构（同 seed，靠哈希确认身份）----------
    cfg_ref = copy.deepcopy(cfg_self)
    cfg_ref["segformer"]["model"]["exchange"] = "none"
    cfg_ref["segformer"]["experiment"] = "LateDetail_reference"

    set_seed(seed, True)
    ref = build_model(cfg_ref["segformer"]).to(dev)
    set_seed(seed, True)
    mdl_self = build_model(cfg_self["segformer"]).to(dev)
    set_seed(seed, True)
    mdl_cross = build_model(cfg_cross["segformer"]).to(dev)

    sd_ref, sd_self, sd_cross = ref.state_dict(), mdl_self.state_dict(), mdl_cross.state_dict()
    common = sorted(set(sd_ref) & set(sd_self))
    exch_keys = sorted(k for k in sd_self if k.startswith("exchange."))
    missing = sorted(set(sd_ref) - set(sd_self))
    extra = sorted(set(sd_self) - set(sd_ref))

    report["identity"] = {
        "common_keys": len(common),
        "exchange_keys": exch_keys,
        "ref_only_keys": missing,
        "self_only_keys": extra,
        "hash_common_ref": _subset_hash(sd_ref, common),
        "hash_common_self": _subset_hash(sd_self, common),
        "hash_common_cross": _subset_hash(sd_cross, common),
        "hash_exchange_self": _subset_hash(sd_self, exch_keys),
        "hash_exchange_cross": _subset_hash(sd_cross, exch_keys),
    }
    ident = report["identity"]
    g.add("G2", "公共参数初始哈希：LateDetail vs Self", ident["hash_common_ref"] == ident["hash_common_self"],
          ident["hash_common_self"][:16])
    g.add("G2", "公共参数初始哈希：Self vs Cross", ident["hash_common_self"] == ident["hash_common_cross"],
          ident["hash_common_cross"][:16])
    g.add("G2", "新增模块初始哈希：Self vs Cross", ident["hash_exchange_self"] == ident["hash_exchange_cross"],
          ident["hash_exchange_cross"][:16])
    g.add("G2", "Self 相对 LateDetail 只多出 exchange.*", len(extra) == len(exch_keys) and not missing,
          {"extra": len(extra), "missing": len(missing)})

    # ---------- G1 前向形状 ----------
    shapes = {}
    for tag, m in (("self", mdl_self), ("cross", mdl_cross)):
        _forward_logits(m, x8[:2])
        shapes[tag] = m._stage_shapes
    report["shapes"] = shapes
    exp = {"input": [2, 10, 128, 128], "stage1": [2, 32, 32, 32], "E_in": [2, 64, 16, 16],
           "E_out": [2, 64, 16, 16], "D_in": [2, 32, 128, 128], "D_out": [2, 32, 128, 128],
           "stage3": [2, 160, 8, 8], "stage4": [2, 256, 4, 4]}
    same = all(shapes[t].get(k) == v for t in shapes for k, v in exp.items())
    g.add("G1", "输入/阶段2/E校正/D/阶段3/阶段4 形状符合计划", same, exp)

    # E_new 必须真的影响后续编码：改 α_E 后 stage3 应变化
    with torch.no_grad():
        mdl_cross.eval()
        hold = mdl_cross.exchange.alpha_e.detach().clone()
        mdl_cross.exchange.alpha_e.fill_(0.05)
        s3_a = _capture_stage3(mdl_cross, x8[:2])
        mdl_cross.exchange.alpha_e.copy_(hold)
        s3_b = _capture_stage3(mdl_cross, x8[:2])
    delta_stage3 = float((s3_a - s3_b).abs().max())
    g.add("G1", "E_new 确实进入阶段 3（改 α_E 后 stage3 变化）", delta_stage3 > 0,
          {"max_abs_delta_stage3": delta_stage3})

    # ---------- G2 初始输出与 LateDetail 逐像素一致 ----------
    lp_ref = _forward_logits(ref, x8)
    for tag, m in (("self", mdl_self), ("cross", mdl_cross)):
        d = float((lp_ref - _forward_logits(m, x8)).abs().max())
        report.setdefault("init_equivalence", {})[tag] = d
        g.add("G2", f"eval 下零系数初始输出 vs LateDetail（{tag}）", d <= 1e-6, {"max_abs_diff": d})

    # ---------- G3 参数分组 / 学习率 / 参数量 ----------
    info = {}
    for tag, m in (("late", ref), ("self", mdl_self), ("cross", mdl_cross)):
        grp = m.param_groups()
        seen_ids = [id(p) for ps in grp.values() for p in ps]
        all_ids = [id(p) for p in m.parameters() if p.requires_grad]
        dup = len(seen_ids) != len(set(seen_ids))
        missing_p = set(all_ids) - set(seen_ids)
        exch_names = sorted(n for n, _ in m.named_parameters() if n.startswith("exchange."))
        exch_in_new = all(
            n in {nm for nm, _ in m.named_parameters() if any(nm == x for x in [n])} for n in exch_names
        )
        # 交换模块参数是否落入 new 组
        new_ids = {id(p) for p in grp["new"]}
        exch_in_new = all(id(p) in new_ids for n, p in m.named_parameters() if n.startswith("exchange."))
        info[tag] = {
            "params": m.count_params(),
            "n_param_tensors": len(all_ids),
            "groups": {k: len(v) for k, v in grp.items()},
            "duplicate_membership": dup,
            "params_missing_from_groups": len(missing_p),
            "exchange_in_new_group": exch_in_new,
        }
    report["param_groups"] = info
    g.add("G3", "参数分组无重复、无遗漏（三架构）",
          all(not v["duplicate_membership"] and v["params_missing_from_groups"] == 0 for v in info.values()),
          {k: {"dup": v["duplicate_membership"], "missing": v["params_missing_from_groups"]} for k, v in info.items()})
    same_n = info["self"]["params"] == info["cross"]["params"]
    g.add("G3", "Self / Cross 可训练参数精确相等", same_n,
          {"self": info["self"]["params"], "cross": info["cross"]["params"]})
    g.add("G3", "交换模块参数落在 new 组（6e-4），未被并入 encoder",
          all(v["exchange_in_new_group"] for v in info.values()),
          {k: v["exchange_in_new_group"] for k, v in info.items()})

    opt_probe = _make_optimizer(mdl_cross, cfg_cross)
    lr_rows = [{"lr": round(float(pg["lr"]), 8), "n_tensors": len(pg["params"])} for pg in opt_probe.param_groups]
    report["optimizer_lr_groups"] = lr_rows
    lrs = sorted({r["lr"] for r in lr_rows})
    g.add("G3", "学习率组只有 6e-5(encoder) 与 6e-4(其余)",
          lrs == [6e-05, 0.0006] or lrs == [0.0006, 6e-05] or set(lrs) == {6e-05, 0.0006}, lrs)

    # ---------- G4 真实反传：20 步窗口 ----------
    for tag, m, cfgm in (("self", mdl_self, cfg_self), ("cross", mdl_cross, cfg_cross)):
        opt = _make_optimizer(m, cfgm)
        sched = _make_scheduler(opt, cfgm, int(args.steps))
        m.train()
        opt.zero_grad(set_to_none=True)
        out = m(x8)
        loss, _ = _loss(cfgm, out["logits"], batch8["mask"].to(dev), batch8["valid"].to(dev), out.get("base_logits"))
        loss.backward()
        first_norms = _grad_norms(m)
        opt.zero_grad(set_to_none=True)
        rows = []
        for step in range(20):
            loss, gn, cur = _one_step(m, batch8, cfgm, opt, sched, dev)
            rows.append({"step": step + 1, "loss": float(loss), **{k: float(v) for k, v in gn.items()}})
        nonzero = {k: sum(1 for r in rows if abs(r[k]) > 0) for k in ("p", "h_e", "h_d", "alpha_e", "alpha_d")}
        moved = {k: abs(rows[-1][k] - first_norms[k]) if k in first_norms else None for k in rows[-1]}
        report.setdefault("grad_window", {})[tag] = {"first_step_norms": first_norms, "rows": rows,
                                                    "nonzero_steps": nonzero}
        g.add("G4", f"20 步内两方向均获得有限梯度（{tag}）",
              all(nonzero[k] > 0 for k in ("p", "h_e", "h_d", "alpha_e", "alpha_d")), nonzero)
        finite = all(math.isfinite(r["loss"]) for r in rows)
        g.add("G4", f"20 步内 loss 有限（{tag}）", finite, {"loss_last": rows[-1]["loss"]})

    # ---------- G5 干预诊断 ----------
    diag = {}
    with torch.no_grad():
        for tag, m in (("cross", mdl_cross), ("self", mdl_self)):
            m.eval()
            m.exchange.alpha_e.fill_(0.0)
            m.exchange.alpha_d.fill_(0.0)
            d0 = _capture_e_d(m, x8[:2])
            # 方向独立：α_D=0 时 D 必须完全不变
            d_eq = float((d0["D_new"] - d0["D_in"]).abs().max())
            # 强制 α_D≠0 后，D 必须改变
            m.exchange.alpha_d.fill_(0.05)
            d1 = _capture_e_d(m, x8[:2])
            d_chg = float((d1["D_new"] - d1["D_in"]).abs().max())
            # 缩放 D：cross 应改变 stage3，self 不应
            m.exchange.alpha_e.fill_(0.05)
            m.exchange.alpha_d.fill_(0.0)
            s3_base = _capture_stage3(m, x8[:2], scale_d=1.0)
            s3_pert = _capture_stage3(m, x8[:2], scale_d=3.0)
            s3_delta = float((s3_base - s3_pert).abs().max())
            m.exchange.alpha_e.fill_(0.0)
            diag[tag] = {"D_unchanged_when_alpha_d_zero": d_eq, "D_changed_when_alpha_d_nonzero": d_chg,
                         "stage3_delta_under_D_scaling": s3_delta}
    report["intervention"] = diag
    g.add("G5", "α_D=0 时 D_new ≡ D（两架构）",
          all(v["D_unchanged_when_alpha_d_zero"] == 0.0 for v in diag.values()),
          {k: v["D_unchanged_when_alpha_d_zero"] for k, v in diag.items()})
    g.add("G5", "α_D≠0 时 D_new 改变（两架构）",
          all(v["D_changed_when_alpha_d_nonzero"] > 0 for v in diag.values()),
          {k: v["D_changed_when_alpha_d_nonzero"] for k, v in diag.items()})
    g.add("G5", "缩放 D 改变 Cross 阶段3 输入", diag["cross"]["stage3_delta_under_D_scaling"] > 0,
          diag["cross"]["stage3_delta_under_D_scaling"])
    g.add("G5", "缩放 D 不改变 Self 阶段3 输入（无跨路影响）",
          diag["self"]["stage3_delta_under_D_scaling"] == 0.0,
          diag["self"]["stage3_delta_under_D_scaling"])

    # ---------- G6 100 步拟合 + α 与两路更新 ----------
    fit = {}
    for tag, m, cfgm in (("self", mdl_self, cfg_self), ("cross", mdl_cross, cfg_cross)):
        set_seed(seed, True)
        fresh = build_model(cfgm["segformer"]).to(dev)
        opt = _make_optimizer(fresh, cfgm)
        sched = _make_scheduler(opt, cfgm, int(args.steps))
        curve = []
        for step in range(int(args.steps)):
            loss, _, _ = _one_step(fresh, batch8, cfgm, opt, sched, dev)
            if (step + 1) % 20 == 0 or step == 0:
                iou, alpha = _eval_batch(fresh, batch8, dev), fresh.exchange.alphas()
                curve.append({"step": step + 1, "loss": round(loss, 6), "batch_iou": round(iou, 6),
                              **{k: round(v, 8) for k, v in alpha.items()}})
        fit[tag] = {"curve": curve, "final_params": fresh.count_params()}
        g.add("G6", f"{args.steps} 步后能在固定 8 样本上降 loss（{tag}）",
              curve[-1]["loss"] < curve[0]["loss"], {"first": curve[0]["loss"], "last": curve[-1]["loss"]})
        g.add("G6", f"两方向残差系数已离开零（{tag}）",
              curve[-1]["alpha_e_abs_max"] > 0 and curve[-1]["alpha_d_abs_max"] > 0,
              {"alpha_e_max": curve[-1]["alpha_e_abs_max"], "alpha_d_max": curve[-1]["alpha_d_abs_max"]})
    report["fit_100"] = fit

    # ---------- G7 checkpoint 往返 / batch64 显存 / FP32 ----------
    CKPT_TMP.parent.mkdir(parents=True, exist_ok=True)
    from landslide_seg.training.segformer_trainer import _save_checkpoint

    fresh = build_model(cfg_cross["segformer"]).to(dev)
    opt = _make_optimizer(fresh, cfg_cross)
    sched = _make_scheduler(opt, cfg_cross, 10)
    meta = {"stats_sha256": None, "manifest": {}, "model_identity": {}}
    _save_checkpoint(CKPT_TMP, fresh, opt, sched, 0, 0.0, 0, cfg_cross, meta, {"seed": seed}, extra={})
    reloaded, _ = load_checkpoint(CKPT_TMP, device=str(dev))
    d_lp = float((_forward_logits(fresh, x8[:4]) - _forward_logits(reloaded, x8[:4])).abs().max())
    g.add("G7", "checkpoint 保存/重载输出一致", d_lp == 0.0, {"max_abs_diff": d_lp})

    mem = {}
    try:
        big = collate([ds8[i % len(ds8)] for i in range(64)]).copy()
        big["image"] = big["image"].repeat(8, 1, 1, 1)[:64].to(dev)
        big["mask"] = big["mask"].repeat(8, 1, 1)[:64].to(dev)
        big["valid"] = big["valid"].repeat(8, 1, 1)[:64].to(dev)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        m = build_model(cfg_cross["segformer"]).to(dev)
        m.train()
        o = m(big["image"])
        loss, _ = _loss(cfg_cross, o["logits"], big["mask"], big["valid"], o.get("base_logits"))
        loss.backward()
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        mem = {
            "batch": int(big["image"].shape[0]),
            "loss": round(float(loss), 6),
            "peak_alloc_mib": round(torch.cuda.max_memory_allocated(dev) / 2**20, 1) if dev.type == "cuda" else None,
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved(dev) / 2**20, 1) if dev.type == "cuda" else None,
            "all_grads_finite": all(bool(torch.isfinite(g).all()) for g in grads),
            "n_grad_tensors": len(grads),
            "param_dtypes": sorted({str(p.dtype) for p in m.parameters()}),
        }
        g.add("G7", "batch64 前反向无 OOM 且梯度全有限",
              mem["all_grads_finite"] and mem["batch"] == 64, mem)
        g.add("G7", "全参数 FP32（无静默 AMP）", mem["param_dtypes"] == ["torch.float32"], mem["param_dtypes"])
    except RuntimeError as exc:
        mem = {"error": f"{type(exc).__name__}: {exc}"}
        g.add("G7", "batch64 前反向无 OOM", False, mem)
    report["memory"] = mem

    # ---------- 汇总 ----------
    report["gates"] = g.rows
    report["all_pass"] = g.ok()
    report["failed_hard"] = [r for r in g.rows if r["hard"] and not r["ok"]]
    OUT_JSON.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=_jsonable), encoding="utf-8")
    OUT_MD.write_text(_to_md(report), encoding="utf-8")
    print(f"\n[done] all_pass={report['all_pass']} -> {OUT_MD}")
    if not report["all_pass"]:
        sys.exit(1)


# ----------------------------------------------------------------------
def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, torch.dtype):
        return str(o)
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return str(o)


def _grad_norms(model) -> dict:
    def n(prefix, first_only=False):
        vals = [p.grad.detach().norm().item() for nm, p in model.named_parameters()
                if nm.startswith(prefix) and p.grad is not None]
        if not vals:
            return 0.0
        return float(sum(vals)) if not first_only else float(vals[0])

    return {"p": n("exchange.p."), "h_e": n("exchange.h_e."), "h_d": n("exchange.h_d."),
            "alpha_e": float(model.exchange.alpha_e.grad.norm()) if model.exchange.alpha_e.grad is not None else 0.0,
            "alpha_d": float(model.exchange.alpha_d.grad.norm()) if model.exchange.alpha_d.grad is not None else 0.0}


def _one_step(model, batch, cfg, opt, sched, dev):
    model.train()
    image = batch["image"].to(dev)
    mask = batch["mask"].to(dev)
    valid = batch["valid"].to(dev)
    opt.zero_grad(set_to_none=True)
    out = model(image)
    loss, _ = _loss(cfg, out["logits"], mask, valid, out.get("base_logits"))
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite loss")
    loss.backward()
    clip = float(cfg["segformer"]["optim"]["grad_clip"])
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    gn = _grad_norms(model) if model.exchange is not None else {}
    opt.step()
    sched.step()
    return float(loss.detach()), gn, model


@torch.no_grad()
def _eval_batch(model, batch, dev) -> float:
    model.eval()
    prob = model(batch["image"].to(dev))["logits"].float()[:, 0].sigmoid().cpu().numpy()
    gt, valid = batch["mask"].numpy().astype(bool), batch["valid"].numpy().astype(bool)
    cs = [confusion(prob[i] >= 0.5, gt[i], valid[i]) for i in range(len(prob))]
    return summarize(cs)["iou"] or 0.0


def _capture_stage3(model, x, scale_d: float = 1.0):
    """诊断副本：把 D 按 scale_d 缩放后，返回阶段 3 的输入特征。"""
    with torch.no_grad():
        d = model.stem(x) * float(scale_d)
        enc = model.model.segformer.encoder
        cfg = model.model.config
        hidden = x
        stages = []
        for idx in range(4):
            hidden, hh, ww = enc.patch_embeddings[idx](hidden)
            for layer in enc.block[idx]:
                hidden = layer(hidden, hh, ww, False)[0]
            hidden = enc.layer_norm[idx](hidden)
            if idx != 3 or cfg.reshape_last_stage:
                hidden = hidden.reshape(x.shape[0], hh, ww, -1).permute(0, 3, 1, 2).contiguous()
            if idx == 1:
                hidden, _ = model.exchange(hidden, d)
            if idx == 2:
                return hidden.detach().clone()
            stages.append(hidden)
    return stages[-1]


def _capture_e_d(model, x):
    """诊断副本：跑前两个编码阶段，返回 E/D 交换前后的张量。"""
    with torch.no_grad():
        d_in = model.stem(x)
        enc = model.model.segformer.encoder
        hidden = x
        for idx in range(2):
            hidden, hh, ww = enc.patch_embeddings[idx](hidden)
            for layer in enc.block[idx]:
                hidden = layer(hidden, hh, ww, False)[0]
            hidden = enc.layer_norm[idx](hidden)
            hidden = hidden.reshape(x.shape[0], hh, ww, -1).permute(0, 3, 1, 2).contiguous()
        e_in = hidden
        e_new, d_new = model.exchange(e_in, d_in)
    return {"E_in": e_in, "D_in": d_in, "E_new": e_new, "D_new": d_new}


def _to_md(rep: dict) -> str:
    L = ["# v4r4 接口验证报告（训练前门槛）\n",
         f"结论：**{'全部通过' if rep['all_pass'] else '存在未通过项，禁止启动正式训练'}**\n",
         f"设备 `{rep['device']}`；固定样本 {len(rep.get('fixed_samples', []))} 个。\n"]
    L.append("## 1. 初始参数身份\n")
    idn = rep["identity"]
    L.append(f"- 公共参数张量 {idn['common_keys']} 个；新增交换参数 {len(idn['exchange_keys'])} 个")
    L.append(f"- LateDetail 公共哈希 `{idn['hash_common_ref'][:32]}`")
    L.append(f"- Self 公共哈希 `{idn['hash_common_self'][:32]}`")
    L.append(f"- Cross 公共哈希 `{idn['hash_common_cross'][:32]}`")
    L.append(f"- Self 新增哈希 `{idn['hash_exchange_self'][:32]}`")
    L.append(f"- Cross 新增哈希 `{idn['hash_exchange_cross'][:32]}`")
    L.append(f"- 交换参数键：{', '.join(idn['exchange_keys'])}\n")

    L.append("## 2. 门槛逐项\n")
    L.append("| 门槛 | 检查项 | 结果 | 说明 |")
    L.append("|---|---|---|---|")
    for r in rep["gates"]:
        flag = "PASS" if r["ok"] else ("FAIL" if r["ok"] is False else "INFO")
        d = r["detail"]
        if isinstance(d, dict):
            d = ", ".join(f"{k}={v}" for k, v in list(d.items())[:6])
        elif isinstance(d, list):
            d = ", ".join(str(x) for x in d[:6])
        L.append(f"| {r['gate']} | {r['check']} | {flag} | {str(d)[:180]} |")

    L.append("\n## 3. 参数量与分组\n")
    L.append("| 架构 | 总参数 | encoder | decoder | new | 交换模块 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for k, v in rep["param_groups"].items():
        p = v["params"]
        L.append(f"| {k} | {p['total']:,} | {p['encoder']:,} | {p['decoder']:,} | {p['new']:,} | "
                 f"{p['total'] - 3753602:,} |")
    L.append(f"\n优化器学习率组：{rep['optimizer_lr_groups']}\n")

    L.append("## 4. 100 步拟合（固定 8 样本，无增强）\n")
    L.append("| 架构 | 步 | loss | batch IoU | α_E max | α_D max |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for tag, v in rep.get("fit_100", {}).items():
        for row in v["curve"]:
            L.append(f"| {tag} | {row['step']} | {row['loss']:.6f} | {row['batch_iou']:.4f} | "
                     f"{row['alpha_e_abs_max']:.3e} | {row['alpha_d_abs_max']:.3e} |")

    L.append("\n## 5. 显存与数值\n")
    L.append(f"```\n{json.dumps(rep.get('memory'), indent=1, ensure_ascii=False)}\n```\n")

    L.append("## 6. 干预诊断\n")
    L.append("| 架构 | α_D=0 时 D 变化 | α_D≠0 时 D 变化 | 缩放 D 后 stage3 变化 |")
    L.append("|---|---:|---:|---:|")
    for tag, v in rep.get("intervention", {}).items():
        L.append(f"| {tag} | {v['D_unchanged_when_alpha_d_zero']:.3e} | "
                 f"{v['D_changed_when_alpha_d_nonzero']:.3e} | {v['stage3_delta_under_D_scaling']:.3e} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
