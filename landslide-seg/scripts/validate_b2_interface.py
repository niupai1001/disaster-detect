"""P2-M1：B2 接入接口验证（训练前门槛）。

按计划要求逐项核对，全部通过才允许起正式训练：
1. 权重身份：文件字节数/sha256、config 架构字段、state_dict 关键形状、下载元信息；
2. 官方加载：`from_pretrained(output_loading_info=True)` 的 missing/unexpected/mismatched；
3. 规格：decoder 维度、参数量、128px 下各阶段形状；
4. 真实前反向：batch 64、梯度有限、无 OOM、峰值显存；
5. 少量 TRAIN 拟合：固定 8 个样本 100 次更新，loss 必须下降；
6. checkpoint 保存/重载输出一致；
7. 数值精度：全参数 FP32。

不做的事：不跑正式训练、不改数据/损失/预算、不用随机初始化冒充预训练。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch  # noqa: E402

from landslide_seg.config import file_sha256, load_segformer_config, resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import (  # noqa: E402
    S2SegDataset, collate, filter_split, load_stats, read_manifest,
)
from landslide_seg.models.factory import build_model  # noqa: E402
from landslide_seg.models.segformer_probe import state_digest  # noqa: E402
from landslide_seg.training.segformer_trainer import (  # noqa: E402
    _loss, _make_optimizer, _make_scheduler, _save_checkpoint, load_checkpoint, set_seed,
)
from landslide_seg.training.metrics import confusion, summarize  # noqa: E402

CFG = "configs/segformer/p2m1/segformer_b2_latedetail_alltrain_s1234.yaml"
OUT = ROOT / "reports" / "p2m1"
STEPS_FIT = 100
N_FIXED = 8
CKPT = ROOT / "outputs" / "segformer" / "_p2m1_validation" / "roundtrip.pt"


class Gate:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, gate: str, name: str, ok: bool, detail, *, hard: bool = True) -> None:
        self.rows.append({"gate": gate, "check": name, "ok": bool(ok), "hard": bool(hard), "detail": detail})
        tag = "PASS" if ok else ("FAIL" if hard else "WARN")
        print(f"[{tag}] {gate} · {name}: {detail}")

    def ok(self) -> bool:
        return all(r["ok"] for r in self.rows if r["hard"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CFG)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    cfg = load_segformer_config(args.config)
    s = cfg["segformer"]
    m = s["model"]
    wdir = resolve_path(m["pretrained_dir"])
    g = Gate()
    rep: dict = {"config": args.config, "device": str(dev)}

    # ---------- 1. 权重身份 ----------
    cfg_json = json.loads((wdir / "config.json").read_text(encoding="utf-8"))
    meta_path = wdir / "download_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else None
    binp = wdir / "pytorch_model.bin"
    sd_raw = torch.load(binp, map_location="cpu", weights_only=True)
    identity = {
        "dir": str(wdir).replace("\\", "/"),
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": file_sha256(p)}
                  for p in sorted(wdir.iterdir()) if p.is_file()},
        "config": {k: cfg_json.get(k) for k in
                   ("decoder_hidden_size", "depths", "hidden_sizes", "num_attention_heads",
                    "sr_ratios", "patch_sizes", "strides", "num_channels", "reshape_last_stage")},
        "state_dict_numel": sum(v.numel() for v in sd_raw.values()),
        "markers": {
            "linear_c_out": int(sd_raw["decode_head.linear_c.0.proj.weight"].shape[0]),
            "classifier": list(sd_raw["decode_head.classifier.weight"].shape),
            "first_conv": list(sd_raw["segformer.encoder.patch_embeddings.0.proj.weight"].shape),
        },
        "download_meta": meta,
        "dir_name_matches_content": "768" in json.dumps(cfg_json.get("decoder_hidden_size")),
    }
    rep["identity"] = identity
    b2_arch_ok = (
        identity["config"]["decoder_hidden_size"] == 768
        and identity["config"]["depths"] == [3, 4, 6, 3]
        and identity["config"]["hidden_sizes"] == [64, 128, 320, 512]
        and identity["markers"]["linear_c_out"] == 768
        and identity["markers"]["first_conv"][:2] == [64, 3]
    )
    g.add("G1", "权重文件内容 = B2-768 架构", b2_arch_ok, identity["markers"])

    name_mismatch = meta and "512-512" in str(meta.get("repo", ""))
    g.add("G1", "目录名/仓库名与内容一致（信息项，不阻断）", not name_mismatch,
          {"dir": wdir.name, "meta_repo": (meta or {}).get("repo"),
           "note": "目录名写 512-512，但 config 与张量均为 B2-768；以内容为准，已在报告中标注"},
          hard=False)

    # ---------- 2. 官方加载 ----------
    from transformers import SegformerForSemanticSegmentation
    model_hf, info = SegformerForSemanticSegmentation.from_pretrained(
        str(wdir), output_loading_info=True)
    load_info = {k: (len(v) if isinstance(v, list) else v) for k, v in info.items()}
    rep["hf_load_info"] = {k: (v if not isinstance(v, list) else v[:8]) for k, v in info.items()}
    g.add("G2", "from_pretrained 无 missing/unexpected/mismatched",
          not info.get("missing_keys") and not info.get("unexpected_keys") and not info.get("mismatched_keys"),
          load_info)
    del model_hf

    # ---------- 3. 规格 ----------
    set_seed(int(s["seed"]), True)
    model = build_model(s, pretrained_dir=wdir).to(dev)
    counts = model.count_params()
    dec_dim = int(model.model.config.decoder_hidden_size)
    with torch.no_grad():
        probe = torch.zeros(2, 10, 128, 128, device=dev)
        out = model(probe)
        feat = model._decoder_feat
    spec = {
        "backbone": model.backbone, "decoder_hidden": dec_dim, "params": counts,
        "decoder_feat_shape": list(feat.shape) if feat is not None else None,
        "logits_shape": list(out["logits"].shape),
        "delta_vs_b0_total": counts["total"] - 3_753_602,
    }
    rep["spec"] = spec
    g.add("G3", "decoder 维度 = 768 且 logits 为 [B,1,128,128]",
          dec_dim == 768 and list(out["logits"].shape) == [2, 1, 128, 128], spec)
    g.add("G3", "参数量与离线核算一致（27,430,338）", counts["total"] == 27_430_338, counts["total"])

    # ---------- 数据 ----------
    records = read_manifest(resolve_path(s["data"]["manifest"]))
    stats = load_stats(resolve_path(s["data"]["stats"]))
    train_recs = filter_split(records, "train")
    ordered = sorted(train_recs, key=lambda r: r["sample_id"])
    picks = list({ordered[int(round(i * (len(ordered) - 1) / (N_FIXED - 1)))]["sample_id"]:
                  ordered[int(round(i * (len(ordered) - 1) / (N_FIXED - 1)))] for i in range(N_FIXED)}.values())
    ds8 = S2SegDataset(picks, stats, augment=False, preload=True)
    batch8 = collate([ds8[i] for i in range(len(ds8))])
    rep["n_train"] = len(train_recs)

    # ---------- 4. 真实前反向 batch64 ----------
    big = {k: (v.repeat(8, *([1] * (v.dim() - 1)))[:64] if torch.is_tensor(v) else v)
           for k, v in batch8.items()}
    big["image"] = big["image"].to(dev)
    big["mask"] = big["mask"].to(dev)
    big["valid"] = big["valid"].to(dev)
    model.train()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    o = model(big["image"])
    loss, _ = _loss(cfg, o["logits"], big["mask"], big["valid"], o.get("base_logits"))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    mem = {
        "batch": int(big["image"].shape[0]), "loss": round(float(loss), 6),
        "n_grad_tensors": len(grads),
        "all_finite": all(bool(torch.isfinite(g).all()) for g in grads),
        "peak_alloc_mib": round(torch.cuda.max_memory_allocated(dev) / 2**20, 1) if dev.type == "cuda" else None,
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(dev) / 2**20, 1) if dev.type == "cuda" else None,
        "param_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
    }
    rep["memory"] = mem
    g.add("G4", "batch64 前反向无 OOM 且梯度全有限",
          mem["all_finite"] and mem["batch"] == 64 and mem["n_grad_tensors"] > 0, mem)
    g.add("G4", "全参数 FP32", mem["param_dtypes"] == ["torch.float32"], mem["param_dtypes"])

    # ---------- 5. 少量 TRAIN 拟合 ----------
    set_seed(int(s["seed"]), True)
    fit_model = build_model(s, pretrained_dir=wdir).to(dev)
    opt = _make_optimizer(fit_model, cfg)
    sched = _make_scheduler(opt, cfg, STEPS_FIT)
    curve = []
    for step in range(STEPS_FIT):
        fit_model.train()
        opt.zero_grad(set_to_none=True)
        out = fit_model(batch8["image"].to(dev))
        loss, _ = _loss(cfg, out["logits"], batch8["mask"].to(dev), batch8["valid"].to(dev),
                        out.get("base_logits"))
        if not math.isfinite(float(loss)):
            raise FloatingPointError("nonfinite fit loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fit_model.parameters(),
                                       float(cfg["segformer"]["optim"]["grad_clip"]),
                                       error_if_nonfinite=True)
        opt.step(); sched.step()
        if step % 20 == 0 or step == STEPS_FIT - 1:
            curve.append({"step": step + 1, "loss": round(float(loss), 6),
                          "batch_iou": round(_iou(fit_model, batch8, dev), 4)})
    rep["fit"] = curve
    g.add("G5", "固定 8 样本 100 步能降 loss", curve[-1]["loss"] < curve[0]["loss"],
          {"first": curve[0]["loss"], "last": curve[-1]["loss"], "iou_last": curve[-1]["batch_iou"]})

    # ---------- 6. checkpoint 往返 ----------
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    _save_checkpoint(CKPT, fit_model, opt, sched, 0, 0.0, 0, cfg,
                     {"stats_sha256": None, "manifest": {}, "model_identity": {}}, {"seed": int(s["seed"])},
                     extra={})
    reloaded, _ = load_checkpoint(CKPT, device=str(dev))
    with torch.no_grad():
        a = fit_model.eval()(batch8["image"].to(dev))["logits"]
        b = reloaded.eval()(batch8["image"].to(dev))["logits"]
    d = float((a - b).abs().max())
    g.add("G6", "checkpoint 重载输出一致", d == 0.0, {"max_abs_diff": d})
    g.add("G6", "重载后参数哈希一致",
          state_digest(fit_model.state_dict()) == state_digest(reloaded.state_dict()),
          state_digest(reloaded.state_dict())[:16])

    rep["gates"] = g.rows
    rep["all_pass"] = g.ok()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "b2_interface_validation.json").write_text(
        json.dumps(rep, indent=1, ensure_ascii=False, default=str), encoding="utf-8")

    L = ["# B2 接口验证（训练前门槛）\n",
         f"结论：**{'全部通过' if rep['all_pass'] else '存在未通过项，禁止起训'}**\n",
         "## 1. 权重身份\n",
         f"- 目录 `{identity['dir']}`",
         f"- `pytorch_model.bin` {identity['files']['pytorch_model.bin']['bytes']:,} B，"
         f"sha256 `{identity['files']['pytorch_model.bin']['sha256']}`",
         f"- config: decoder {identity['config']['decoder_hidden_size']}，depths {identity['config']['depths']}，"
         f"hidden {identity['config']['hidden_sizes']}",
         f"- 张量标记: linear_c→{identity['markers']['linear_c_out']}，"
         f"classifier {identity['markers']['classifier']}，首层 {identity['markers']['first_conv']}",
         f"- 下载元信息: `{json.dumps(meta, ensure_ascii=False)}`",
         "\n> **命名不一致**：目录与 `download_meta.repo` 写的是 `…ade-512-512`，"
         "但 config 与全部张量都是 B2-768（decoder 768 / depths 3,4,6,3）。"
         "本报告以文件内容为准，未取得上游哈希校验（HF API 不可达），身份链只到本地文件。\n",
         "\n## 2. 规格\n", "```", json.dumps(spec, indent=1, ensure_ascii=False), "```",
         "\n## 3. 门槛\n", "| 门槛 | 检查 | 结果 | 说明 |", "|---|---|---|---|"]
    for r in rep["gates"]:
        det = r["detail"]
        if isinstance(det, dict):
            det = ", ".join(f"{k}={v}" for k, v in list(det.items())[:6])
        tag = "PASS" if r["ok"] else ("FAIL" if r["hard"] else "WARN")
        L.append(f"| {r['gate']} | {r['check']} | {tag} | {str(det)[:150]} |")
    (OUT / "b2_interface_validation.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n[done] all_pass={rep['all_pass']} -> {OUT / 'b2_interface_validation.md'}")
    if not rep["all_pass"]:
        sys.exit(1)


@torch.no_grad()
def _iou(model, batch, dev) -> float:
    model.eval()
    prob = model(batch["image"].to(dev))["logits"].float()[:, 0].sigmoid().cpu().numpy()
    gt, valid = batch["mask"].numpy().astype(bool), batch["valid"].numpy().astype(bool)
    cs = [confusion(prob[i] >= 0.5, gt[i], valid[i]) for i in range(len(prob))]
    return summarize(cs)["iou"] or 0.0


if __name__ == "__main__":
    main()
