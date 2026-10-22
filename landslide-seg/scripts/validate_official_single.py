"""BENCH_OFFICIAL 单期协议的最小验证（训练前）。

对应计划步骤 3 的门槛：验证读取、前反向、重载与指标，且不以随机初始化代替预训练。

检查项：
1. 协议文件一致性（`validate_protocol` 在官方 TRAIN/VAL 上通过）；
2. 三个 split 的读取（抽样，含 valid/标签范围）；
3. 指标口径：用一个可手算的微型样例核对 pooled IoU/Dice，并显式给出
   「仅 valid 像素」与「全像素」两种口径的差值（官方实现用全像素）；
4. 真实前反向：官方 TRAIN 上 batch 64，梯度有限、无 OOM、FP32；
5. 预训练权重加载身份；
6. checkpoint 保存/重载输出一致。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402
import torch  # noqa: E402

from landslide_seg.config import file_sha256, load_segformer_config, resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import (  # noqa: E402
    S2SegDataset, collate, filter_split, load_stats, read_manifest,
)
from landslide_seg.models.factory import build_model  # noqa: E402
from landslide_seg.training.protocol import validate_protocol  # noqa: E402
from landslide_seg.training.segformer_trainer import (  # noqa: E402
    _loss, _make_optimizer, _make_scheduler, _save_checkpoint, load_checkpoint, set_seed,
)
from landslide_seg.training.metrics import confusion, summarize  # noqa: E402

CFG = "configs/segformer/official_single/segformer_b0_latedetail_officialsplit_singledate_s1234.yaml"
OUT = ROOT / "reports" / "official_single"
CKPT = ROOT / "outputs" / "segformer" / "_official_single_validation" / "roundtrip.pt"


def main() -> None:
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = load_segformer_config(CFG)
    s = cfg["segformer"]
    gates: list[dict] = []

    def gate(name: str, ok: bool, detail, hard: bool = True):
        gates.append({"check": name, "ok": bool(ok), "hard": hard, "detail": detail})
        tag = "PASS" if ok else ("FAIL" if hard else "WARN")
        print(f"[{tag}] {name}: {detail}")

    rep: dict = {"config": CFG, "device": str(dev)}

    # ---------- 1. 协议一致性 ----------
    recs = read_manifest(resolve_path(s["data"]["manifest"]))
    stats = load_stats(resolve_path(s["data"]["stats"]))
    pinfo = validate_protocol(cfg, recs, stats)
    rep["protocol"] = pinfo
    gate("validate_protocol 通过", pinfo is not None, pinfo)

    train = filter_split(recs, "train")
    val = filter_split(recs, "val")
    test_recs = read_manifest(resolve_path("manifests/official_single/test.jsonl"))
    rep["counts"] = {"train": len(train), "val": len(val), "test": len(test_recs)}
    gate("三个 split 均非空", all(len(x) > 0 for x in (train, val, test_recs)), rep["counts"])

    # ---------- 2. 读取抽样 ----------
    probe = {}
    for name, rs in (("train", train), ("val", val), ("test", test_recs)):
        picks = [rs[i] for i in (0, len(rs) // 2, len(rs) - 1)]
        ds = S2SegDataset(picks, stats, augment=False, preload=True)
        b = collate([ds[i] for i in range(len(ds))])
        probe[name] = {
            "n": len(ds),
            "image": list(b["image"].shape), "dtype": str(b["image"].dtype),
            "mask_values": sorted({int(v) for v in np.unique(b["mask"].numpy())}),
            "valid_ratio_mean": round(float(b["valid"].float().mean()), 4),
            "sample_ids": [str(x) for x in b["sample_id"]],
        }
        ok = probe[name]["image"][1:] == [10, 128, 128] and set(probe[name]["mask_values"]) <= {0, 1}
        gate(f"{name} 读取正常", ok, probe[name]["image"])
    rep["read_probe"] = probe

    # ---------- 3. 指标口径 ----------
    # 手算例：2x2，预测 [1,1,0,0]，GT [1,0,1,0]，全 valid
    pred = np.array([[True, True], [False, False]])
    gt = np.array([[True, False], [True, False]])
    valid_all = np.ones((2, 2), dtype=bool)
    c_all = confusion(pred, gt, valid_all)
    pooled_all = summarize([c_all])
    # 把「右上角那个 FP」标为无效：IoU 只看 tp/fp/fn，所以只有排除 FP/FN 才会改变 IoU
    valid_mask = np.array([[True, False], [True, True]])
    c_val = confusion(pred, gt, valid_mask)
    pooled_val = summarize([c_val])
    rep["metric_probe"] = {
        "case": "2x2, pred=[[T,T],[F,F]], gt=[[T,F],[T,F]], tp=1 fp=1 fn=1",
        "iou_all_pixels": pooled_all["iou"],            # 1/3
        "iou_drop_fp_pixel": pooled_val["iou"],         # 1/2（fp 被排除）
        "expected_all": 1 / 3,
        "expected_drop_fp": 1 / 2,
    }
    gate("pooled IoU 与手算一致（全像素 1/3）",
         abs(pooled_all["iou"] - 1 / 3) < 1e-12, pooled_all["iou"])
    gate("valid 口径差异可复现（排除 FP 像素后 1/3 → 1/2）",
         abs(pooled_val["iou"] - 0.5) < 1e-12,
         {"all": pooled_all["iou"], "valid_only": pooled_val["iou"],
          "note": "官方 torchmetrics 用全像素；本项目主口径用 valid 像素。"
                  "IoU/Dice/P/R 只在被排除像素为 FP 或 FN 时才变化，FPR 则必然变化。"
                  "并表前须按同一口径复算"},
         hard=False)

    # ---------- 4. 预训练加载 + 前反向 ----------
    set_seed(int(s["seed"]), True)
    model = build_model(s).to(dev)
    rep["params"] = model.count_params()
    rep["load_info"] = {k: (len(v) if isinstance(v, list) else v)
                        for k, v in (model.load_info or {}).items()}
    rep["weight_sha256"] = file_sha256(resolve_path(s["model"]["pretrained_dir"]) / "pytorch_model.bin")
    gate("预训练权重已加载（非随机初始化）",
         isinstance(model.load_info, dict) and not model.load_info.get("error_msgs"),
         {"missing": len(model.load_info.get("missing_keys", [])),
          "unexpected": len(model.load_info.get("unexpected_keys", [])),
          "pretrained": s["model"]["pretrained_source"]})

    ds_tr = S2SegDataset(train[:64], stats, augment=False, preload=True)
    batch = collate([ds_tr[i] for i in range(min(64, len(ds_tr)))])
    batch["image"] = batch["image"].to(dev)
    batch["mask"] = batch["mask"].to(dev)
    batch["valid"] = batch["valid"].to(dev)
    model.train()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    out = model(batch["image"])
    loss, _ = _loss(cfg, out["logits"], batch["mask"], batch["valid"], out.get("base_logits"))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    mem = {"batch": int(batch["image"].shape[0]), "loss": round(float(loss.detach()), 6),
           "n_grad": len(grads),
           "all_finite": all(bool(torch.isfinite(g).all()) for g in grads),
           "peak_alloc_mib": round(torch.cuda.max_memory_allocated(dev) / 2**20, 1) if dev.type == "cuda" else None,
           "dtypes": sorted({str(p.dtype) for p in model.parameters()})}
    rep["forward_backward"] = mem
    gate("batch64 前反向无 OOM 且梯度有限", mem["all_finite"] and mem["batch"] == 64, mem)
    gate("全参数 FP32", mem["dtypes"] == ["torch.float32"], mem["dtypes"])

    # ---------- 5. 重载 ----------
    opt = _make_optimizer(model, cfg)
    sched = _make_scheduler(opt, cfg, 10)
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    _save_checkpoint(CKPT, model, opt, sched, 0, 0.0, 0, cfg,
                     {"stats_sha256": file_sha256(resolve_path(s["data"]["stats"])),
                      "manifest": {}, "model_identity": {}}, {"seed": int(s["seed"])}, extra={})
    reloaded, _ = load_checkpoint(CKPT, device=str(dev))
    with torch.no_grad():
        a = model.eval()(batch["image"])["logits"]
        b = reloaded.eval()(batch["image"])["logits"]
    d = float((a - b).abs().max())
    gate("checkpoint 重载输出一致", d == 0.0, {"max_abs_diff": d})

    rep["gates"] = gates
    rep["all_pass"] = all(g["ok"] for g in gates if g["hard"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "interface_validation.json").write_text(
        json.dumps(rep, indent=1, ensure_ascii=False, default=str), encoding="utf-8")

    L = ["# BENCH_OFFICIAL 单期协议 · 最小验证\n",
         f"结论：**{'全部通过' if rep['all_pass'] else '存在未通过项'}**\n",
         f"- 样本数：{rep['counts']}",
         f"- 协议：{json.dumps(pinfo, ensure_ascii=False)}",
         f"- 参数：{rep['params']}",
         f"- 预训练权重 sha256：`{rep['weight_sha256']}`\n",
         "## 门槛\n", "| 检查 | 结果 | 说明 |", "|---|---|---|"]
    for g in gates:
        det = g["detail"]
        if isinstance(det, dict):
            det = ", ".join(f"{k}={v}" for k, v in list(det.items())[:5])
        L.append(f"| {g['check']} | {'PASS' if g['ok'] else ('FAIL' if g['hard'] else 'WARN')} | {str(det)[:160]} |")
    (OUT / "interface_validation.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n[done] all_pass={rep['all_pass']} -> {OUT / 'interface_validation.md'}")
    if not rep["all_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
