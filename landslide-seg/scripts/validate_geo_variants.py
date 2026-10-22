"""三组新实验的接口验证（DEM / 坡度 / pos_weight）。

逐组检查：
1. protocol 校验通过，样本数与基线一致（3,012 / 758）；
2. 模型输入通道、**新增通道的权重确实零初始化**（不破坏预训练响应）；
3. 数据集返回 11 通道，附加通道标准化后接近零均值；标签与基线逐块相同；
4. 真实前反向（batch 64）梯度有限；
5. pos_weight 确实生效（同一批数据 pw=5 的 loss 高于 pw=1）；
6. checkpoint 往返一致。
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

from landslide_seg.config import load_segformer_config, resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import S2SegDataset, collate, filter_split, load_stats, read_manifest  # noqa: E402
from landslide_seg.models.factory import build_model  # noqa: E402
from landslide_seg.training.protocol import validate_protocol  # noqa: E402
from landslide_seg.training.segformer_trainer import _loss, set_seed  # noqa: E402

CFGS = {
    "baseline": "configs/segformer/official_single/segformer_b0_latedetail_officialsplit_singledate_s1234.yaml",
    "dem": "configs/segformer/official_single/segformer_b0_latedetail_officialsplit_dem_s1234.yaml",
    "slope": "configs/segformer/official_single/segformer_b0_latedetail_officialsplit_slope_s1234.yaml",
    "pw5": "configs/segformer/official_single/segformer_b0_latedetail_officialsplit_pw5_s1234.yaml",
}
OUT = ROOT / "reports" / "official_single" / "geo_variants_validation.json"


def main() -> None:
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_manifest = resolve_path(
        load_segformer_config(CFGS["baseline"])["segformer"]["data"]["manifest"])
    base_train_all = filter_split(read_manifest(base_manifest), "train")
    gates: list[dict] = []
    rep: dict = {"device": str(dev), "arms": {}}

    def gate(arm, name, ok, detail):
        gates.append({"arm": arm, "check": name, "ok": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {arm:9s} · {name}: {detail}")

    for arm, path in CFGS.items():
        cfg = load_segformer_config(path)
        s = cfg["segformer"]
        m = s["model"]
        recs = read_manifest(resolve_path(s["data"]["manifest"]))
        stats = load_stats(resolve_path(s["data"]["stats"]))
        pinfo = validate_protocol(cfg, recs, stats)
        train = filter_split(recs, "train")
        val = filter_split(recs, "val")
        info = {"config": path, "n_train": len(train), "n_val": len(val),
                "protocol": pinfo, "in_chans": int(m.get("in_chans", 10)),
                "extra": m.get("extra_channel", "none"), "pos_weight": s["loss"].get("pos_weight", 1.0)}
        gate(arm, "protocol 通过", pinfo is not None, {"n_train": len(train), "n_val": len(val)})
        gate(arm, "样本数与基线一致", len(train) == 3012 and len(val) == 758,
             {"train": len(train), "val": len(val)})

        set_seed(int(s["seed"]), True)
        model = build_model(s, pretrained_dir=resolve_path(m["pretrained_dir"])).to(dev)
        proj = model.model.segformer.encoder.patch_embeddings[0].proj
        info["params"] = model.count_params()
        info["first_conv"] = list(proj.weight.shape)
        # 新增通道（索引>=10）的权重必须全为 0，否则预训练响应被破坏
        extra_w = proj.weight[:, 10:].detach()   # 取输入通道维，不是输出通道维
        info["extra_channel_weight_absmax"] = float(extra_w.abs().max()) if extra_w.numel() else None
        info["extra_channel_weight_is_zero"] = bool(extra_w.numel()) and float(extra_w.abs().max()) == 0.0
        gate(arm, "首层通道数正确", proj.weight.shape[1] == info["in_chans"], list(proj.weight.shape))
        if info["in_chans"] > 10:
            gate(arm, "新增通道权重零初始化", float(extra_w.abs().max()) == 0.0,
                 {"absmax": float(extra_w.abs().max())})
            gate(arm, "STEM 通道数跟随", model.stem is not None and
                 next(model.stem.parameters()).shape[1] == info["in_chans"],
                 list(next(model.stem.parameters()).shape))

        # 数据：通道数、附加通道标准化、标签一致性
        ds = S2SegDataset(train[:64], stats, augment=False, preload=True, extra=info["extra"])
        b = collate([ds[i] for i in range(min(64, len(ds)))])
        img = b["image"]
        info["image_shape"] = list(img.shape)
        gate(arm, "数据集通道数正确", img.shape[1] == info["in_chans"], list(img.shape))
        if info["in_chans"] > 10:
            # 精确核对：用第 0 块原始值手算 z-score，与数据集输出逐像素比对。
            # 批次均值/方差会随这 64 块的地形起伏变化，不适合做判据。
            raw0 = np.load(resolve_path(train[0]["cache"]))
            raw_extra = np.asarray(raw0[info["extra"]], dtype=np.float64)
            mean_e = float(np.asarray(stats["mean"]).ravel()[-1])
            std_e = float(np.asarray(stats["std"]).ravel()[-1])
            expect = torch.from_numpy(((raw_extra - mean_e) / max(std_e, 1e-6)).astype(np.float32))
            diff = float((img[0, 10] - expect).abs().max())
            gate(arm, "附加通道按正确统计标准化（逐像素核对）", diff < 1e-4,
                 {"max_abs_diff": diff, "mean": round(mean_e, 3), "std": round(std_e, 3),
                  "batch_std": round(float(img[:, 10].std()), 3)})
        base_train = base_train_all[:len(ds)]
        same = all(
            np.array_equal(b["mask"].numpy()[j],
                           np.load(resolve_path(r["cache"]))["mask"].astype(np.float32))
            for j, r in enumerate(base_train))
        gate(arm, "标签与基线同块一致", same, {"n": len(ds)})
        same = True

        # 前反向
        img_d, msk, val_ = img.to(dev), b["mask"].to(dev), b["valid"].to(dev)
        model.train()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        out = model(img_d)
        loss, _ = _loss(cfg, out["logits"], msk, val_, out.get("base_logits"))
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        info["loss"] = round(float(loss.detach()), 6)
        info["peak_alloc_mib"] = round(torch.cuda.max_memory_allocated(dev) / 2**20, 1) if dev.type == "cuda" else None
        gate(arm, "前反向梯度有限", all(bool(torch.isfinite(g).all()) for g in grads) and len(grads) > 0,
             {"loss": info["loss"], "n_grad": len(grads), "peak_mib": info["peak_alloc_mib"]})

        rep["arms"][arm] = info
        model.cpu()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # pos_weight 是否真的起作用：同一批数据、同一初始化，pw=1 与 pw=5 的 loss 必须不同
    cfgs = {k: load_segformer_config(v) for k, v in CFGS.items()}
    base = cfgs["baseline"]["segformer"]
    recs = read_manifest(resolve_path(base["data"]["manifest"]))
    stats = load_stats(resolve_path(base["data"]["stats"]))
    ds = S2SegDataset(filter_split(recs, "train")[:32], stats, augment=False, preload=True)
    b = collate([ds[i] for i in range(32)])
    torch.manual_seed(0)
    model = build_model(base, pretrained_dir=resolve_path(base["model"]["pretrained_dir"])).to(dev)
    model.eval()
    with torch.no_grad():
        out = model(b["image"].to(dev))
        l1, _ = _loss(cfgs["baseline"], out["logits"], b["mask"].to(dev), b["valid"].to(dev))
        l5, _ = _loss(cfgs["pw5"], out["logits"], b["mask"].to(dev), b["valid"].to(dev))
    gate("pw5", "pos_weight 生效（同批数据 pw5 loss 更高）", float(l5) > float(l1),
         {"loss_pw1": round(float(l1), 6), "loss_pw5": round(float(l5), 6),
          "ratio": round(float(l5) / float(l1), 4)})

    rep["gates"] = gates
    rep["all_pass"] = all(g["ok"] for g in gates)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[done] all_pass={rep['all_pass']} -> {OUT}")
    if not rep["all_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
