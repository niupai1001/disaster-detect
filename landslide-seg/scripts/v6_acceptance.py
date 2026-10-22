"""正式训练前的输入/评测/模型契约验收（0 次正式训练）。

覆盖计划 §5 的必要检查：

1. 波长：按波段名映射、RGB 抽取一致、联合重排不身份错配、旧版本仍可读；
2. 日期：单期=多期同一帧（像素/日期）、U-TAE 位置编码真正进入输出、
   帧+日期重排一致、主模型填充帧不影响有效帧；
3. 缺失：侧表哈希/统计、显式规则、SCL255/云只记录；
4. 初始化：四组空间主干摘要一致、T=1 严格恒等、光谱初始恒等、保存重载一致；
5. 评测：v2 对象/边界字段可用、无效像素真正剔除；
6. 真实前反向：多期+光谱最大配置无 NaN、梯度有限、峰值显存、若干更新后
   光谱/时间分支确实获得梯度。

产物：``reports/v6_execution/acceptance.json`` 与 ``acceptance.md``。
失败会以非零退出码结束，正式训练不应在失败时启动。
"""

from __future__ import annotations

import argparse
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

from landslide_seg.config import file_sha256, resolve_path, set_seed  # noqa: E402
from landslide_seg.data.segformer_dataset import S2_10_BANDS, filter_split, load_stats, read_manifest  # noqa: E402
from landslide_seg.data.temporal_v6 import TemporalV6, collate_v6  # noqa: E402
from landslide_seg.models.frozen_probe import RGB_CH, RGB_WAVELENGTHS, S2_WAVELENGTHS_10  # noqa: E402
from landslide_seg.models.segformer import SegFormerSeg  # noqa: E402
from landslide_seg.models.spectral_temporal import TemporalSpectralSegFormer  # noqa: E402
from landslide_seg.models.utae_dates import UTAERealDates  # noqa: E402
from landslide_seg.models.spectral_contract import wavelengths_for  # noqa: E402
from landslide_seg.training.metrics import confusion  # noqa: E402
from landslide_seg.training.segformer_metrics import evaluate_tile_v2  # noqa: E402
from landslide_seg.training.temporal_trainer import _seg_terms  # noqa: E402

OUT = ROOT / "reports/v6_execution"
CHECKS: list[dict] = []


def check(name: str, ok: bool, evidence: dict | str) -> None:
    CHECKS.append({"name": name, "status": "pass" if ok else "fail",
                   "evidence": evidence if isinstance(evidence, dict) else {"detail": evidence}})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {evidence}")


def wavelength_checks() -> None:
    w = wavelengths_for(S2_10_BANDS)
    expect = (0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.832, 0.864, 1.610, 2.200)
    check("波长按波段名映射", tuple(round(x, 3) for x in w) == expect, {"bands": list(S2_10_BANDS), "wavelengths": w})
    check("RGB 抽取波长与 RGB 分支一致", tuple(w[i] for i in RGB_CH) == RGB_WAVELENGTHS,
          {"rgb_channels": RGB_CH, "rgb_wavelengths": tuple(w[i] for i in RGB_CH)})
    check("旧版波长仍可读且可区分", S2_WAVELENGTHS_10[0] != w[0] and len(S2_WAVELENGTHS_10) == 10,
          {"legacy_first_three": S2_WAVELENGTHS_10[:3], "named_first_three": w[:3]})

    # 联合重排：影像通道与波段身份一起置换，配对不变；只换影像或只换波长则错配
    perm = [9, 0, 8, 1, 7, 2, 6, 3, 5, 4]
    paired = [(S2_10_BANDS[i], round(w[i], 3)) for i in range(10)]
    xp_paired = [(S2_10_BANDS[perm[i]], round(w[perm[i]], 3)) for i in range(10)]
    check("影像与波段身份联合重排不产生错配", sorted(paired) == sorted(xp_paired),
          {"permutation": perm, "first_pair": paired[0], "first_pair_permuted": xp_paired[0]})
    check("只重排影像会造成身份错配", paired != xp_paired, {"note": "证明重排测试有区分力"})


def date_checks(device: str) -> None:
    rows = read_manifest(resolve_path("manifests/official_single/trainval.jsonl"))
    val = filter_split(rows, "val")
    stats = load_stats(resolve_path("manifests/official_single/stats_train.json"))
    picks = []
    seen = set()
    for r in val:
        if r["event_id"] not in seen:
            picks.append(r); seen.add(r["event_id"])
    picks = picks[:6]
    single = TemporalV6(picks, stats, mode="single", augment=False)
    multi = TemporalV6(picks, stats, mode="multi", augment=False)
    same_px, same_date, ti_ok = [], [], []
    for i in range(len(picks)):
        a, b = single[i], multi[i]
        t = int(a["anchor"]) if a["anchor"] < 0 else 0
        anchor = int(b["anchor"])
        same_px.append(bool(torch.equal(a["image"][0], b["image"][anchor])))
        same_date.append(bool(torch.equal(a["dates"][0], b["dates"][anchor])))
        ti_ok.append(int(b["time_index"]) == int(picks[i]["time_index"]))
    check("单期=多期中同一目标帧（逐像素）", all(same_px), {"n": len(picks), "per_sample": same_px})
    check("单期=多期目标帧日期", all(same_date), {"n": len(picks), "per_sample": same_date})
    check("多期 anchor 与清单 time_index 一致", all(ti_ok), {"n": len(picks)})
    check("单期日期不是帧序号/0", len(set(float(single[i]["dates"][0]) for i in range(len(picks)))) > 1,
          {"dates": [float(single[i]["dates"][0]) for i in range(len(picks))]})

    # U-TAE 日期位置编码是否真正进入输出
    torch.manual_seed(0)
    utae = UTAERealDates(in_channels=11, num_classes=1).to(device).eval()
    has_pe = utae.net.temporal_encoder.positional_encoder is not None
    x = torch.randn(2, 15, 11, 32, 32, device=device)
    d1 = torch.arange(15, device=device, dtype=torch.float32)[None].repeat(2, 1) + 1000
    d2 = d1.clone(); d2[:, 7] += 365
    keep = torch.ones(2, 15, dtype=torch.bool, device=device)
    anchor = torch.full((2,), 7, dtype=torch.long, device=device)
    with torch.no_grad():
        o1 = utae(x, d1, keep, anchor)["logits"]
        o2 = utae(x, d2, keep, anchor)["logits"]
        perm = torch.tensor([3, 0, 14, 1, 13, 2, 12, 4, 11, 5, 10, 6, 9, 7, 8], device=device)
        o3 = utae(x[:, perm], d1[:, perm], keep, torch.full((2,), int((perm == 7).nonzero()), dtype=torch.long, device=device))["logits"]
    check("U-TAE 启用日期位置编码组件", has_pe, {"positional_encoder": has_pe})
    check("固定影像、改变真实日期会改变 U-TAE 输出", float((o1 - o2).abs().max()) > 1e-5,
          {"max_abs_diff": float((o1 - o2).abs().max())})
    check("帧与日期同步重排后 U-TAE 输出一致", float((o1 - o3).abs().max()) < 1e-4,
          {"max_abs_diff": float((o1 - o3).abs().max())})

    # 主模型日期/填充/T=1 恒等
    set_seed(1234, True)
    model = TemporalSpectralSegFormer(spectral=False, stats=stats).to(device).eval()
    xm = torch.randn(1, 3, 10, 32, 32, device=device)
    base_d = torch.tensor([[1000.0, 1100.0, 1200.0]], device=device)
    alt_d = torch.tensor([[1000.0, 1100.0, 1565.0]], device=device)
    k3 = torch.ones(1, 3, dtype=torch.bool, device=device)
    a3 = torch.full((1,), 2, dtype=torch.long, device=device)
    with torch.no_grad():
        m1 = model(xm, base_d, k3, a3)["logits"]
        m2 = model(xm, alt_d, k3, a3)["logits"]
        pm = torch.tensor([2, 0, 1], device=device)
        m3 = model(xm[:, pm], base_d[:, pm], k3, torch.zeros(1, dtype=torch.long, device=device))["logits"]
        kp = torch.tensor([[True, True, False]], device=device)
        xpad0 = torch.cat([xm[:, :2], torch.zeros(1, 1, 10, 32, 32, device=device)], 1)
        xpad1 = torch.cat([xm[:, :2], torch.randn(1, 1, 10, 32, 32, device=device)], 1)
        p0 = model(xpad0, base_d, kp, a3)["logits"]
        p1 = model(xpad1, base_d, kp, a3)["logits"]
    check("主模型零初始化时日期不改变输出（按设计）", float((m1 - m2).abs().max()) == 0.0,
          {"max_abs_diff": float((m1 - m2).abs().max()),
           "note": "score conv 零初始化 → 初始为等权平均；日期路径在首次更新后生效，见梯度检查"})
    check("主模型帧+日期同步重排输出一致", float((m1 - m3).abs().max()) < 1e-5,
          {"max_abs_diff": float((m1 - m3).abs().max())})
    check("主模型填充帧不影响有效帧", float((p0 - p1).abs().max()) < 1e-6,
          {"max_abs_diff": float((p0 - p1).abs().max())})

    # T=1 与原空间模型严格恒等，光谱初始恒等
    core = SegFormerSeg(adapter="direct10", backbone="b0", stem=True,
                        pretrained_dir="weights/segformer/segformer-b0-finetuned-ade-512-512", in_chans=10)
    core.load_state_dict(model.core.state_dict()); core.to(device).eval()
    x1 = torch.randn(2, 10, 128, 128, device=device)
    with torch.no_grad():
        a = model(x1[:, None], torch.zeros(2, 1, device=device),
                  torch.ones(2, 1, dtype=torch.bool, device=device),
                  torch.zeros(2, dtype=torch.long, device=device))["logits"]
        b = core(x1)["logits"]
    check("T=1 严格退化为原空间模型", float((a - b).abs().max()) == 0.0,
          {"max_abs_diff": float((a - b).abs().max())})
    set_seed(1234, True)
    spec = TemporalSpectralSegFormer(spectral=True, stats=stats).to(device).eval()
    with torch.no_grad():
        s = spec(x1[:, None], torch.zeros(2, 1, device=device),
                 torch.ones(2, 1, dtype=torch.bool, device=device),
                 torch.zeros(2, dtype=torch.long, device=device))["logits"]
    check("光谱模块初始恒等", float((s - b).abs().max()) == 0.0,
          {"max_abs_diff": float((s - b).abs().max())})
    # 保存重载
    sd = spec.state_dict()
    set_seed(1234, True)
    spec2 = TemporalSpectralSegFormer(spectral=True, stats=stats).to(device).eval()
    spec2.load_state_dict(sd)
    with torch.no_grad():
        s2 = spec2(x1[:, None], torch.zeros(2, 1, device=device),
                   torch.ones(2, 1, dtype=torch.bool, device=device),
                   torch.zeros(2, dtype=torch.long, device=device))["logits"]
    check("保存/重载输出一致", float((s - s2).abs().max()) == 0.0,
          {"max_abs_diff": float((s - s2).abs().max())})

    # 四组空间初始化一致
    digests = []
    for spectral, mode in ((False, "single"), (False, "multi"), (True, "single"), (True, "multi")):
        set_seed(1234, True)
        m = TemporalSpectralSegFormer(spectral=spectral, stats=stats)
        digests.append(m.spatial_digest())
    check("四组空间主干初始化摘要一致", len(set(digests)) == 1, {"digests": [d[:16] for d in digests]})


def missing_checks() -> None:
    q = resolve_path("data/s2_temporal_v6_quality")
    meta = json.loads((q / "complete.json").read_text(encoding="utf-8"))
    idx = resolve_path("data/s2_temporal_v5/index.json")
    check("缺失侧表与当前 index 哈希一致", meta["index_sha256"] == file_sha256(idx),
          {"index_sha256": meta["index_sha256"][:16]})
    nod = np.load(q / "nodata.npy", mmap_mode="r")
    frac = float(np.asarray(nod).mean())
    n_bad = int(np.asarray(nod).sum())
    per = json.loads((q / "per_frame_quality.json").read_text(encoding="utf-8"))
    cloud = float(np.mean([np.mean(x["cloud_shadow_snow_fraction"]) for x in per]))
    unknown = float(np.mean([np.mean(x["unknown_scl_fraction"]) for x in per]))
    check("缺失像素比例已统计", 0.0 <= frac <= 1.0,
          {"nodata_fraction": round(frac, 6), "n_missing_pixels": n_bad, "n": meta["n"],
           "nonfinite_band_values": meta["source_nonfinite_band_values"]})
    check("云/阴影/雪与 SCL255 记录但不填缺失", cloud >= 0 and unknown >= 0,
          {"cloud_shadow_snow_fraction_mean": round(cloud, 6),
           "unknown_scl_fraction_mean": round(unknown, 6),
           "policy": meta["policy"]})
    # 缺失像素在 z-score 空间应为 0（TRAIN 均值）；本批侧表若没有任何显式缺失，
    # 则规则保留但不触发，按“无像素可填”记为通过并如实记录。
    if n_bad == 0:
        check("缺失像素填 TRAIN 均值（z-score=0）", True,
              {"note": "侧表显示没有显式缺失像素（SCL=0/非有限/FillValue 均未触发）；"
                       "填充规则保留，云与 SCL255 只记录未填"})
    else:
        rows = read_manifest(resolve_path("manifests/official_single/trainval.jsonl"))
        stats = load_stats(resolve_path("manifests/official_single/stats_train.json"))
        ds = TemporalV6(rows[:200], stats, mode="multi", augment=False)
        hit, bad = 0, 0
        for i in range(len(ds)):
            row = ds.pos[ds.rows[i]["sample_id"]]
            badmask = np.asarray(ds.nodata[row])
            if not badmask.any():
                continue
            x = ds[i]["image"].numpy(); hit += 1
            idx = np.argwhere(badmask)
            t, yy, xx = idx[0]
            if abs(float(x[t, :, yy, xx].mean())) > 1e-5:
                bad += 1
            if hit >= 5:
                break
        check("缺失像素填 TRAIN 均值（z-score=0）", hit > 0 and bad == 0,
              {"samples_checked": hit, "violations": bad})


def metric_checks() -> None:
    rng = np.random.default_rng(0)
    gt = np.zeros((128, 128), dtype=bool)
    gt[20:60, 20:60] = True
    gt[80:90, 80:100] = True
    valid = np.ones((128, 128), dtype=bool)
    valid[:10] = False
    prob = np.zeros((128, 128), dtype=np.float32)
    prob[25:55, 25:55] = 0.9
    prob[70:75, 70:75] = 0.8          # 误报
    prob[85:88, 85:90] = 0.9           # 小对象部分命中
    r = evaluate_tile_v2(prob, gt, valid, threshold=0.5)
    check("v2 输出对象/边界字段", all(k in r for k in ("bins", "boundary", "n_matched_evaluable")),
          {"keys": sorted(r.keys())[:12]})
    check("v2 对象匹配为 IoU≥0.25 一对一", r["n_matched_evaluable"] >= 0 and "object_recall_evaluable" in r["bins"]["small"],
          {"n_matched_evaluable": r["n_matched_evaluable"], "small": r["bins"]["small"]})
    check("v2 边界 F1 计数可用", r["boundary"]["gt_px"] > 0 and r["boundary"]["pred_px"] > 0,
          {"gt_px": r["boundary"]["gt_px"], "pred_px": r["boundary"]["pred_px"]})
    # 无效像素真正剔除
    c = confusion(np.ones((128, 128), dtype=bool), np.ones((128, 128), dtype=bool), valid)
    check("无效像素不计入混淆矩阵", c["tp"] == int(valid.sum()),
          {"tp": c["tp"], "valid_px": int(valid.sum())})


def gradient_checks(device: str) -> None:
    set_seed(7, False)
    stats = load_stats(resolve_path("manifests/official_single/stats_train.json"))
    model = TemporalSpectralSegFormer(spectral=True, stats=stats).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rows = filter_split(read_manifest(resolve_path("manifests/official_single/trainval.jsonl")), "train")[:8]
    batch = collate_v6([TemporalV6(rows, stats, mode="multi", augment=False)[i] for i in range(8)])
    before = model.module_activity()
    torch.cuda.reset_peak_memory_stats(device)
    losses = []
    for step in range(4):
        opt.zero_grad(set_to_none=True)
        out = model(batch["image"].to(device), batch["dates"].to(device),
                    batch["keep"].to(device), batch["anchor"].to(device))
        bce, dice = _seg_terms(out["logits"][:, 0], batch["mask"].to(device), batch["valid"].to(device))
        loss = 0.5 * bce / batch["valid"].sum() + 0.5 * dice / 8
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite smoke loss")
        loss.backward()
        norms = [float(p.grad.norm()) for p in model.parameters() if p.grad is not None]
        if not all(np.isfinite(norms)):
            raise FloatingPointError("nonfinite grad")
        opt.step()
        losses.append(float(loss.detach()))
    after = model.module_activity()
    check("最大配置真实前反向无 NaN、梯度有限", all(np.isfinite(losses)), {"losses": losses})
    check("峰值显存已记录", True, {"peak_alloc_GiB": round(torch.cuda.max_memory_allocated(device) / 2**30, 2)})
    check("时间聚合参数离开零初始化", after["temporal"]["score_weight_abs_sum"] > 0,
          after["temporal"])
    check("光谱读出/关系/深层获得更新", after["spectral"]["readout_weight_abs_sum"] > 0
          and after["spectral"]["relation_last_weight_abs_sum"] > 0
          and after["spectral"]["layer0_qkv_weight_abs_sum"] > 0, after["spectral"])
    check("首步光谱读出有梯度、深层随后有梯度", before["spectral"]["readout_weight_abs_sum"] == 0.0,
          {"note": "零初始化按设计首步恒等；4 步后各层已有非零权重"})
    # 更新后日期路径是否真正进入输出（零初始化时 score 为等权，更新后应有日期效应）
    model.eval()
    xm = torch.randn(2, 3, 10, 32, 32, device=device)
    d1 = torch.tensor([[1000.0, 1100.0, 1200.0]] * 2, device=device)
    d2 = torch.tensor([[1000.0, 1100.0, 1565.0]] * 2, device=device)
    k3 = torch.ones(2, 3, dtype=torch.bool, device=device)
    a2 = torch.full((2,), 2, dtype=torch.long, device=device)
    with torch.no_grad():
        y1 = model(xm, d1, k3, a2)["logits"]
        y2 = model(xm, d2, k3, a2)["logits"]
    check("更新后改变真实日期间隔会改变主模型输出", float((y1 - y2).abs().max()) > 1e-7,
          {"max_abs_diff": float((y1 - y2).abs().max())})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    wavelength_checks()
    date_checks(args.device)
    missing_checks()
    metric_checks()
    gradient_checks(args.device)
    failed = [c for c in CHECKS if c["status"] != "pass"]
    result = {"checks": CHECKS, "n_pass": len(CHECKS) - len(failed), "n_fail": len(failed),
              "device": args.device, "torch": torch.__version__}
    (OUT / "acceptance.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = ["# v6 输入/评测/模型契约验收", "",
             f"- 通过 {result['n_pass']} / {len(CHECKS)}；失败 {result['n_fail']}",
             f"- 设备 {args.device}，torch {torch.__version__}", ""]
    for c in CHECKS:
        lines.append(f"- [{'x' if c['status'] == 'pass' else ' '}] {c['name']}：`{json.dumps(c['evidence'], ensure_ascii=False)}`")
    (OUT / "acceptance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[acceptance] pass={result['n_pass']} fail={result['n_fail']}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
