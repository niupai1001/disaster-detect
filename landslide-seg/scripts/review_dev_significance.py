"""DEV 结果差异的显著性核对。

目的:把 v4r1/v4r2/v4r3 里"看起来有先后"的 DEV 均值,换成带不确定性的比较。
只读 per_tile_best_val.csv(每图块 tp/fp/fn),不重新推理。

做法:
  1. 按事件重算 pooled IoU 与 dev_event_mean,验证与 CSV 里记录的数值一致;
  2. 对每条比较做配对 bootstrap(同一批图块、同一事件分层),5000 次,
     统计 dev_event_mean 差值的分布 -> 95% 区间 + P(差值>0);
  3. 额外报告逐事件符号一致性(2 个事件里几个同向)。

注意:DEV 只有 2 个事件,事件级有效样本量 = 2。分层 bootstrap 会把
"图块数"当成样本量,置信区间偏窄;它的用途是判断"这点差距是不是可能
纯由图块抽样造成",而不是替代多种子实验。
"""

from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_MD = ROOT / "reports" / "v4_dev_significance_2026-10-20.md"
OUT_JSON = ROOT / "reports" / "v4_dev_significance_2026-10-20.json"

RUNS = {
    # 标签: per_tile_best_val.csv 路径
    "detailres_n025": "outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_detailres_n025_u3600_s1234/1234/eval/per_tile_best_val.csv",
    "detailres_n050": "outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_detailres_n050_u3600_s1234/1234/eval/per_tile_best_val.csv",
    "detailres_n075": "outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_detailres_n075_u3600_s1234/1234/eval/per_tile_best_val.csv",
    "detailres_n100": "outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_detailres_n100_u3600_s1234/1234/eval/per_tile_best_val.csv",
    "standard_n025": "outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_n025_u3600_s1234/1234/eval/per_tile_best_val.csv",
    "standard_n100": "outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_n100_u3600_s1234/1234/eval/per_tile_best_val.csv",
    "probe_frozen_best": "outputs/segformer/s2_10band_v4r2_probe/segformer_b0_direct10_frozen_best_decodeprobe_n100_u3600_s1234/1234/eval/per_tile_best_val.csv",
    "probe_frozen_last": "outputs/segformer/s2_10band_v4r2_probe/segformer_b0_direct10_frozen_last_decodeprobe_n100_u3600_s1234/1234/eval/per_tile_best_val.csv",
    "dofa_upernet": "outputs/segformer/s2_10band_v4r3_dofa/dofa_base_upernet_pyramid_binary_n100_u3600_s1234/1234/eval/per_tile_best_val.csv",
    "low_encoder_lr": "outputs/segformer/s2_10band_v4r3_low_encoder_lr/segformer_b0_direct10_detailres_pretrained_encoder_low_lr_n100_u3600_s1234/1234/eval/per_tile_best_val.csv",
}

COMPARISONS = [
    ("detailres_n075", "detailres_n100", "学习曲线峰值:75% vs 100% 池"),
    ("detailres_n100", "standard_n100", "detailres vs standard(同池 100%)"),
    ("detailres_n050", "detailres_n025", "数据扩容 25% -> 50%"),
    ("detailres_n100", "detailres_n050", "数据扩容 50% -> 100%"),
    ("detailres_n100", "probe_frozen_best", "联合微调 vs 冻结最优编码器探针"),
    ("detailres_n100", "low_encoder_lr", "编码器全 LR vs 低 LR"),
    ("detailres_n100", "dofa_upernet", "SegFormer-B0 vs DOFA+UPerNet"),
    ("detailres_n100", "probe_frozen_last", "联合微调 vs 冻结末态编码器探针"),
]

N_BOOT = 5000
SEED = 20261020


def read_rows(path: Path) -> dict[str, dict]:
    import csv

    rows = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            sid = r["sample_id"]
            rows[sid] = {
                "event": r["event_id"],
                "tp": int(float(r["tp"])),
                "fp": int(float(r["fp"])),
                "fn": int(float(r["fn"])),
            }
    return rows


def iou_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    if denom == 0:
        return float("nan")
    return tp / denom


def pooled(rows: dict[str, dict], sample_ids: list[str]) -> dict[str, float]:
    """返回 {'overall': iou, '<event>': iou, 'event_mean': mean}."""
    per_event: dict[str, list[int]] = {}
    tp = fp = fn = 0
    for sid in sample_ids:
        r = rows[sid]
        tp += r["tp"]
        fp += r["fp"]
        fn += r["fn"]
        per_event.setdefault(r["event"], [0, 0, 0])
        per_event[r["event"]][0] += r["tp"]
        per_event[r["event"]][1] += r["fp"]
        per_event[r["event"]][2] += r["fn"]
    out = {"overall": iou_from_counts(tp, fp, fn)}
    ev_ious = []
    for ev, (a, b, c) in sorted(per_event.items()):
        v = iou_from_counts(a, b, c)
        out[ev] = v
        if not math.isnan(v):
            ev_ious.append(v)
    out["event_mean"] = sum(ev_ious) / len(ev_ious) if ev_ious else float("nan")
    out["n_events"] = len(ev_ious)
    return out


def fixed_step_table() -> dict:
    """从 train_summary.json 的 history 取固定步数的 dev event_mean_iou。"""
    steps = [200, 500, 800, 1200, 2400, 3600]
    rows = {}
    pats = sorted(ROOT.glob("outputs/segformer/s2_10band_v4*/**/train_summary.json"))
    for p in pats:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        name = d.get("experiment") or p.parent.parent.name
        hist = d.get("history") or []
        if not hist:
            continue
        dev = {}
        for x in hist:
            dv = x.get("dev")
            if isinstance(dv, dict) and dv.get("event_mean_iou") is not None:
                dev[int(x["global_step"])] = float(dv["event_mean_iou"])
        if not dev:
            continue
        # 只保留真正跑满 3600 的 run,避免 smoke/validation 混进来
        if max(dev.keys()) < 3600:
            continue
        train_last = float(hist[-1].get("train_clean", {}).get("pooled", {}).get("iou") or float("nan"))
        lag = [v for s, v in dev.items() if s >= 600]
        rows[name] = {
            "dev": {s: dev[s] for s in steps if s in dev},
            "best": float(d.get("best_val_iou") or max(dev.values())),
            "best_step": int(d.get("best_step") or max(dev, key=dev.get)),
            "train_last": train_last,
            "plateau_mean": (sum(lag) / len(lag)) if lag else float("nan"),
            "plateau_sd": (statistics.pstdev(lag) if len(lag) > 1 else float("nan")),
            "n_evals": len(dev),
        }
    # 按固定步数 500 的 DEV 降序排,方便读
    ordered = dict(sorted(rows.items(), key=lambda kv: -kv[1]["dev"].get(500, -1)))
    return {"steps": steps, "rows": ordered}


def main() -> None:
    data: dict[str, dict[str, dict]] = {}
    missing = []
    for tag, rel in RUNS.items():
        p = ROOT / rel
        if not p.exists():
            missing.append(f"{tag}: {rel}")
            continue
        data[tag] = read_rows(p)

    # 以 detailres_n100 的样本集为公共参照,并要求所有 run 完全一致
    ref_tag = "detailres_n100"
    ref_ids = sorted(data[ref_tag].keys())
    for tag, rows in data.items():
        if set(rows.keys()) != set(ref_ids):
            raise SystemExit(f"[fatal] {tag} 的图块集合与参照不一致: {len(rows)} vs {len(ref_ids)}")

    single = {}
    for tag, rows in data.items():
        single[tag] = pooled(rows, ref_ids)

    rng = random.Random(SEED)

    # 事件分层索引,保证每次重采样两个事件的图块数保持不变
    by_event: dict[str, list[str]] = {}
    for sid in ref_ids:
        by_event.setdefault(data[ref_tag][sid]["event"], []).append(sid)
    for ev in by_event:
        by_event[ev].sort()

    comp_rows = []
    for a, b, desc in COMPARISONS:
        if a not in data or b not in data:
            continue
        obs = single[a]["event_mean"] - single[b]["event_mean"]
        diffs = []
        for _ in range(N_BOOT):
            ids = []
            for ev, pool_ids in by_event.items():
                n = len(pool_ids)
                ids.extend(rng.choices(pool_ids, k=n))
            ma = pooled(data[a], ids)["event_mean"]
            mb = pooled(data[b], ids)["event_mean"]
            diffs.append(ma - mb)
        diffs.sort()
        lo = diffs[int(0.025 * N_BOOT)]
        hi = diffs[int(0.975 * N_BOOT) - 1]
        p_pos = sum(1 for d in diffs if d > 0) / N_BOOT
        # 逐事件符号一致性
        signs = []
        for ev in by_event:
            da = single[a].get(ev, float("nan")) - single[b].get(ev, float("nan"))
            if not math.isnan(da):
                signs.append((ev, da))
        consistent = sum(1 for _, d in signs if d > 0)
        zero_cross = lo <= 0 <= hi
        comp_rows.append(
            {
                "a": a,
                "b": b,
                "desc": desc,
                "obs": obs,
                "ci_lo": lo,
                "ci_hi": hi,
                "p_positive": p_pos,
                "sign_consistent": f"{consistent}/{len(signs)}",
                "per_event": {ev: d for ev, d in signs},
                "excludes_zero": not zero_cross,
            }
        )

    # ---- 输出 ----
    lines = []
    lines.append("# DEV 差异显著性核对(v4 三批实验)\n")
    lines.append(
        "数据源:各 run 的 `eval/per_tile_best_val.csv`(每图块 tp/fp/fn),"
        f"公共参照集 {len(ref_ids)} 块(`detailres_n100`,10 个 run 图块集合完全一致才继续)。"
    )
    lines.append(
        f"配对分层 bootstrap {N_BOOT} 次,固定种子 {SEED};按事件分层,不跨事件打乱。\n"
    )

    lines.append("## 1. 逐 run 重算(校验记录的数值)\n")
    lines.append("| run | DEV 均值 | Hiroshima | Thrissur | 总体 pooled | 事件数 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for tag in RUNS:
        if tag not in single:
            continue
        s = single[tag]
        evs = {k: v for k, v in s.items() if k not in ("overall", "event_mean", "n_events")}
        h = evs.get("hiroshima", float("nan"))
        t = evs.get("thrissur", float("nan"))
        lines.append(
            f"| `{tag}` | {s['event_mean']:.4f} | {h:.4f} | {t:.4f} | {s['overall']:.4f} | {s['n_events']} |"
        )

    lines.append("\n## 2. 配对比较\n")
    lines.append("| 比较 | Δ(DEV 均值) | 95% 区间 | P(Δ>0) | 逐事件同向 | 区间跨 0 |")
    lines.append("|---|---:|---|---:|---:|---|")
    order = {c[0]: i for i, c in enumerate(COMPARISONS)}
    comp_rows.sort(key=lambda r: -abs(r["obs"]))
    for r in comp_rows:
        lines.append(
            f"| {r['desc']} | {r['obs']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | "
            f"{r['p_positive']:.3f} | {r['sign_consistent']} | {'是' if not r['excludes_zero'] else '否'} |"
        )

    lines.append("\n## 3. 逐事件差值\n")
    evs = sorted({ev for r in comp_rows for ev in r["per_event"]})
    lines.append("| 比较 | " + " | ".join(evs) + " |")
    lines.append("|---|" + "---:|" * len(evs))
    for r in comp_rows:
        cells = " | ".join(f"{r['per_event'].get(ev, float('nan')):+.4f}" for ev in evs)
        lines.append(f"| {r['desc']} | {cells} |")

    # ---- 固定步数对照:消除"在 36 个 checkpoint 里挑最大"带来的乐观偏差 ----
    fixed = fixed_step_table()
    lines.append("\n## 4. 固定步数 DEV 均值(消除 max-selection 偏差)\n")
    lines.append(
        "上面的 `best` 是在整条轨迹里挑最大值,而 DEV 同时也是选择集,"
        "所以那个数字系统性偏高。下表直接取固定步数的 DEV 事件均值,"
        "同一列可以横向比。\n"
    )
    if fixed["steps"]:
        hdr = (
            "| run | " + " | ".join(str(s) for s in fixed["steps"])
            + " | 平台均值(≥600步) | 平台 sd | best | best@步 | best−平台均值 | 训练 IoU@末尾 |"
        )
        lines.append(hdr)
        lines.append("|---|" + "---:|" * (len(fixed["steps"]) + 6))
        for name, row in fixed["rows"].items():
            cells = " | ".join(
                f"{row['dev'].get(s, float('nan')):.4f}" if s in row["dev"] else "-" for s in fixed["steps"]
            )
            lines.append(
                f"| `{name}` | {cells} | {row['plateau_mean']:.4f} | {row['plateau_sd']:.4f} | "
                f"{row['best']:.4f} | {row['best_step']} | {row['best'] - row['plateau_mean']:+.4f} | "
                f"{row['train_last']:.4f} |"
            )
        lines.append("")
        lines.append("同一固定步数下的极差(最大 − 最小),用于判断这一列能不能区分模型:\n")
        lines.append("| 步数 | 最小值 | 最大值 | 极差 |")
        lines.append("|---:|---:|---:|---:|")
        for s in fixed["steps"]:
            vals = [r["dev"][s] for r in fixed["rows"].values() if s in r["dev"]]
            if len(vals) < 2:
                continue
            lines.append(f"| {s} | {min(vals):.4f} | {max(vals):.4f} | {max(vals) - min(vals):.4f} |")

    if missing:
        lines.append("\n## 5. 缺失 run\n")
        for m in missing:
            lines.append(f"- {m}")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_JSON.write_text(
        json.dumps(
            {
                "n_tiles": len(ref_ids),
                "n_boot": N_BOOT,
                "seed": SEED,
                "single": {k: {kk: (None if isinstance(vv, float) and math.isnan(vv) else vv) for kk, vv in v.items()} for k, v in single.items()},
                "pairwise": comp_rows,
                "fixed_step": fixed,
                "missing": missing,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[done] -> {OUT_MD}")
    print(f"[done] -> {OUT_JSON}")
    for r in comp_rows:
        print(
            f"[cmp] {r['desc']}: {r['obs']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] "
            f"sign={r['sign_consistent']} excludes0={r['excludes_zero']}"
        )


if __name__ == "__main__":
    main()
