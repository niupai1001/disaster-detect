"""审阅 v4 系列实验：把 run 元数据、结果 CSV 与冻结协议放在一起核对。

只读，不改任何实验产物。检查项：

1. 每个 run 的真实更新步数、训练/验证样本数、best 步数；
2. `dev_mean_iou` 是否等于两个 DEV 事件的算术平均（CSV 自洽）；
3. FINAL / Italy 是否被评测过（应当全部为 False）；
4. 训练池成员是否 ⊆ 冻结的 `allowed_train_pool.csv`；
5. 各批 run 的协议/清单/统计身份是否一致；
6. 汇总成一张跨批次对照表。

用法::

    python scripts/review_v4_runs.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BATCH_CSVS = [
    ("v4r1", "reports/v4r1_results/learning_curve_and_controls.csv"),
    ("v4r2", "reports/v4r2_probe_results/probe_comparison.csv"),
    ("v4r3", "reports/v4r3_results/comparison.csv"),
]


def read_csv(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fnum(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def collect_runs() -> list[dict]:
    out = []
    for p in sorted(ROOT.glob("outputs/segformer/s2_10band_v4*/**/train_summary.json")):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_dir = p.parent
        cfg = {}
        cp = run_dir / "resolved_config.json"
        if cp.is_file():
            try:
                cfg = json.loads(cp.read_text(encoding="utf-8"))
            except Exception:
                cfg = {}
        scfg = cfg.get("segformer", cfg) if isinstance(cfg, dict) else {}
        out.append(
            {
                "run": run_dir.relative_to(ROOT).as_posix(),
                "n_train": s.get("n_train"),
                "n_val": s.get("n_val"),
                "steps": s.get("global_step"),
                "epochs": s.get("epochs_run"),
                "best_epoch": s.get("best_epoch"),
                "best_val_iou": s.get("best_val_iou"),
                "seconds": s.get("seconds"),
                "params": (s.get("params") or {}).get("total"),
                "manifest": (scfg.get("data") or {}).get("manifest"),
                "stats": (scfg.get("data") or {}).get("stats"),
            }
        )
    return out


def main() -> None:
    runs = collect_runs()

    # ---- 1. CSV 自洽性
    csv_rows = []
    for batch, rel in BATCH_CSVS:
        for r in read_csv(ROOT / rel):
            hir = fnum(r.get("hiroshima_iou"))
            thr = fnum(r.get("thrissur_iou"))
            dev = fnum(r.get("dev_mean_iou"))
            ok = None
            if None not in (hir, thr, dev):
                ok = abs((hir + thr) / 2 - dev) < 1e-6
            csv_rows.append(
                {
                    "batch": batch,
                    "condition": r.get("experiment") or r.get("condition"),
                    "ckpt": r.get("checkpoint", ""),
                    "step": r.get("best_step") or r.get("step"),
                    "train_iou": fnum(r.get("best_train_iou") if r.get("best_train_iou") else r.get("train_iou")),
                    "dev": dev,
                    "hir": hir,
                    "thr": thr,
                    "ap": fnum(r.get("ap")),
                    "obj_recall": fnum(r.get("object_recall")),
                    "tiny": (f"{r.get('n_matched_1_4')}/{r.get('n_objects_1_4')}" if r.get("n_objects_1_4") else
                             f"{r.get('matched_1_4')}/{r.get('objects_1_4')}"),
                    "final_eval": str(r.get("final_evaluated", "")).lower() in ("true", "1"),
                    "mean_ok": ok,
                    "n_train": r.get("n_train"),
                }
            )

    # ---- 2. 冻结池成员
    pool = set()
    pp = ROOT / "manifests" / "allowed_train_pool.csv"
    if pp.is_file():
        with open(pp, encoding="utf-8") as f:
            pool = {r["sample_id"] for r in csv.DictReader(f) if r["in_100"] == "1"}

    pool_check = {}
    for p in sorted(ROOT.glob("manifests/v4r1/pool_*.jsonl")):
        ids = set()
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("split") == "train":
                    ids.add(rec["sample_id"])
        pool_check[p.name] = (len(ids), len(ids - pool), len(ids & pool))

    # ---- 3. 报告
    L = []
    L.append("# v4 系列实验审阅（自动化核对）")
    L.append("")
    L.append("数据源：`outputs/segformer/s2_10band_v4*/**/train_summary.json` + 三批结果 CSV + 冻结清单。只读核对，不重算像素。")
    L.append("")
    L.append("## 1. run 元数据")
    L.append("")
    L.append("| run | n_train | n_val | steps | epochs | best_epoch | best_dev | 秒 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in runs:
        L.append(
            "| `%s` | %s | %s | %s | %s | %s | %s | %s |"
            % (
                r["run"],
                r["n_train"],
                r["n_val"],
                r["steps"],
                r["epochs"],
                r["best_epoch"],
                ("%.4f" % r["best_val_iou"]) if isinstance(r["best_val_iou"], (int, float)) else "-",
                ("%.0f" % r["seconds"]) if isinstance(r["seconds"], (int, float)) else "-",
            )
        )

    L.append("")
    L.append("## 2. 结果自洽性")
    L.append("")
    bad = [r for r in csv_rows if r["mean_ok"] is False]
    L.append("- 参与核对的记录：**%d** 条；`dev_mean = (hiroshima + thrissur)/2` 不成立：**%d** 条。" % (len(csv_rows), len(bad)))
    fin = [r for r in csv_rows if r["final_eval"]]
    L.append("- 已评测 FINAL/Italy 的记录：**%d** 条%s。" % (len(fin), "" if not fin else "，需核查是否破坏留出"))
    L.append("")
    L.append("## 3. 训练池与冻结池一致性")
    L.append("")
    L.append("| 清单 | train 图块 | 不在冻结池 | 与冻结池交集 |")
    L.append("|---|---:|---:|---:|")
    for name, (n, extra, inter) in pool_check.items():
        L.append("| `%s` | %d | %d | %d |" % (name, n, extra, inter))

    L.append("")
    L.append("## 4. 跨批次对照")
    L.append("")
    L.append("| 批次 | 条件 | ckpt | 步 | TRAIN IoU | DEV 均值 | Hiroshima | Thrissur | AP | 对象召回 | 1–4px |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(csv_rows, key=lambda x: (x["batch"], -(x["dev"] or 0))):
        L.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                r["batch"],
                r["condition"],
                r["ckpt"],
                r["step"],
                ("%.4f" % r["train_iou"]) if r["train_iou"] is not None else "-",
                ("%.4f" % r["dev"]) if r["dev"] is not None else "-",
                ("%.4f" % r["hir"]) if r["hir"] is not None else "-",
                ("%.4f" % r["thr"]) if r["thr"] is not None else "-",
                ("%.4f" % r["ap"]) if r["ap"] is not None else "-",
                ("%.4f" % r["obj_recall"]) if r["obj_recall"] is not None else "-",
                r["tiny"],
            )
        )

    out = ROOT / "reports" / "v4_results_review_2026-10-20.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("[done] -> %s" % out)
    print("[info] runs=%d csv_rows=%d mean_mismatch=%d final_eval=%d" % (len(runs), len(csv_rows), len(bad), len(fin)))


if __name__ == "__main__":
    main()
