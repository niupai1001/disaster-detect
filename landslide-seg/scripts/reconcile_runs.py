"""P0：历史 run 预算对账。

目的：回答"为什么历史是 3,000 steps、新组是 3,600 steps"这类口径冲突。
只读 ``outputs/segformer/**/run_meta.json`` 与 ``train_summary.json``，不改任何产物。

输出::

    reports/P0_run_reconciliation.csv
    reports/P0_run_reconciliation.md

用法::

    python scripts/reconcile_runs.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


FIELDS = [
    "data_version",
    "experiment",
    "seed",
    "n_train",
    "n_val",
    "batch_size",
    "steps_per_epoch",
    "epochs_cfg",
    "epochs_run",
    "total_steps",
    "global_step",
    "encoder_lr",
    "head_lr",
    "amp",
    "best_epoch",
    "best_val_iou",
    "seconds",
    "seconds_per_epoch",
    "peak_alloc_mib",
    "params_total",
    "sampling_mode",
    "manifest_sha256",
]


def _get(cfg: dict, *keys, default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def collect(root: Path) -> list[dict]:
    rows: list[dict] = []
    for meta_path in sorted(root.glob("outputs/segformer/**/run_meta.json")):
        run_dir = meta_path.parent
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cfg = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
        summary: dict = {}
        sp = run_dir / "train_summary.json"
        if sp.is_file():
            try:
                summary = json.loads(sp.read_text(encoding="utf-8"))
            except Exception:
                summary = {}

        scfg = cfg.get("segformer", cfg)  # resolved_config 外层包一层 segformer
        n_train = meta.get("n_train") or summary.get("n_train")
        batch = _get(scfg, "train", "batch_size")
        spe = meta.get("steps_per_epoch")
        if spe is None and n_train and batch:
            spe = math.ceil(n_train / batch)
        total = meta.get("total_steps") or summary.get("global_step")
        epochs_cfg = _get(scfg, "train", "epochs")
        epochs_run = summary.get("epochs_run") or (total // spe if (total and spe) else None)

        parts = run_dir.parts
        version = meta.get("data_version", "?")
        rows.append(
            {
                "data_version": version,
                "experiment": meta.get("experiment", parts[-3] if len(parts) >= 3 else "?"),
                "seed": meta.get("seed"),
                "n_train": n_train,
                "n_val": meta.get("n_val") or summary.get("n_val"),
                "batch_size": batch,
                "steps_per_epoch": spe,
                "epochs_cfg": epochs_cfg,
                "epochs_run": epochs_run,
                "total_steps": total,
                "global_step": summary.get("global_step"),
                "encoder_lr": _get(scfg, "optim", "encoder_lr"),
                "head_lr": _get(scfg, "optim", "head_lr"),
                "amp": _get(scfg, "train", "amp"),
                "best_epoch": summary.get("best_epoch"),
                "best_val_iou": summary.get("best_val_iou"),
                "seconds": summary.get("seconds"),
                "seconds_per_epoch": summary.get("seconds_per_epoch"),
                "peak_alloc_mib": summary.get("peak_alloc_mib"),
                "params_total": (summary.get("params") or meta.get("params") or {}).get("total"),
                "sampling_mode": meta.get("sampling", {}).get("mode"),
                "manifest_sha256": (meta.get("manifest") or {}).get("sha256", "")[:12],
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})


def write_md(rows: list[dict], path: Path) -> None:
    lines = [
        "# P0 run 预算对账",
        "",
        "自动生成，勿手改。数据源：`outputs/segformer/**/run_meta.json` + `train_summary.json`。",
        "",
        "## 1. 结论",
        "",
        "- 训练预算 = `epochs_cfg × ceil(n_train / batch_size)`，即**固定 epoch、步数随数据量增长**。",
        "- 因此 v1 的 3,000 steps 与 v3 的 3,600 steps 不是两套协议，而是同一协议下 N 从 1,277 变 1,480 的结果。",
        "- 做数据量学习曲线时，若沿用该协议，数据量与优化预算会被绑定；v4 统一改为固定 `target_steps`。",
        "",
        "## 2. 逐 run 明细",
        "",
        "| 数据版本 | 组 | seed | n_train | bs | 步/epoch | epochs(cfg/实跑) | steps(计划/实际) | best_epoch | best val IoU | 秒/epoch | 峰值 MiB | 采样 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            "| {v} | {e} | {s} | {n} | {b} | {spe} | {ec}/{er} | {ts}/{gs} | {be} | {iou} | {spe_s} | {mem} | {samp} |".format(
                v=r["data_version"],
                e=r["experiment"],
                s=r["seed"],
                n=r["n_train"],
                b=r["batch_size"],
                spe=r["steps_per_epoch"],
                ec=r["epochs_cfg"],
                er=r["epochs_run"],
                ts=r["total_steps"],
                gs=r["global_step"],
                be=r["best_epoch"],
                iou=f"{r['best_val_iou']:.4f}" if isinstance(r["best_val_iou"], (int, float)) else "-",
                spe_s=f"{r['seconds_per_epoch']:.2f}" if isinstance(r["seconds_per_epoch"], (int, float)) else "-",
                mem=f"{r['peak_alloc_mib']:.0f}" if isinstance(r["peak_alloc_mib"], (int, float)) else "-",
                samp=r["sampling_mode"] or "-",
            )
        )

    same = [r for r in rows if r["total_steps"] and r["steps_per_epoch"] and r["epochs_cfg"]]
    mismatch = [r for r in same if r["total_steps"] != r["epochs_cfg"] * r["steps_per_epoch"]]
    lines += [
        "",
        "## 3. 一致性检查",
        "",
        f"- 参与检查：{len(same)} 个 run；`total_steps == epochs × steps_per_epoch` 不符：{len(mismatch)} 个。",
    ]
    for r in mismatch:
        lines.append(
            f"  - {r['data_version']}/{r['experiment']}/{r['seed']}: "
            f"记录 {r['total_steps']} vs 推得 {r['epochs_cfg'] * r['steps_per_epoch']}"
        )
    lines += ["", "## 4. 复现", "", "```bash", "python scripts/reconcile_runs.py", "```", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out-dir", default=str(ROOT / "reports"))
    args = ap.parse_args()

    rows = collect(Path(args.root))
    if not rows:
        raise SystemExit("未找到任何 run_meta.json")
    out = Path(args.out_dir)
    write_csv(rows, out / "P0_run_reconciliation.csv")
    write_md(rows, out / "P0_run_reconciliation.md")
    print(f"[done] {len(rows)} 个 run → {out / 'P0_run_reconciliation.md'}")


if __name__ == "__main__":
    main()
