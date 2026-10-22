"""重启后续跑 v6 未完成的训练与评测（串行链，按 GPU 分工）。

GPU0 链：主模型多期无光谱 → 单期光谱 → 多期光谱（每步完成即评测）。
GPU1 链：U-TAE 15 期真实日期 →（若多帧收益仍明确）截至单期日期组 → 评测。

每步都可重复执行：已完成的 train_summary 会跳过；有 last.pt 时自动 --resume。
所有命令与日志写入 ``reports/v6_execution/resume_<name>.log``。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = ROOT / "reports/v6_execution"


def run(name: str, cmd: list[str]) -> None:
    log = OUT / f"resume_{name}.log"
    print(f"[run] {name}: {' '.join(cmd)}", flush=True)
    with log.open("w", encoding="utf-8") as fh:
        p = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    print(f"[done] {name} exit={p.returncode}", flush=True)
    if p.returncode != 0:
        raise RuntimeError(f"{name} failed with {p.returncode}; see {log}")


def complete(run_dir: Path) -> bool:
    p = run_dir / "train_summary.json"
    if not p.is_file():
        return False
    try:
        return bool(json.loads(p.read_text(encoding="utf-8")).get("complete"))
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu0", default="cuda:0")
    ap.add_argument("--gpu1", default="cuda:1")
    ap.add_argument("--historical-official", action="store_true",
                    help="显式选择历史官方实验；仅在当前任务已要求该历史复现时使用")
    args = ap.parse_args()
    if not args.historical_official:
        ap.error("这是历史官方split恢复器，当前默认event_ood_v4r1固定事件留出。"
                 "本次未启动训练。阅读docs/RESEARCH_INDEX.md；"
                 "只有当前任务明确要求历史官方复现时才使用--historical-official。")
    OUT.mkdir(parents=True, exist_ok=True)

    main_root = ROOT / "outputs/segformer/v6_main"
    groups = [
        ("main_multi_baseline", "temporal_multi_baseline.yaml",
         "SegFormer-Temporal-Multi15-Baseline"),
        ("main_single_spectral", "temporal_single_spectral.yaml",
         "SegFormer-Temporal-Single-Spectral"),
        ("main_multi_spectral", "temporal_multi_spectral.yaml",
         "SegFormer-Temporal-Multi15-Spectral"),
    ]

    def gpu0_chain():
        for name, cfg, exp in groups:
            d = main_root / exp / "1234"
            if not complete(d):
                cmd = [PY, "-X", "utf8", "scripts/train_temporal.py",
                       "--config", f"configs/segformer/v6_main/{cfg}", "--device", args.gpu0]
                if (d / "last.pt").is_file():
                    cmd.append("--resume")
                run(name, cmd)
            for ck in ("best", "last"):
                target = d / "eval" / f"eval_{ck}.json"
                if (d / f"{ck}.pt").is_file() and not (target.is_file() and target.stat().st_size > 0):
                    run(f"{name}_{ck}_eval", [PY, "-X", "utf8", "scripts/evaluate_temporal.py",
                                              "--checkpoint", str(d / f"{ck}.pt"),
                                              "--splits", "val,test", "--device", args.gpu0,
                                              "--save-predictions"])

    def gpu1_chain():
        d = ROOT / "outputs/utae_v6/utae_multi_realdates/1234"
        if not complete(d):
            cmd = [PY, "-X", "utf8", "scripts/train_utae_v6.py", "--mode", "multi",
                   "--device", args.gpu1]
            if (d / "last.pt").is_file():
                cmd.append("--resume")
            run("utae_multi", cmd)
        if not (d / "eval" / "eval_summary.json").is_file():
            run("utae_multi_eval", [PY, "-X", "utf8", "scripts/evaluate_utae_v6.py", "--mode", "multi",
                                    "--checkpoints", "best,last", "--device", args.gpu1,
                                    "--save-predictions"])
        # 条件组：修正后多帧收益仍明显（val 差 > 0.02）才训练截至单期日期组
        sm = json.loads((d / "train_summary.json").read_text(encoding="utf-8"))["best_val_pooled_iou"]
        ss_path = ROOT / "outputs/utae_v6/utae_single_realdates/1234/train_summary.json"
        ss = json.loads(ss_path.read_text(encoding="utf-8"))["best_val_pooled_iou"]
        gain = float(sm) - float(ss)
        (OUT / "utae_gain_decision.json").write_text(json.dumps(
            {"single_val": ss, "multi_val": sm, "gain": gain, "run_history": gain > 0.02}, indent=1),
            encoding="utf-8")
        if gain > 0.02:
            hd = ROOT / "outputs/utae_v6/utae_history_realdates/1234"
            if not complete(hd):
                cmd = [PY, "-X", "utf8", "scripts/train_utae_v6.py", "--mode", "history",
                       "--device", args.gpu1]
                if (hd / "last.pt").is_file():
                    cmd.append("--resume")
                run("utae_history", cmd)
            run("utae_history_eval", [PY, "-X", "utf8", "scripts/evaluate_utae_v6.py",
                                      "--mode", "history", "--checkpoints", "best,last",
                                      "--device", args.gpu1, "--save-predictions"])

    errors = []

    def wrap(fn):
        try:
            fn()
        except Exception as exc:  # keep the other chain running, report at the end
            errors.append(repr(exc))
            print(f"[error] {exc}", flush=True)

    t0 = threading.Thread(target=wrap, args=(gpu0_chain,), daemon=False)
    t1 = threading.Thread(target=wrap, args=(gpu1_chain,), daemon=False)
    t0.start(); t1.start(); t0.join(); t1.join()
    run("summarize", [PY, "-X", "utf8", "scripts/summarize_v6.py"])
    if errors:
        raise SystemExit("; ".join(errors))


if __name__ == "__main__":
    main()
