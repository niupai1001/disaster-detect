"""四组消融的顺序执行器：A0 → A1 → B0 → B1，每组多个 seed。

每组都从**同一份官方预训练权重**重新开始（A1/B1 不加载 A0/B0 的已训练权重），
公共设置（batch、epochs、lr、增强、损失、划分）来自 ``configs/segformer/base.yaml``，
组间只切换 ``model.adapter`` 与 ``model.stem``。

用法::

    python scripts/run_segformer_ablation.py --order A0 A1 B0 B1 --seeds 1234 2026 3407
    python scripts/run_segformer_ablation.py --order A0 B0 --seeds 1234 --epochs 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:  # Windows 控制台/管道默认 cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch  # noqa: E402

from landslide_seg.config import load_segformer_config, resolve_path  # noqa: E402
from landslide_seg.training.segformer_trainer import train_one  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", nargs="+", default=["A0", "A1", "B0", "B1"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1234, 2026, 3407])
    ap.add_argument("--config-dir", default=str(ROOT / "configs" / "segformer"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag", default="", help="产物文件名后缀，避免覆盖其它数据版本的结果")
    args = ap.parse_args()

    cfg_dir = Path(args.config_dir)
    runs: list[dict] = []
    started = time.time()
    for exp in args.order:
        cfg_path = cfg_dir / f"{exp.lower()}.yaml"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"配置不存在: {cfg_path}")
        for seed in args.seeds:
            cfg = load_segformer_config(cfg_path)
            cfg["segformer"]["seed"] = int(seed)
            if args.epochs is not None:
                cfg["segformer"]["train"]["epochs"] = int(args.epochs)
            if args.batch_size is not None:
                cfg["segformer"]["train"]["batch_size"] = int(args.batch_size)
            print(f"\n===== run {exp} seed={seed} =====", flush=True)
            summary = train_one(cfg, device=args.device, resume=args.resume)
            runs.append(
                {
                    "experiment": exp,
                    "seed": int(seed),
                    "best_epoch": summary["best_epoch"],
                    "best_val_iou": summary["best_val_iou"],
                    "epochs_run": summary["epochs_run"],
                    "seconds": summary["seconds"],
                    "run_dir": summary["run_dir"],
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    out = {
        "order": args.order,
        "seeds": args.seeds,
        "total_seconds": round(time.time() - started, 1),
        "runs": runs,
    }
    out_dir = resolve_path("outputs/segformer") / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"ablation{args.tag}_summary.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print("\n===== summary =====")
    for r in runs:
        print(f"  {r['experiment']:3s} seed={r['seed']:5d} "
              f"best_epoch={r['best_epoch']:3d} val_iou={r['best_val_iou']:.4f} "
              f"({r['seconds']/60:.1f} min)")
    print(f"[saved] {out_dir / f'ablation{args.tag}_summary.json'}")


if __name__ == "__main__":
    main()
