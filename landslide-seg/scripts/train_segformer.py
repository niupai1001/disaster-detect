"""训练入口：读 YAML → 跑一次完整训练（best/last + 逐 epoch 指标）。

用法::

    python scripts/check_research_protocol.py --config configs/segformer/v4r1/segformer_b0_direct10_detailres_n100_u3600_s1234.yaml
    python scripts/train_segformer.py --config configs/segformer/v4r1/segformer_b0_direct10_detailres_n100_u3600_s1234.yaml

正式新实验按项目AGENTS.md预检原固定事件留出；此入口必须显式传配置，不回落到历史a0。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:  # Windows 控制台/管道默认 cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from landslide_seg.config import load_segformer_config  # noqa: E402
from landslide_seg.training.segformer_trainer import train_one  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--device", default=None, help="cuda:0 / cuda:1 / cpu；默认自动")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    cfg = load_segformer_config(args.config, args.set)
    if args.epochs is not None:
        cfg["segformer"]["train"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["segformer"]["train"]["batch_size"] = int(args.batch_size)
    if args.max_steps is not None:
        cfg["segformer"]["train"]["max_steps"] = int(args.max_steps)

    summary = train_one(cfg, device=args.device, resume=args.resume)
    print(json.dumps({k: v for k, v in summary.items() if k != "history"},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
