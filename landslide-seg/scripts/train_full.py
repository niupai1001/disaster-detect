"""可复现训练入口（YAML 驱动）。

一切来自配置文件：模型结构、损失权重、优化器、数据清单、随机种子、输出目录。
命令行只做覆盖：``--set loss.lambda_coarse=0.3``。

用法：
  python scripts/train_full.py --config configs/default.yaml
  python scripts/train_full.py --config configs/default.yaml --set train.epochs=3 --set data.batch_size=4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from landslide_seg.config import (  # noqa: E402
    file_identity,
    load_config,
    manifest_identity,
    set_seed,
)
from landslide_seg.data.certified import (  # noqa: E402
    filter_records,
    make_batches,
    read_manifest,
)
from landslide_seg.models.build import DEFAULT_WEIGHTS, build_from_config  # noqa: E402
from landslide_seg.training.losses import CombinedLoss  # noqa: E402
from landslide_seg.training.trainer import Trainer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", dest="overrides", nargs="*", action="append", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    flat = [x for group in args.overrides for x in group]
    cfg = load_config(args.config, flat)
    out_dir = Path(args.out or cfg["output_dir"])
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    seed_info = set_seed(int(cfg["seed"]), bool(cfg.get("deterministic", True)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device} seed={seed_info['seed']}")

    manifest_path = Path(cfg["data"]["manifest"])
    records = read_manifest(manifest_path)
    split_map = {
        "train": filter_records(records, split="train"),
        "val": filter_records(records, split="val"),
        "test": filter_records(records, split="test"),
    }
    print("[data]", {k: len(v) for k, v in split_map.items()})
    if not split_map["train"]:
        print("[BLOCKED] 训练集为空，先运行 scripts/make_splits.py")
        return 2

    model = build_from_config(cfg, device=device)
    loss = CombinedLoss(**{k: float(v) for k, v in cfg["loss"].items()})
    trainer = Trainer(
        model,
        loss,
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"]["weight_decay"]),
        grad_clip=float(cfg["optim"]["grad_clip"]),
        device=device,
        out_dir=out_dir,
    )

    bs = int(cfg["data"]["batch_size"])
    history = []
    for epoch in range(int(cfg["train"]["epochs"])):
        trainer.state.epoch = epoch
        batches = make_batches(
            split_map["train"], bs, shuffle=True, seed=int(cfg["seed"]) + epoch
        )
        for i, batch in enumerate(batches):
            log = trainer.train_step(batch)
            history.append(log)
            if (i + 1) % int(cfg["train"].get("log_every", 1)) == 0:
                l = log["loss"]
                print(
                    f"[ep{epoch} {i+1}/{len(batches)}] total={l['total']:.4f} "
                    f"fine={l['fine']:.4f} coarse={l['coarse']:.4f} "
                    f"presence={l['presence']:.4f} region_text={l['region_text']:.4f} "
                    f"lat={log['latency_s']:.2f}s"
                )
            spe = cfg["train"].get("steps_per_epoch")
            if spe and i + 1 >= int(spe):
                break

    metrics = {}
    for split in ("val", "test"):
        if split_map[split]:
            b = make_batches(split_map[split], bs, shuffle=False, seed=0,
                             boa_offset=float(cfg["data"].get("boa_offset_dn", 0.0)))
            metrics[split] = trainer.evaluate(
                b,
                threshold=float(cfg["eval"]["threshold"]),
                use_presence_gate=bool(cfg["eval"].get("use_presence_gate", False)),
            )
            print(f"[{split}]", json.dumps(metrics[split], ensure_ascii=False))

    meta = {
        "config": cfg,
        "seed_info": seed_info,
        "weights": {
            "terramind": file_identity(ROOT / "weights" / "TerraMind_v1_small.pt"),
            "anysat": file_identity(DEFAULT_WEIGHTS["anysat"]),
            "remoteclip": file_identity(DEFAULT_WEIGHTS["remoteclip"]),
        },
        "manifest": manifest_identity(manifest_path),
        "splits": {k: [r["sample_id"] for r in v] for k, v in split_map.items()},
        "params": model.parameter_report(),
        "device": device,
        "torch": torch.__version__,
    }
    ckpt = trainer.save(out_dir / "checkpoints" / "last.pt", extra=meta)
    (out_dir / "logs" / "train_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "logs" / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "logs" / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[ckpt] {ckpt} ({ckpt.stat().st_size / 1e6:.1f} MB)")
    print(
        f"[params] total={meta['params']['total'] / 1e6:.1f}M "
        f"trainable={meta['params']['trainable'] / 1e6:.1f}M"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
