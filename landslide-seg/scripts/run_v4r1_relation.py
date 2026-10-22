"""跑 event_ood_v4r1 的六组"按位置建模波段关系"实验（顺序执行，训练后立即 DEV 评测）。"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PY = r"C:/Users/yzh/.workbuddy/binaries/python/envs/tm/Scripts/python.exe"
CFG = ROOT / "configs" / "segformer" / "v4r1_spectral_relation"
ARMS = ("full", "nobias", "globalbias", "channelid", "layer1", "head1")
LOG = ROOT / "outputs" / "segformer" / "s2_10band_v4r1_spectral_relation" / "_logs"


def run_dir(arm: str) -> Path:
    cfg = yaml.safe_load((CFG / f"{arm}.yaml").read_text(encoding="utf-8"))["segformer"]
    return ROOT / cfg["output_dir"] / cfg["data"]["version"] / cfg["experiment"] / str(cfg["seed"])


def done(arm: str) -> bool:
    p = run_dir(arm) / "train_summary.json"
    if not p.is_file():
        return False
    try:
        return int(json.loads(p.read_text(encoding="utf-8")).get("global_step", 0)) >= 3600
    except Exception:
        return False


def eval_done(arm: str) -> bool:
    return (run_dir(arm) / "eval" / "eval_val.json").is_file()


def call(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        fh.flush()
        return subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT).returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    state = {}
    for arm in ARMS:
        if not done(arm):
            cmd = [PY, "-X", "utf8", "scripts/train_segformer.py",
                   "--config", f"configs/segformer/v4r1_spectral_relation/{arm}.yaml",
                   "--device", args.device]
            if (run_dir(arm) / "last.pt").is_file():
                cmd.append("--resume")
            rc = call(cmd, LOG / f"{arm}.log")
            if rc != 0:
                state[arm] = f"train_failed({rc})"
                continue
        if not eval_done(arm):
            rc = call([PY, "-X", "utf8", "scripts/evaluate_segformer.py",
                       "--checkpoint", str(run_dir(arm) / "best.pt"), "--split", "val",
                       "--tag", "val", "--device", args.device], LOG / f"{arm}_eval.log")
            state[arm] = f"done(eval={rc})"
        else:
            state[arm] = "skipped"
        (ROOT / "outputs/segformer/s2_10band_v4r1_spectral_relation/_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
