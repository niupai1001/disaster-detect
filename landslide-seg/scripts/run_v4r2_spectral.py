"""顺序执行 S1/S3/S4 光谱批次：预检 → 2000 步训练 → best/last/step1000/1500 评测。

用法::

    python -X utf8 scripts/run_v4r2_spectral.py --device cuda:1
    python -X utf8 scripts/run_v4r2_spectral.py --device cuda:1 --only s4_learn_gate

每个 run 完成后立即写 ``queue.json``；训练完成不等于验收通过。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PY = r"C:/Users/yzh/.workbuddy/binaries/python/envs/tm/Scripts/python.exe"
CFG = ROOT / "configs" / "segformer" / "v4r2_spectral"
OUT = ROOT / "outputs" / "segformer" / "v4r2_spectral"
LOG = OUT / "_logs"
QUEUE = OUT / "queue.json"
ARMS = (
    "s1_indices", "s1_control",
    "s3_a_linear", "s3_b_idattn", "s3_c_wavelength_attn", "s3_d_local_bias", "s3_e_global_bias",
    "s4_add", "s4_fixed_gate", "s4_learn_gate", "s4_capctrl",
)
EVAL_TAGS = ("best_val", "last_val", "step1000_val", "step1500_val")


def arm_config(arm: str) -> dict:
    return yaml.safe_load((CFG / f"{arm}.yaml").read_text(encoding="utf-8-sig"))["segformer"]


def run_dir(arm: str) -> Path:
    cfg = arm_config(arm)
    return ROOT / cfg["output_dir"] / cfg["data"]["version"] / cfg["experiment"] / str(cfg["seed"])


def training_done(arm: str) -> bool:
    p = run_dir(arm) / "train_summary.json"
    if not p.is_file():
        return False
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
        return bool(s.get("complete")) and int(s.get("global_step", 0)) == 2000
    except Exception:
        return False


def missing_evals(arm: str) -> list[str]:
    return [t for t in EVAL_TAGS if not (run_dir(arm) / "eval" / f"eval_{t}.json").is_file()]


def call(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        fh.flush()
        return subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT).returncode


def load_state() -> dict:
    if QUEUE.is_file():
        return json.loads(QUEUE.read_text(encoding="utf-8"))
    return {"batch": "v4r2_spectral", "started": None, "arms": {}}


def save_state(state: dict) -> None:
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE.write_text(json.dumps(state, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    arms = [a for a in ARMS if args.only is None or a in args.only]
    state = load_state()
    state.setdefault("started", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    for arm in arms:
        cfg_path = CFG / f"{arm}.yaml"
        record = state["arms"].get(arm, {})
        record.update(config=str(cfg_path.relative_to(ROOT)).replace("\\", "/"),
                      run_dir=str(run_dir(arm).relative_to(ROOT)).replace("\\", "/"),
                      experiment=arm_config(arm)["experiment"])
        # 1) 预检（只读）
        rc = call([PY, "-X", "utf8", "scripts/check_research_protocol.py",
                   "--config", str(cfg_path.relative_to(ROOT))], LOG / f"{arm}_preflight.log")
        record["preflight"] = "PASS" if rc == 0 else f"FAIL({rc})"
        if rc != 0:
            record["status"] = "PREFLIGHT_FAILED"
            state["arms"][arm] = record
            save_state(state)
            continue
        # 2) 训练（可续跑）
        if not training_done(arm):
            cmd = [PY, "-X", "utf8", "scripts/train_segformer.py",
                   "--config", str(cfg_path.relative_to(ROOT)), "--device", args.device]
            if (run_dir(arm) / "last.pt").is_file():
                cmd.append("--resume")
            t0 = time.time()
            rc = call(cmd, LOG / f"{arm}.log")
            record["train_seconds"] = round(time.time() - t0, 1)
            record["train_exit"] = rc
            if rc != 0 or not training_done(arm):
                record["status"] = f"TRAIN_INCOMPLETE({rc})"
                state["arms"][arm] = record
                save_state(state)
                continue
        record["status"] = "TRAINED"
        # 3) 评测
        for tag, ckpt in (("best_val", "best.pt"), ("last_val", "last.pt"),
                          ("step1000_val", "step1000.pt"), ("step1500_val", "step1500.pt")):
            path = run_dir(arm) / "eval" / f"eval_{tag}.json"
            if path.is_file():
                continue
            ck = run_dir(arm) / ckpt
            if not ck.is_file():
                record.setdefault("missing_checkpoints", []).append(ckpt)
                continue
            rc = call([PY, "-X", "utf8", "scripts/evaluate_segformer.py",
                       "--checkpoint", str(ck.relative_to(ROOT)), "--split", "val",
                       "--tag", tag, "--device", args.device,
                       "--skip-efficiency"], LOG / f"{arm}_{tag}.log")
            if rc != 0:
                record.setdefault("eval_failures", []).append(tag)
        record["evals_missing"] = missing_evals(arm)
        record["status"] = "TRAINED_AWAITING_AUDIT" if not record["evals_missing"] else "TRAINED_AWAITING_EVALUATION"
        state["arms"][arm] = record
        save_state(state)
        print(f"{arm}: {record['status']}", flush=True)
    state["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save_state(state)
    print(json.dumps(state, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
