"""event_ood_v4r1 地形对照的依赖感知调度（两卡并行）。

任务图：
- ``pretrain_ssl``  (无依赖)              -> weights/terrain/v4r1_ssl.pt
- ``pretrain_scoring`` / ``pretrain_extra`` (依赖 pretrain_ssl)
- 训练臂：demraw / capcontrol 无依赖；ssl、frozenalpha 依赖 pretrain_ssl；
  sslsup 依赖 pretrain_ssl + pretrain_scoring；sslbudget 依赖 pretrain_ssl + pretrain_extra。

每个训练任务结束后自动跑 DEV 评测（``--split val``），产物落在各自 run 目录。

用法::

    python scripts/run_v4r1_terrain.py --dry-run
    python scripts/run_v4r1_terrain.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = r"C:/Users/yzh/.workbuddy/binaries/python/envs/tm/Scripts/python.exe"
CFG = ROOT / "configs" / "segformer" / "v4r1_terrain_raw"
LOGS = ROOT / "outputs" / "segformer" / "s2_10band_v4r1_p20_geo" / "_logs"
VERSION = "s2_10band_v4r1_geo_raw"

# name -> (kind, deps)
JOBS = {
    "pretrain_ssl": ("pretrain_ssl", []),
    "demraw": ("train", []),
    "capcontrol": ("train", []),
    "pretrain_scoring": ("pretrain_scoring", ["pretrain_ssl"]),
    "pretrain_extra": ("pretrain_extra", ["pretrain_ssl"]),
    "ssl": ("train", ["pretrain_ssl"]),
    "frozenalpha": ("train", ["pretrain_ssl"]),
    "sslsup": ("train", ["pretrain_ssl", "pretrain_scoring"]),
    "sslbudget": ("train", ["pretrain_ssl", "pretrain_extra"]),
}

PRETRAIN_CMDS = {
    "pretrain_ssl": ["--task", "ssl"],
    "pretrain_scoring": ["--task", "scoring"],
    "pretrain_extra": ["--task", "ssl_extra"],
}


def run_dir(name: str) -> Path:
    import yaml

    cfg = yaml.safe_load((CFG / f"{name}.yaml").read_text(encoding="utf-8"))["segformer"]
    return ROOT / cfg["output_dir"] / cfg["data"]["version"] / cfg["experiment"] / str(cfg["seed"])


def train_done(name: str) -> bool:
    p = run_dir(name) / "train_summary.json"
    if not p.is_file():
        return False
    try:
        return int(json.loads(p.read_text(encoding="utf-8")).get("global_step", 0)) >= 3600
    except Exception:
        return False


def eval_done(name: str) -> bool:
    return (run_dir(name) / "eval" / "eval_best_val.json").is_file()


def done(name: str) -> bool:
    kind = JOBS[name][0]
    if kind == "train":
        return train_done(name) and eval_done(name)
    task = {"pretrain_ssl": "ssl", "pretrain_scoring": "scoring", "pretrain_extra": "ssl_extra"}[name]
    return (ROOT / "weights" / "terrain" / f"v4r1raw_{task}.pt").is_file()


def call(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        fh.flush()
        return subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT).returncode


def execute(name: str, device: str) -> str:
    kind = JOBS[name][0]
    if kind.startswith("pretrain"):
        rc = call([PY, "-X", "utf8", "scripts/pretrain_terrain.py", *PRETRAIN_CMDS[name],
                   "--device", device], LOGS / f"{name}.log")
        return f"done(rc={rc})"
    if not train_done(name):
        cmd = [PY, "-X", "utf8", "scripts/train_segformer.py",
               "--config", f"configs/segformer/v4r1_terrain_raw/{name}.yaml", "--device", device]
        if (run_dir(name) / "last.pt").is_file():
            cmd.append("--resume")
        rc = call(cmd, LOGS / f"{name}.log")
        if rc != 0:
            return f"train_failed(rc={rc})"
    if not eval_done(name):
        rc = call([PY, "-X", "utf8", "scripts/evaluate_segformer.py",
                   "--checkpoint", str(run_dir(name) / "best.pt"), "--split", "val",
                   "--tag", "val", "--device", device], LOGS / f"{name}_eval.log")
        return f"done(eval_rc={rc})"
    return "skipped"


def worker(device: str, state: dict, lock: threading.Lock, failed: set) -> None:
    while True:
        with lock:
            ready = [n for n, (_, deps) in JOBS.items()
                     if n not in state and not (set(deps) & failed) and all(d in state for d in deps)]
        if not ready:
            with lock:
                pending = [n for n in JOBS if n not in state]
            if not pending or all(set(JOBS[n][1]) & failed for n in pending):
                return
            time.sleep(20)
            continue
        name = ready[0]
        with lock:
            state[name] = "running"
        result = execute(name, device)
        with lock:
            state[name] = result
            if "failed" in result:
                failed.add(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--state", default="outputs/segformer/s2_10band_v4r1_geo_raw/_state_terrain.json")
    args = ap.parse_args()
    if args.dry_run:
        for n, (kind, deps) in JOBS.items():
            print(f"{n:18s} {kind:16s} deps={deps} done={done(n)}")
        return
    state: dict = {}
    lock = threading.Lock()
    failed: set = set()
    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(dev, state, lock, failed), daemon=True)
               for dev in ("cuda:0", "cuda:1")]
    for t in threads:
        t.start()
    path = ROOT / args.state
    while any(t.is_alive() for t in threads):
        time.sleep(30)
        path.write_text(json.dumps({"elapsed_s": round(time.time() - t0), "jobs": state},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    path.write_text(json.dumps({"elapsed_s": round(time.time() - t0), "jobs": state},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
