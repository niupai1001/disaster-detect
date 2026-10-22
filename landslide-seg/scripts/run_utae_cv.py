"""历史四折事件协议 · U-TAE 12 次训练的顺序调度（非默认开发队列）。

每张卡一条链，链内按 single → history → multi 顺序执行（短的先完成，便于尽早
看到方向）；已完成的 run 跳过。训练结束后自动调用 ``evaluate_utae_cv.py``。

用法::

    python scripts/run_utae_cv.py --historical-event-cv --dry-run

仅在当前任务明确要求历史四折复查时使用；后续默认原2223/669固定事件留出。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from check_research_protocol import audit_config

ROOT = Path(__file__).resolve().parents[1]
PY = r"C:/Users/yzh/.workbuddy/binaries/python/envs/tm/Scripts/python.exe"
MODES = ("single", "history", "multi")
CHAINS = {"cuda:0": ("F1", "F2"), "cuda:1": ("F3", "F4")}
LOG_DIR = ROOT / "outputs" / "utae_cv" / "_logs"


def run_dir(fold: str, mode: str) -> Path:
    return ROOT / "outputs" / "utae_cv" / fold / mode / "1234"


def done(fold: str, mode: str) -> bool:
    p = run_dir(fold, mode) / "train_summary.json"
    if not p.is_file():
        return False
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(s.get("complete")) and int(s.get("epochs", 0)) == 75


def call(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        fh.flush()
        p = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
    return p.returncode


def chain(device: str, folds: tuple[str, ...], state: dict, lock: threading.Lock) -> None:
    for fold in folds:
        for mode in MODES:
            key = f"{fold}/{mode}"
            if done(fold, mode):
                with lock:
                    state[key] = "skipped(complete)"
                continue
            t0 = time.time()
            rc = call([PY, "-X", "utf8", "scripts/train_utae_cv.py",
                       "--config", f"configs/utae/event_cv/{fold}.yaml",
                       "--mode", mode, "--device", device, "--resume"],
                      LOG_DIR / f"{fold}_{mode}.log")
            if rc != 0:
                with lock:
                    state[key] = f"train_failed(rc={rc})"
                continue
            rc = call([PY, "-X", "utf8", "scripts/evaluate_utae_cv.py",
                       "--config", f"configs/utae/event_cv/{fold}.yaml",
                       "--mode", mode, "--device", device, "--save-predictions"],
                      LOG_DIR / f"{fold}_{mode}_eval.log")
            with lock:
                state[key] = f"done(rc_eval={rc},{round(time.time() - t0)}s)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--historical-event-cv", action="store_true", help="显式复查历史四折；不是当前默认开发队列")
    ap.add_argument("--state", default="outputs/utae_cv/run_state.json")
    args = ap.parse_args()
    if not args.historical_event_cv:
        ap.error("当前默认原固定事件留出基线，不再自动安排四折。历史复查才使用--historical-event-cv。")
    # Validate all folds before any worker can launch a subprocess.
    for fold in ("F1", "F2", "F3", "F4"):
        check = audit_config(f"configs/utae/event_cv/{fold}.yaml", protocol="EVENT_CV_V5")
        print(f"[protocol] {check['protocol']}/{fold} seed={check['seed']}")
    plan = {dev: list(folds) for dev, folds in CHAINS.items()}
    if args.dry_run:
        for dev, folds in plan.items():
            for f in folds:
                for m in MODES:
                    print(f"{dev}  {f}/{m:8s} run_dir={run_dir(f, m)}  done={done(f, m)}")
        return
    state: dict = {}
    lock = threading.Lock()
    threads = [threading.Thread(target=chain, args=(dev, tuple(folds), state, lock), daemon=True)
               for dev, folds in plan.items()]
    t0 = time.time()
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(30)
        (ROOT / args.state).write_text(
            json.dumps({"elapsed_s": round(time.time() - t0), "jobs": state}, ensure_ascii=False, indent=1),
            encoding="utf-8")
    (ROOT / args.state).write_text(
        json.dumps({"elapsed_s": round(time.time() - t0), "jobs": state}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
