"""历史四折 DEM 按"臂"批量训练 + 评测（非当前原固定留出队列）。

每个臂 = 四折 × (训练 3600 更新 + best/last 评测)。臂名对应配置文件名
``configs/segformer/event_cv_v5_dem/<折>_<臂>.yaml``，评测用同折的 ``test.jsonl``。

用法::

    python scripts/run_dem_arm.py --arm demraw --historical-event-cv --dry-run

仅在当前任务明确要求历史四折复查时使用；默认原2223/669配置尚需单独适配。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

from check_research_protocol import audit_config

ROOT = Path(__file__).resolve().parents[1]
PY = r"C:/Users/yzh/.workbuddy/binaries/python/envs/tm/Scripts/python.exe"
FOLDS = ("F1", "F2", "F3", "F4")
CHAINS = {"cuda:0": ("F1", "F2"), "cuda:1": ("F3", "F4")}
CFG_DIR = ROOT / "configs" / "segformer" / "event_cv_v5_dem"
LOG_DIR = ROOT / "outputs" / "segformer" / "event_cv_v5_dem" / "_logs"


def discovery(arm: str, fold: str) -> Path:
    import yaml

    cfg = yaml.safe_load((CFG_DIR / f"{fold}_{arm}.yaml").read_text(encoding="utf-8"))["segformer"]
    return ROOT / cfg["output_dir"] / cfg["data"]["version"] / cfg["experiment"] / str(cfg["seed"])


def train_done(arm: str, fold: str) -> bool:
    p = discovery(arm, fold) / "train_summary.json"
    if not p.is_file():
        return False
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    return int(s.get("global_step", 0)) >= 3600


def eval_done(arm: str, fold: str) -> bool:
    return (discovery(arm, fold) / "eval" / "eval_best_test.json").is_file()


def call(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        fh.flush()
        return subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT).returncode


def chain(arm: str, device: str, folds: tuple[str, ...], state: dict, lock: threading.Lock) -> None:
    for fold in folds:
        key = f"{fold}"
        if not train_done(arm, fold):
            cmd = [PY, "-X", "utf8", "scripts/train_segformer.py",
                   "--config", f"configs/segformer/event_cv_v5_dem/{fold}_{arm}.yaml",
                   "--device", device]
            if (discovery(arm, fold) / "last.pt").is_file():
                cmd.append("--resume")
            rc = call(cmd, LOG_DIR / f"{arm}_{fold}.log")
            if rc != 0:
                with lock:
                    state[key] = f"train_failed(rc={rc})"
                continue
        if not eval_done(arm, fold):
            out = discovery(arm, fold)
            rc = call([PY, "-X", "utf8", "scripts/evaluate_segformer.py",
                       "--checkpoint", str(out / "best.pt"), "--split", "test",
                       "--manifest", f"manifests/event_cv_v5/{fold}/test.jsonl",
                       "--tag", "test", "--save-predictions", "--device", device],
                      LOG_DIR / f"{arm}_{fold}_eval.log")
            with lock:
                state[key] = f"done(rc_eval={rc})"
        else:
            with lock:
                state[key] = "skipped"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--historical-event-cv", action="store_true", help="显式复查历史四折；不是当前默认开发队列")
    ap.add_argument("--state", default=None)
    args = ap.parse_args()
    if not args.historical_event_cv:
        ap.error("当前默认原固定事件留出基线；本脚本为历史四折队列，需显式--historical-event-cv。")
    # Fail before starting either GPU chain if any fold uses a wrong protocol.
    for fold in FOLDS:
        check = audit_config(CFG_DIR / f"{fold}_{args.arm}.yaml", protocol="EVENT_CV_V5")
        print(f"[protocol] {check['protocol']}/{fold} seed={check['seed']}")
    if args.dry_run:
        for dev, folds in CHAINS.items():
            for f in folds:
                print(f"{dev} {f:3s} train_done={train_done(args.arm, f)} eval_done={eval_done(args.arm, f)} "
                      f"dir={discovery(args.arm, f)}")
        return
    state: dict = {}
    lock = threading.Lock()
    t0 = time.time()
    threads = [threading.Thread(target=chain, args=(args.arm, dev, tuple(folds), state, lock), daemon=True)
               for dev, folds in CHAINS.items()]
    for t in threads:
        t.start()
    state_path = ROOT / (args.state or f"outputs/segformer/event_cv_v5_dem/_state_{args.arm}.json")
    while any(t.is_alive() for t in threads):
        time.sleep(30)
        state_path.write_text(json.dumps({"arm": args.arm, "elapsed_s": round(time.time() - t0),
                                          "folds": state}, ensure_ascii=False, indent=1), encoding="utf-8")
    state_path.write_text(json.dumps({"arm": args.arm, "elapsed_s": round(time.time() - t0),
                                      "folds": state}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
