"""Live status: one JSON file and one PNG, refreshed on a timer.

Run it in the background and open outputs/status.json / outputs/status_curves.png
whenever you want the current state — no need to ask for a report.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TMP = ROOT / ".tmp"


def read_jsonl(path):
    rows = []
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    except Exception:
        pass
    return rows


def runs():
    out = []
    for path in sorted(OUT.rglob("*.jsonl")):
        rows = read_jsonl(path)
        if not rows or "loss" not in rows[-1]:
            continue
        losses = [r["loss"] for r in rows]
        out.append({
            "name": path.stem,
            "group": path.parent.name,
            "file": str(path.relative_to(ROOT)),
            "points": len(rows),
            "step": rows[-1].get("step"),
            "last_loss": losses[-1],
            "min_loss": min(losses),
            "max_loss": max(losses),
            "first_step_loss": losses[0],
            "updated": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            "stale_s": round(time.time() - path.stat().st_mtime),
        })
    return out


def download():
    log = TMP / "companion.log"
    info = {"log": str(log.relative_to(ROOT)), "groups_done": None, "groups_total": None,
            "landSat_files": 0, "hls_files": 0, "gb": 0.0, "updated": None}
    if log.exists():
        tail = [l for l in log.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
        for line in reversed(tail):
            if line.startswith("["):
                try:
                    frac = line.split("]")[0].strip("[")
                    done, total = frac.split("/")
                    info["groups_done"] = int(done)
                    info["groups_total"] = int(total)
                    info["elapsed_s"] = int(line.rsplit("}", 1)[1].strip().rstrip("s"))
                    break
                except Exception:
                    continue
        info["updated"] = datetime.fromtimestamp(log.stat().st_mtime).isoformat(timespec="seconds")
    for key, sub in (("landSat_files", "sen12_landsat"), ("hls_files", "sen12_hls")):
        d = ROOT / "data" / sub
        if d.exists():
            files = list(d.glob("*.nc"))
            info[key] = len(files)
            info["gb"] = round(info["gb"] + sum(f.stat().st_size for f in files) / 1e9, 2)
    return info


def plot(all_runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keep = {"e1_official", "e1_gpu", "dino_official"}
    shown = [r for r in all_runs if r["group"] in keep]
    groups = {}
    for r in shown:
        groups.setdefault(r["group"], []).append(r)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    for r in shown:
        rows = read_jsonl(ROOT / r["file"])
        xs = [x.get("step") for x in rows]
        ys = [x["loss"] for x in rows]
        # raw 5-step points are noisy; a rolling median keeps the trend readable
        smooth = []
        for i in range(len(ys)):
            window = ys[max(0, i - 6):i + 1]
            smooth.append(sorted(window)[len(window) // 2])
        label = r["name"] if r["group"] != "e1_official" else f"E1 {r['name']}"
        # a finished run also stops updating; only flag runs that are short *and*
        # silent, which is what an abandoned or crashed run looks like
        group_max = max(x["step"] or 0 for x in all_runs if x["group"] == r["group"])
        abandoned = r["stale_s"] > 600 and (r["step"] or 0) < 0.9 * group_max
        ax.plot(xs, smooth, lw=1.4,
                label=label + (" (abandoned)" if abandoned else ""),
                ls="--" if abandoned else "-", alpha=0.5 if abandoned else 1.0)
    ax.set_ylim(0, 7)
    ax.set_xlabel("step")
    ax.set_ylabel("DINO loss")
    ax.set_title("DINO loss — current experiments only")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax2 = axes[1]
    d = download()
    labels = ["Landsat", "HLS"]
    values = [d["landSat_files"], d["hls_files"]]
    ax2.bar(labels, values, color=["#4C72B0", "#55A868"])
    ax2.set_title(f"companion patches downloaded ({d['gb']} GB)")
    for i, v in enumerate(values):
        ax2.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def snapshot():
    all_runs = runs()
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "runs": all_runs,
        "download": download(),
        "how_to_read": {
            "status.json": "this file; 'runs' holds every training log found under outputs/",
            "status_curves.png": "left: loss curves; right: companion download counts",
            "stale_s": "seconds since that log was last written; growing means the run stopped",
        },
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "status.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    try:
        plot(all_runs, OUT / "status_curves.png")
    except Exception as exc:
        payload["plot_error"] = f"{type(exc).__name__}: {exc}"
        (OUT / "status.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args(argv)

    while True:
        payload = snapshot()
        print(f"[{payload['generated']}] runs={len(payload['runs'])} "
              f"download={payload['download']['landSat_files']}+{payload['download']['hls_files']}",
              flush=True)
        if not args.watch:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())

