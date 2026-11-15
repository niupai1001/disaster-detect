"""Batch 2 driver: recover the remaining Landsat windows through the existing queue.

The frozen release is read-only. Tasks, canonical input and the previous event log
are copied into `data_expansion_v2` (or a probe root) so `run_remote` resumes only
the pending part of the queue and never rewrites the frozen directory.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "outputs/observation_model/encoder_v2/data_expansion_v2"
DEFAULT_RECEIPT = ROOT / "outputs/encoder_v3/data_download_plan/batch_b_landsat_completion"
FROZEN = ROOT / "observation_model/outputs/encoder_v2/data_expansion"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--cap", type=int, default=2_000_000_000)
    parser.add_argument("--limit", type=int, default=0, help="0 runs the whole pending queue")
    args = parser.parse_args()

    root = args.root
    if args.limit:
        root = args.receipt / "probe_root"
        root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FROZEN / "canonical_input.jsonl", root / "canonical_input.jsonl")
        lines = (FROZEN / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
        pending = [line for line in lines
                   if json.loads(line)["kind"] in {"cached_landsat", "landsat_for_local_anchor"}]
        # One cached_landsat task and the rest anchors: both code paths get exercised.
        cached = [line for line in pending if json.loads(line)["kind"] == "cached_landsat"][:1]
        anchors = [line for line in pending if json.loads(line)["kind"] == "landsat_for_local_anchor"][: args.limit - len(cached)]
        (root / "tasks.jsonl").write_text("\n".join(cached + anchors) + "\n", encoding="utf-8")
        (root / "remote_events.jsonl").write_text("", encoding="utf-8")
    args.receipt.mkdir(parents=True, exist_ok=True)
    os.environ["DSH_EXPANSION_OUT"] = str(root)
    # run_remote reserves 288 MiB for the S2 calibration, so the override carries it.
    os.environ["DSH_EXPANSION_CAP"] = str(args.cap + 288 * 1024**2)
    sys.path.insert(0, str(ROOT))

    from observation_model.expand_training_data import run_remote

    run_remote()

    network = json.loads((root / "network.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in
              (root / "remote_events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = {}
    for event in events:
        counts[event.get("status")] = counts.get(event.get("status"), 0) + 1
    summary = {
        "mode": "probe" if args.limit else "complete",
        "limit": args.limit, "root": str(root), "receipt": str(args.receipt),
        "events": len(events), "status_counts": counts,
        "network": network, "network_cap": network.get("cap_bytes"),
    }
    write_json(args.receipt / ("probe_summary.json" if args.limit else "summary.json"), summary)
    write_json(args.receipt / ("probe_network.json" if args.limit else "network_batch_b.json"), network)
    print("BATCH2", json.dumps({"mode": summary["mode"], "events": len(events),
                                "status_counts": counts,
                                "received": network.get("received_bytes"),
                                "cap": network.get("cap_bytes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
