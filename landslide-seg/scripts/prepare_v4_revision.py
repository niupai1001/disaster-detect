"""Build v4r1 from existing pools/caches, preserving all v4 artifacts."""
from __future__ import annotations

import copy
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import yaml
from landslide_seg.config import file_sha256
from landslide_seg.training.protocol import canonical_hash
from prepare_segformer_data import compute_stats
from make_pool_manifests import read_jsonl, write_jsonl


def main():
    out = ROOT / "manifests/v4r1"
    if (out / "protocol.json").exists():
        raise FileExistsError("v4r1 already frozen; do not overwrite")
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / "manifests/s2_10band_v4_p20.jsonl"
    rows = read_jsonl(source)
    index = {r["sample_id"]: r for r in rows}
    assert len(index) == len(rows)
    legacy = json.loads((ROOT / "manifests/split_v4.json").read_text(encoding="utf-8"))
    roles = {event: role for role, events in legacy["design"].items() for event in events}
    pools = {str(k): set() for k in (25, 50, 75, 100)}
    with (ROOT / "manifests/allowed_train_pool.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            assert roles[r["event_id"]] == "train"
            for k in pools:
                if int(r[f"in_{k}"]):
                    pools[k].add(r["sample_id"])
    assert pools["25"] < pools["50"] < pools["75"] < pools["100"]
    dev = sorted(sid for sid, r in index.items() if roles[r["event_id"]] == "dev")
    files = {}
    for k, members in pools.items():
        records = []
        for sid in sorted(members):
            rec = dict(index[sid], split="train", protocol_role="TRAIN", pool_fraction=int(k))
            records.append(rec)
        records.extend(dict(index[sid], split="val", protocol_role="DEV") for sid in dev)
        dest = out / f"pool_{k}.jsonl"
        write_jsonl(records, dest)
        files[str(dest).replace("\\", "/")] = file_sha256(dest)
        print(f"[pool] {k}: train={len(members)} dev={len(dev)}", flush=True)
    fit_ids = sorted(pools["25"])
    fit = [index[sid] for sid in fit_ids]
    stats_path = out / "stats_core25.json"
    stats = compute_stats(fit, stats_path)
    stats.update(data_version="s2_10band_v4r1_p20", fit_split="train_common_core25",
                 fit_sample_ids=fit_ids, fit_events=dict(Counter(r["event_id"] for r in fit)),
                 source_manifest_sha256=file_sha256(source), unit="DN, official harmonized product",
                 note="Fixed train-only preprocessing fitted on the common 559 labelled training tiles; DEV/FINAL/DIAG excluded.")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    files[str(stats_path).replace("\\", "/")] = file_sha256(stats_path)
    # Only TRAIN/DEV cache files are read. FINAL identities come from the old manifest.
    members = {}
    for sid, r in index.items():
        role = {"dev": "val"}.get(roles[r["event_id"]], roles[r["event_id"]])
        members[sid] = {key: r[key] for key in ("event_id", "cache", "cache_sha256", "date", "time_index")}
        members[sid]["role"] = role
        if role in ("train", "val"):
            if file_sha256(r["cache"]) != r["cache_sha256"]:
                raise ValueError(f"Cache differs from source manifest: {sid}")
    protocol = {"version": "event_ood_v4r1", "frozen": True, "design": legacy["design"],
                "source_manifest_sha256": file_sha256(source), "legacy_split_file_sha256": file_sha256(ROOT / "manifests/split_v4.json"),
                "members": members, "dev_ids": dev, "pools": {k: sorted(v) for k, v in pools.items()},
                "files": files, "selection": "equal-weight mean IoU of hiroshima and thrissur every 100 successful updates",
                "final_history": "newzealand already inspected in v4 smoke and old statistics; prospectively frozen, not historically blind",
                "final_policy": "no FINAL inference during development; no FINAL-driven research decisions",
                "negative_policy": "no verified negative-tile FPR; unannotated tiles are not negatives"}
    protocol["sha256"] = canonical_hash(protocol)
    (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=1), encoding="utf-8")
    base = yaml.safe_load((ROOT / "configs/segformer/v4/base.yaml").read_text(encoding="utf-8"))
    s = base["segformer"]
    s["data"].update(version="s2_10band_v4r1_p20", manifest="manifests/v4r1/pool_100.jsonl",
                     pool_fraction=100, stats="manifests/v4r1/stats_core25.json", protocol="manifests/v4r1/protocol.json")
    s["train"].update(budget_mode="updates", eval_every_updates=100, max_steps=3600, batch_size=64)
    s["train"].pop("target_steps", None)
    s["eval"]["selection_metric"] = "dev_event_mean_iou"
    dest = ROOT / "configs/segformer/v4r1"
    dest.mkdir(parents=True, exist_ok=True)
    registry = []
    for stem, k in ((True, 25), (True, 100), (True, 50), (True, 75), (False, 25), (False, 100)):
        cfg = copy.deepcopy(base)
        exp = f"segformer_b0_direct10_{'detailres_' if stem else ''}n{k:03d}_u3600_s1234"
        cfg["segformer"]["experiment"] = exp
        cfg["segformer"]["model"]["stem"] = stem
        cfg["segformer"]["data"].update(manifest=f"manifests/v4r1/pool_{k}.jsonl", pool_fraction=k)
        path = dest / f"{exp}.yaml"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        registry.append({"experiment": exp, "config": str(path.relative_to(ROOT)).replace("\\", "/"), "stem": stem, "pool": k})
    (dest / "experiments.json").write_text(json.dumps(registry, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[frozen] {protocol['sha256']} common_stats={len(fit_ids)} dev={len(dev)} six_configs={dest}", flush=True)


if __name__ == "__main__":
    main()
