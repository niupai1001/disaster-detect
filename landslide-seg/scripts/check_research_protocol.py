"""Read-only, CPU-only preflight for the project's current event-held-out protocol.

This checks actual manifests and provenance, never just a config's display name.
No model/data cache/checkpoint is loaded or modified. Historical official runs are
intentionally outside this current-development checker.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "docs/assets/research/research_policy.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def merge(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in update.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def load_config(path: Path, overrides: list[str]) -> dict:
    # Match current trainers' single-level _base semantics; do not silently
    # infer legacy defaults or support deeper inheritance the trainer ignores.
    raw = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    base_ref = raw.pop("_base", None)
    base = {}
    if base_ref:
        base = yaml.safe_load((path.parent / base_ref).read_text(encoding="utf-8-sig")) or {}
        require("_base" not in base, "Nested _base is not supported by current trainers")
    cfg = merge(base, raw)
    for override in overrides:
        require("=" in override, "--set requires key=value")
        key, value = override.split("=", 1)
        parts, node = key.split("."), cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        node[parts[-1]] = value
    return cfg


def audit_utae_baseline(path: Path, cfg: dict, policy: dict, root: Path, policy_path: Path) -> dict:
    """U-TAE 在默认固定留出协议下的预检：成员、统计、角色与选模。"""
    require("utae_cv" in cfg and "segformer" not in cfg, "Expect a utae_cv config for the default protocol")
    s = cfg["utae_cv"]
    require(s.get("seed") == policy["seed"], "Seed differs from project policy")
    d = s["protocol"]
    directory = resolve(policy["manifest_root"], root)
    pool = resolve(d["pool"], root)
    stats = resolve(d["stats"], root)
    require(pool == directory / policy["pool_file"], "Manifest paths do not match the original held-out baseline")
    require(stats == directory / policy["stats_file"], "U-TAE baseline normalization must be stats_core25.json")
    require(d.get("roles") == {"train": policy["train_role"], "inner": policy["inner_role"]},
            "U-TAE role mapping differs from current policy")
    selection = s["eval"].get("selection")
    require(selection == policy["selection"]["utae_cv"], "Selection must use DEV event-mean IoU")
    lock = directory / policy["protocol_file"]
    protocol = read_json(lock)
    payload = {k: v for k, v in protocol.items() if k != "sha256"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                       separators=(",", ":")).encode()).hexdigest()
    require(protocol.get("frozen") and protocol.get("sha256") == digest, "Frozen protocol hash mismatch")
    require(protocol.get("version", "").split("+")[0] == policy["default_protocol"],
            "Frozen protocol version mismatch")
    registered = {resolve(k, root): v for k, v in protocol["files"].items()}
    for item in (pool, stats):
        require(registered.get(item) == sha(item), f"File unregistered or modified: {item.name}")
    rows = [json.loads(line) for line in pool.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    train = [r for r in rows if r.get("split") == "train"]
    dev = [r for r in rows if r.get("split") == "val"]
    require(len(train) + len(dev) == len(rows), "Unexpected pool role")
    require({r["sample_id"] for r in train} == set(protocol["pools"]["100"]), "TRAIN membership mismatch")
    require({r["sample_id"] for r in dev} == set(protocol["dev_ids"]), "DEV membership mismatch")
    require({r["event_id"] for r in train} == set(protocol["design"]["train"])
            and {r["event_id"] for r in dev} == set(protocol["design"]["dev"]),
            "Event membership mismatch")
    norm = read_json(stats)
    fit = set(norm.get("fit_sample_ids", []))
    require(bool(fit) and fit <= {r["sample_id"] for r in train},
            "Normalization must use TRAIN only, excluding DEV/FINAL/DIAG")
    return {"ok": True, "config": str(path), "config_sha256": sha(path),
            "effective_config_sha256": hashlib.sha256(
                json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "policy_sha256": sha(policy_path), "protocol": policy["default_protocol"], "trainer": "utae_cv",
            "seed": s["seed"], "selection": selection,
            "counts": {"train": len(train), "dev": len(dev)},
            "features": {"dem": s.get("data", {}).get("dem", False),
                         "in_channels": s.get("data", {}).get("in_channels", 10)},
            "normalization_fit_n": len(fit),
            "files": {str(p.relative_to(root)): sha(p) for p in (lock, pool, stats)},
            "warnings": ["DEV is a development set, not blind TEST; FINAL/Italy diagnostic evaluation is not performed by this checker"]}


def audit_temporal(data: dict, s: dict, rows: list[dict], root: Path) -> dict:
    """只读核对时间缓存：成员、参考帧日期、帧数、quality 侧表哈希、汇聚开关。"""
    tdir = resolve(data["temporal_root"], root)
    index_path = tdir / "index.json"
    require(index_path.is_file(), f"Temporal cache index missing: {index_path}")
    index = read_json(index_path)
    members = {r["sample_id"]: r for r in index.get("rows", [])}
    require(bool(members), "Temporal cache index has no rows")
    missing = [r["sample_id"] for r in rows if r["sample_id"] not in members]
    require(not missing, f"Temporal cache lacks {len(missing)} samples, e.g. {missing[:3]}")
    for row in rows:
        rec = members[row["sample_id"]]
        dates = rec["dates"]
        require(len(dates) >= 1, f"Empty date list: {row['sample_id']}")
        require(0 <= int(row["time_index"]) < len(dates), f"time_index out of range: {row['sample_id']}")
        require(dates[int(row["time_index"])] == row["date"],
                f"Reference frame date mismatch: {row['sample_id']}")
    mode = str(s.get("temporal", {}).get("mode", "single"))
    aggregation = str(s.get("temporal", {}).get("aggregation", "identity"))
    require(mode in ("single", "multi", "history"), f"Unknown temporal mode: {mode!r}")
    require(aggregation in ("identity", "mean", "content", "date"),
            f"Unknown temporal aggregation: {aggregation!r}")
    require((mode == "single") == (aggregation == "identity"),
            "single must use aggregation=identity; multi/history must use mean/content/date")
    count = s.get("temporal", {}).get("n_frames")
    if count is not None:
        require(type(count) is int and count > 0, "n_frames must be a positive integer")
        for row in rows:
            dates = members[row["sample_id"]]["dates"]
            available = 1 if mode == "single" else (sum(d <= row["date"] for d in dates) if mode == "history" else len(dates))
            require(available >= count, f"NOT_ELIGIBLE: {row['sample_id']} has {available} frames, needs {count}")
    qinfo = None
    quality = data.get("quality")
    if quality:
        qdir = resolve(quality, root)
        complete = qdir / "complete.json"
        if complete.is_file():
            qmeta = read_json(complete)
            require(qmeta.get("index_sha256") == sha(index_path),
                    "Quality sidecar index hash mismatch")
            qinfo = {"path": str(qdir), "enabled": True,
                     "complete_sha256": sha(complete)}
        else:
            qinfo = {"path": str(qdir), "enabled": False,
                     "note": "No complete.json; missing-value filling disabled"}
    n_frames = sorted({len(r["dates"]) for r in index["rows"]})
    return {"root": str(tdir), "index_sha256": sha(index_path),
            "n_index_rows": len(index["rows"]), "frames_per_sample": n_frames,
            "mode": mode, "aggregation": aggregation, "requested_n_frames": count, "quality": qinfo}


def audit_baseline(path: Path, cfg: dict, policy: dict, root: Path, policy_path: Path) -> dict:
    require("segformer" in cfg, "Original held-out baseline config required; adapt other trainer interfaces first")
    s = cfg["segformer"]
    require(s.get("seed") == policy["seed"], "Seed differs from project policy")
    data = s["data"]
    directory = resolve(policy["manifest_root"], root)
    pool = resolve(data["manifest"], root)
    stats = resolve(data["stats"], root)
    require(pool == directory / policy["pool_file"], "Manifest paths do not match the original held-out baseline")
    require(bool(data.get("protocol")), "Explicit frozen data.protocol is required")
    lock = resolve(data["protocol"], root)
    require(stats.parent == directory and lock.parent == directory, "Statistics/protocol must belong to original held-out baseline")
    require(int(data.get("pool_fraction", 0)) == 100, "Use original full 2223-sample TRAIN baseline")
    selection = s["eval"].get("selection_metric")
    require(selection == policy["selection"]["segformer"], "Selection must use DEV event-mean IoU")
    protocol = read_json(lock)
    payload = {k: v for k, v in protocol.items() if k != "sha256"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    require(protocol.get("frozen") and protocol.get("sha256") == digest, "Frozen protocol hash mismatch")
    require(protocol.get("version", "").split("+")[0] == policy["default_protocol"], "Frozen protocol version mismatch")
    registered = {resolve(k, root): v for k, v in protocol["files"].items()}
    for item in (pool, stats):
        require(registered.get(item) == sha(item), f"File unregistered or modified: {item.name}")
    rows = [json.loads(line) for line in pool.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    train = [r for r in rows if r.get("protocol_role") == policy["train_role"]]
    dev = [r for r in rows if r.get("protocol_role") == policy["inner_role"]]
    require(len(train) + len(dev) == len(rows), "Unexpected pool role")
    require(len({r["sample_id"] for r in rows}) == len(rows), "Duplicate/overlapping sample identities")
    train_ids = {r["sample_id"] for r in train}
    require(train_ids == set(protocol["pools"]["100"]), "TRAIN membership mismatch")
    require({r["sample_id"] for r in dev} == set(protocol["dev_ids"]), "DEV membership mismatch")
    train_events, dev_events = ({r["event_id"] for r in group} for group in (train, dev))
    require(train_events == set(protocol["design"]["train"]) and dev_events == set(protocol["design"]["dev"]), "Original event membership mismatch")
    require(not train_events & dev_events, "TRAIN/DEV events overlap")
    for row in rows:
        member = protocol["members"].get(row["sample_id"], {})
        for field in ("event_id", "cache", "cache_sha256", "date", "time_index"):
            require(row.get(field) == member.get(field), f"Frozen sample identity changed: {row['sample_id']}/{field}")
        require(row.get("split") == member.get("role"), "Sample role differs from frozen protocol")
    norm = read_json(stats)
    fit = set(norm.get("fit_sample_ids", []))
    require(bool(fit) and fit <= train_ids, "Normalization must use TRAIN only, excluding DEV/FINAL/DIAG")
    require(set(norm.get("fit_events", {})) == {protocol["members"][sid]["event_id"] for sid in fit}, "Normalization fit event mismatch")
    reference_path = directory / policy["stats_file"]
    require(registered.get(reference_path) == sha(reference_path), "Original common-core normalization hash mismatch")
    reference = read_json(reference_path)
    require(set(reference["fit_sample_ids"]) == set(protocol["pools"]["25"]), "Original common-core membership mismatch")
    require(norm["mean"][:10] == reference["mean"] and norm["std"][:10] == reference["std"], "Original S2 normalization changed; declare a separate data experiment")
    n_channels = int(s.get("model", {}).get("in_chans", 10))
    require(len(norm["mean"]) == len(norm["std"]) == n_channels, "Normalization/input channel mismatch")
    temporal = None
    if data.get("temporal_root"):
        temporal = audit_temporal(data, s, rows, root)
    return {"ok": True, "config": str(path), "config_sha256": sha(path),
            "effective_config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "policy_sha256": sha(policy_path), "protocol": policy["default_protocol"], "fold": None,
            "seed": s["seed"], "selection": selection, "counts": {"train": len(train), "dev": len(dev)},
            "events": {"train": sorted(train_events), "dev": sorted(dev_events)},
            "normalization_fit_n": len(fit),
            "files": {str(p.relative_to(root)): sha(p) for p in (lock, pool, stats)},
            "temporal": temporal,
            "warnings": ["DEV is a development set, not blind TEST; FINAL/Italy diagnostic evaluation is not performed by this checker"]}


def audit_config(config: str | Path, overrides: list[str] | None = None,
                 root: Path = ROOT, policy_path: Path | None = None,
                 protocol: str | None = None) -> dict:
    root = root.resolve()
    policy_path = policy_path or root / "docs/assets/research/research_policy.json"
    policy = read_json(policy_path)
    path = resolve(str(config), root)
    cfg = load_config(path, overrides or [])
    selected = protocol or policy["default_protocol"]
    if selected == policy["default_protocol"]:
        if "utae_cv" in cfg:
            return audit_utae_baseline(path, cfg, policy, root, policy_path)
        return audit_baseline(path, cfg, policy, root, policy_path)
    require(selected in policy.get("optional_protocols", {}), "Unknown explicit historical protocol")
    policy = merge(policy, policy["optional_protocols"][selected])
    policy["default_protocol"] = selected
    kind = "utae_cv" if "utae_cv" in cfg else "segformer"
    require(kind in cfg, "Expected explicit segformer or utae_cv config")
    settings = cfg[kind]
    require(settings.get("seed") == policy["seed"], "Seed differs from project policy")
    if kind == "utae_cv":
        data = settings["protocol"]
        pool, outer, stats = (resolve(data[k], root) for k in ("pool", "outer", "stats"))
        fold = settings.get("fold")
        require(data.get("roles") == {"train": policy["train_role"], "inner": policy["inner_role"]},
                "U-TAE role mapping differs from current policy")
        selection = settings["eval"].get("selection")
        protocol_path = pool.parent / "protocol.json"
    else:
        data = settings["data"]
        pool, stats = (resolve(data[k], root) for k in ("manifest", "stats"))
        fold = pool.parent.name
        outer = pool.parent / policy["outer_file"]
        selection = settings["eval"].get("selection_metric")
        require(bool(data.get("protocol")), "Explicit frozen data.protocol is required")
        protocol_path = resolve(data["protocol"], root)
        require(int(data.get("pool_fraction", 0)) == 100, "Current formal comparison requires full TRAIN pool")
    require(fold in policy["folds"], "Not a current event fold; official/historical split is not the default")
    fold_dir = resolve(policy["manifest_root"], root) / fold
    require(pool == fold_dir / policy["pool_file"] and outer == fold_dir / policy["outer_file"],
            "Manifest paths do not match the current event protocol")
    require(stats.parent == fold_dir and protocol_path.parent == fold_dir,
            "Statistics/protocol must belong to this same fold")
    require(selection == policy["selection"][kind], "Selection must use INNER event-mean IoU")
    protocol = read_json(protocol_path)
    payload = {k: v for k, v in protocol.items() if k != "sha256"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    require(protocol.get("frozen") and protocol.get("sha256") == digest, "Frozen protocol hash mismatch")
    require(protocol.get("version", "").split("+")[0] == f"{policy['default_protocol']}/{fold}",
            "Frozen protocol version mismatch")
    registered = {resolve(k, root): v for k, v in protocol["files"].items()}
    for item in (pool, outer):
        require(registered.get(item) == sha(item), f"Manifest is unregistered or modified: {item.name}")
    rows = lambda p: [json.loads(line) for line in p.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    pool_rows, outer_rows = rows(pool), rows(outer)
    train = [r for r in pool_rows if r.get("protocol_role") == policy["train_role"]]
    inner = [r for r in pool_rows if r.get("protocol_role") == policy["inner_role"]]
    require(len(train) + len(inner) == len(pool_rows), "Unexpected pool role")
    require(all(r.get("protocol_role") == policy["outer_role"] for r in outer_rows), "Unexpected outer role")
    all_rows = pool_rows + outer_rows
    require(len({r["sample_id"] for r in all_rows}) == len(all_rows), "Duplicate/overlapping sample identities")
    groups = (train, inner, outer_rows)
    events = [{r["event_id"] for r in group} for group in groups]
    require(all(groups) and not any(events[i] & events[j] for i, j in ((0, 1), (0, 2), (1, 2))),
            "Empty role or event overlap")
    for group, key in zip(groups, ("train_events", "inner", "outer")):
        require({r["event_id"] for r in group} == set(protocol["design"][key]), f"Event design mismatch: {key}")
    require({r["sample_id"] for r in train} == set(protocol["pools"]["100"]), "TRAIN membership mismatch")
    require({r["sample_id"] for r in inner} == set(protocol["dev_ids"]), "INNER membership mismatch")
    require({r["sample_id"] for r in outer_rows} == set(protocol["test_ids"]), "OUTER membership mismatch")
    for row in all_rows:
        member = protocol["members"].get(row["sample_id"], {})
        for field in ("event_id", "cache", "cache_sha256", "date", "time_index"):
            require(row.get(field) == member.get(field), f"Frozen sample identity changed: {row['sample_id']}/{field}")
        require(row.get("split") == member.get("role"), "Sample role differs from frozen protocol")
    norm = read_json(stats)
    fit = set(norm.get("fit_sample_ids", []))
    require(fit == {r["sample_id"] for r in train}, "Normalization provenance is not this complete TRAIN pool")
    require(norm.get("fold") == fold and set(norm.get("fit_events", {})) == events[0], "Normalization fold/events mismatch")
    n_channels = int(settings["data"].get("in_channels", 11)) if kind == "utae_cv" else int(settings.get("model", {}).get("in_chans", 10))
    require(len(norm["mean"]) == len(norm["std"]) == n_channels, "Normalization/input channel mismatch")
    # Derived U-TAE stats_dem is not registered in the base S2-only protocol.
    # Its provenance and shared ten S2 statistics can still be checked, but do
    # not claim it has a frozen protocol hash registration that does not exist.
    warnings = []
    if stats in registered:
        require(registered[stats] == sha(stats), "Registered statistics hash mismatch")
    else:
        require(kind == "utae_cv" and stats.name == "stats_dem.json", "Unregistered statistics")
        reference_path = fold_dir / "stats_train.json"
        require(registered.get(reference_path) == sha(reference_path), "Base S2 statistics hash mismatch")
        reference = read_json(reference_path)
        require(norm["mean"][:10] == reference["mean"] and norm["std"][:10] == reference["std"], "Derived stats changed the shared S2 normalization")
        warnings.append("stats_dem provenance checked; DEM values are not registered in the base protocol hash and were not refitted here")
    return {"ok": True, "config": str(path), "config_sha256": sha(path),
            "effective_config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "policy_sha256": sha(policy_path), "protocol": policy["default_protocol"], "fold": fold,
            "seed": settings["seed"], "selection": selection,
            "counts": dict(zip(("train", "inner", "outer"), map(len, groups))),
            "events": dict(zip(("train", "inner", "outer"), map(sorted, events))),
            "files": {str(p.relative_to(root)): sha(p) for p in (protocol_path, pool, outer, stats)},
            "warnings": warnings}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--protocol", default=None, help="默认原固定事件留出；EVENT_CV_V5仅用于显式历史四折复查")
    args = ap.parse_args()
    results = []
    for config in args.config:
        try:
            results.append(audit_config(config, args.set, protocol=args.protocol))
        except (ValueError, KeyError, OSError, TypeError) as exc:
            results.append({"ok": False, "config": config, "error": str(exc)})
    print(json.dumps(results, ensure_ascii=False, indent=1))
    raise SystemExit(0 if all(r["ok"] for r in results) else 1)


if __name__ == "__main__":
    main()
