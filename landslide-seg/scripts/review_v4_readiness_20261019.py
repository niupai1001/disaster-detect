"""Read-only v4 review; writes evidence only, never trains or changes datasets.

Runs the existing train_one control flow with mocked numerical operations to
check update limits. Separately recomputes normalization from real NPZ pixels.
The mocked controller results are NOT model training results.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/segformer/review_v4_20261019"


def read_rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def controller_test(config, manifest):
    """Execute actual train_one AST; mock model, loaders, updates and saves."""
    source = ROOT / "src/landslide_seg/training/segformer_trainer.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_one")
    unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    ast.fix_missing_locations(unit)
    model = SimpleNamespace(count_params=lambda: {}, adapter_report={}, load_info={})
    model.to = lambda device: model
    optimizer = SimpleNamespace(param_groups=[{"lr": 0.0}])
    env = {
        "Path": Path, "json": json, "math": math, "time": time,
        "DEFAULT_PRETRAINED": "weights/unused",
        "set_seed": lambda *a: {}, "_device": lambda *a: SimpleNamespace(type="cpu"),
        "resolve_path": lambda p: ROOT / p, "read_manifest": read_rows,
        "load_stats": lambda p: {}, "filter_split": lambda rows, split: [r for r in rows if r["split"] == split],
        "model_type": lambda *a: "unet", "build_model": lambda *a, **kw: model,
        "torch": SimpleNamespace(__version__="MOCK", version=SimpleNamespace(cuda=None)),
        "_make_optimizer": lambda *a: optimizer, "_make_scheduler": lambda *a: None,
        "S2SegDataset": lambda rows, *a, **kw: SimpleNamespace(set_epoch=lambda *a: None),
        "manifest_identity": lambda *a: {}, "file_sha256": lambda *a: "MOCK",
        "model_identity": lambda *a: {}, "_make_loader": lambda *a: None,
        "_make_train_loader": lambda ds, rows, cfg, bs, seed, ep: math.ceil(len(rows) / bs),
        "_train_epoch": lambda model, n, *a: (0.0, 0, n, 0.0),
        "evaluate_loader": lambda *a: {"iou": 0.0}, "_save_checkpoint": lambda *a: None,
    }
    exec(compile(unit, str(source), "exec"), env)
    cfg = json.loads(json.dumps(config))
    cfg["segformer"]["data"]["manifest"] = str(manifest)
    with tempfile.TemporaryDirectory(prefix="v4_mock_controller_", dir=OUT) as tmp:
        assert Path(tmp).resolve().is_relative_to(OUT.resolve())
        env["_run_dir"] = lambda *a: Path(tmp)
        summary = env["train_one"](cfg, log=lambda *a: None)
    return {k: summary[k] for k in ("n_train", "n_val", "epochs_run", "global_step")}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/segformer/v4/base.yaml").read_text(encoding="utf-8"))
    rows = read_rows(ROOT / "manifests/s2_10band_v4_p20.jsonl")
    split = json.loads((ROOT / "manifests/split_v4.json").read_text(encoding="utf-8"))
    old_train = [r for r in rows if r["split"] == "train"]
    allowed = set(split["design"]["train"])
    stats = json.loads((ROOT / config["segformer"]["data"]["stats"]).read_text(encoding="utf-8"))
    result = {"method": "mocked actual train_one controller + real NPZ statistics; no model training",
              "legacy_stats_fit_events": dict(Counter(r["event_id"] for r in old_train)),
              "out_of_protocol_stats_events": dict(Counter(r["event_id"] for r in old_train if r["event_id"] not in allowed)),
              "pools": {}}
    previous = set()
    for k in (25, 50, 75, 100):
        path = ROOT / f"manifests/v4/pool_{k}.jsonl"
        pool = read_rows(path)
        members = {r["sample_id"] for r in pool if r["split"] == "train"}
        assert previous <= members
        previous = members
        entry = controller_test(config, path)
        entry["val_events"] = dict(Counter(r["event_id"] for r in pool if r["split"] == "val"))
        entry["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result["pools"][str(k)] = entry
    print(json.dumps({"controller": result["pools"], "stats_outside_train": result["out_of_protocol_stats_events"]}), flush=True)
    sums = {name: [0, np.zeros(10), np.zeros(10)] for name in ("legacy", "protocol_train")}
    for i, r in enumerate(old_train):
        with np.load(r["cache"]) as cached:
            pixels = cached["x10"][:, cached["valid"].astype(bool)].astype(np.float64)
        targets = ["legacy"] + (["protocol_train"] if r["event_id"] in allowed else [])
        n, s, ss = pixels.shape[1], pixels.sum(axis=1), (pixels * pixels).sum(axis=1)
        for name in targets:
            sums[name][0] += n
            sums[name][1] += s
            sums[name][2] += ss
        if (i + 1) % 700 == 0:
            print(f"real_cache_stats {i + 1}/{len(old_train)}", flush=True)
    result["recomputed_stats"] = {}
    for name, (n, s, ss) in sums.items():
        mean = s / n
        std = np.sqrt(np.maximum(ss / n - mean * mean, 0))
        result["recomputed_stats"][name] = {"pixels_per_band": n, "mean": mean.tolist(), "std": std.tolist()}
    legacy = result["recomputed_stats"]["legacy"]
    result["stats_match"] = {
        "n_samples": len(old_train) == stats["n_samples"],
        "pixel_counts": all(n == legacy["pixels_per_band"] for n in stats["n_pixels_per_band"]),
        "max_mean_error": float(np.max(np.abs(np.array(legacy["mean"]) - stats["mean"]))),
        "max_std_error": float(np.max(np.abs(np.array(legacy["std"]) - stats["std"]))),
    }
    assert result["stats_match"]["max_mean_error"] < 1e-8
    assert result["stats_match"]["max_std_error"] < 1e-8
    result["existing_smokes"] = {}
    for p in (ROOT / "outputs/segformer/s2_10band_v4_p20").glob("*/1234/train_summary.json"):
        run = json.loads(p.read_text(encoding="utf-8"))
        result["existing_smokes"][run["experiment"]] = {k: run[k] for k in ("global_step", "n_train", "n_val", "best_val_iou", "params")}
    dest = OUT / "readiness_evidence.json"
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["stats_match"]))
    print(dest)


if __name__ == "__main__":
    main()
