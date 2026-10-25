"""USE interface diagnostics on a fixed 128-tile TRAIN subset.

Scope (diagnostic only; NOT a formal experiment):
  * select 128 TRAIN tiles stratified by event and connected-object size;
  * inspect direct-10 vs USE inputs, USE latent, 32->10 bridge, MiT stage 1/2,
    and the full-grid detail stem: activations, gradients, spatial locality;
  * test joint band+metadata permutation, missing/masked bands, valid-region
    handling;
  * run two independent 300-update fitting probes (direct / USE) on the same
    fixed TRAIN split, in isolated output directories.

The probes are fitting diagnostics, not generalisation estimates.  No 2000-step
formal run is started and no formal configuration/checkpoint is modified.

Commands (run from the project root, with the project Python):
  python -X utf8 scripts/diagnose_use_interface.py select
  python -X utf8 scripts/diagnose_use_interface.py represent
  python -X utf8 scripts/diagnose_use_interface.py probe --arm both
  python -X utf8 scripts/diagnose_use_interface.py figures
  python -X utf8 scripts/diagnose_use_interface.py report
  python -X utf8 scripts/diagnose_use_interface.py all
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from landslide_seg.config import (
    PROJECT_ROOT,
    file_sha256,
    load_segformer_config,
    resolve_path,
    set_seed,
)
from landslide_seg.data.optical_observation import OpticalBatch, SUBSETS, collate_optical
from landslide_seg.data.segformer_dataset import (
    S2_10_BANDS,
    S2SegDataset,
    collate,
    filter_split,
    load_stats,
    read_manifest,
)
from landslide_seg.data.use_dataset import USESegDataset
from landslide_seg.models.factory import build_model, model_identity
from landslide_seg.training.losses import bce_loss, dice_loss
from landslide_seg.training.metrics import confusion, summarize
from landslide_seg.training.segformer_trainer import (
    _make_optimizer,
    _make_scheduler,
    load_checkpoint,
)
from landslide_seg.training.segformer_trainer import _seg_loss

ROOT = PROJECT_ROOT
DIAG_ROOT = ROOT / "outputs/use_interface_diagnostics"
DOC_ROOT = ROOT / "docs/assets/research/use_interface_diagnostics"

DIRECT_CONFIG = ROOT / "configs/segformer/use/baseline.yaml"
USE_CONFIG = ROOT / "configs/segformer/use/physical.yaml"
DIRECT_CKPT = ROOT / "outputs/use/s2_10band_v4r1_p20/use_baseline/1234/best.pt"
USE_CKPT = ROOT / "outputs/use/s2_10band_v4r1_p20/use_physical/1234/best.pt"
MANIFEST = ROOT / "manifests/v4r1/pool_100.jsonl"
STATS = ROOT / "manifests/v4r1/stats_core25.json"
QUALITY_INDEX = ROOT / "outputs/use/quality/index.json"

SEED = 1234
SELECT_SEED = 20261024
N_TILES = 128
N_FIT = 96
PROBE_UPDATES = 300
PROBE_BATCH = 8
PROBE_EVAL_EVERY = 50

#: Connected-object size classes.  The evaluation report uses 1-4 / 5-25 / 26+;
#: the diagnostic splits 26+ once more so that large-scar tiles do not dominate.
SIZE_BINS = ((1, 4, "1_4"), (5, 25, "5_25"), (26, 100, "26_100"), (101, 10**9, "101_plus"))
SIZE_LABELS = {"1_4": "1-4 px", "5_25": "5-25 px", "26_100": "26-100 px", "101_plus": "101+ px"}

COLORS = {"direct": "#2f5d8a", "use": "#b5653a"}


# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------
def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    h = hashlib.sha256()
    h.update(str(tuple(tensor.shape)).encode())
    h.update(str(tensor.dtype).encode())
    h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        h.update(key.encode("utf-8"))
        h.update(str(tuple(value.shape)).encode())
        h.update(str(value.dtype).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def hash_file(path: Path) -> str:
    return file_sha256(path)


def _device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def size_class(max_object_px: int) -> str:
    # TRAIN tiles are all label-positive; a degenerate empty mask is kept in the
    # smallest stratum only so selection never drops a tile silently.
    if max_object_px <= 0:
        return "1_4"
    for low, high, name in SIZE_BINS:
        if low <= max_object_px <= high:
            return name
    return "101_plus"


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------
def mask_object_stats(cache: str | Path) -> dict:
    """Connected-object statistics of the binary label mask (8-connectivity)."""
    from scipy import ndimage

    with np.load(cache) as data:
        mask = np.asarray(data["mask"], dtype=bool)
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.int8))
    if count == 0:
        return {"n_objects": 0, "max_object_px": 0, "median_object_px": 0.0, "mask_px": 0}
    sizes = np.bincount(labels.ravel())[1:]
    return {
        "n_objects": int(count),
        "max_object_px": int(sizes.max()),
        "median_object_px": float(np.median(sizes)),
        "mask_px": int(sizes.sum()),
    }


def load_object_stats(records: list[dict], out_dir: Path) -> dict[str, dict]:
    path = out_dir / "selection/train_object_stats.jsonl"
    cached: dict[tuple[str, str], dict] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cached[(row["sample_id"], row["cache_sha256"])] = row
    rows, computed = [], 0
    for rec in records:
        key = (rec["sample_id"], rec["cache_sha256"])
        row = cached.get(key)
        if row is None:
            row = {"sample_id": rec["sample_id"], "cache_sha256": rec["cache_sha256"], **mask_object_stats(rec["cache"])}
            computed += 1
        rows.append(row)
    if computed or not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return {row["sample_id"]: row for row in rows}


def stratified_allocation(counts: dict[Any, int], total: int) -> dict[Any, int]:
    """Proportional allocation with at least one tile per non-empty stratum."""
    keys = sorted(k for k, v in counts.items() if v > 0)
    available = sum(counts[k] for k in keys)
    if total > available:
        raise ValueError(f"Requested {total} tiles but only {available} available")
    if total < len(keys):
        raise ValueError(f"Requested {total} tiles but {len(keys)} non-empty strata")
    base = {k: 1 for k in keys}
    exact = {k: total * counts[k] / available for k in keys}
    remaining = total - len(keys)
    while remaining > 0:
        candidates = [k for k in keys if base[k] < counts[k]]
        if not candidates:
            break
        chosen = max(candidates, key=lambda k: (exact[k] - base[k], counts[k], str(k)))
        base[chosen] += 1
        remaining -= 1
    return base


def allocate_checks(strata_sizes: dict[Any, int], total_check: int) -> dict[Any, int]:
    """Allocate the 32 probe-check tiles proportionally, leaving >=1 fit per stratum."""
    keys = sorted(strata_sizes)
    total = sum(strata_sizes.values())
    exact = {k: total_check * strata_sizes[k] / total for k in keys}
    base = {k: min(int(exact[k]), max(0, strata_sizes[k] - 1)) for k in keys}
    diff = total_check - sum(base.values())
    while diff > 0:
        candidates = [k for k in keys if base[k] < max(0, strata_sizes[k] - 1)]
        if not candidates:
            break
        chosen = max(candidates, key=lambda k: (exact[k] - base[k], strata_sizes[k], str(k)))
        base[chosen] += 1
        diff -= 1
    while diff < 0:
        candidates = [k for k in keys if base[k] > 0]
        if not candidates:
            break
        chosen = min(candidates, key=lambda k: (exact[k] - base[k], str(k)))
        base[chosen] -= 1
        diff += 1
    return base


def selection_key(sample_id: str) -> str:
    return hashlib.sha256(f"{SELECT_SEED}:{sample_id}".encode()).hexdigest()


def build_selection(out_dir: Path = DIAG_ROOT, n_tiles: int = N_TILES) -> dict:
    records = filter_split(read_manifest(MANIFEST), "train")
    stats = load_object_stats(records, out_dir)
    enriched = []
    for rec in records:
        row = dict(rec)
        info = stats[rec["sample_id"]]
        row["object"] = info
        row["size_class"] = size_class(int(info["max_object_px"]))
        enriched.append(row)
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in enriched:
        strata[(rec["event_id"], rec["size_class"])].append(rec)
    allocation = stratified_allocation({k: len(v) for k, v in strata.items()}, n_tiles)
    selected: list[dict] = []
    for key, count in allocation.items():
        members = sorted(strata[key], key=lambda r: selection_key(r["sample_id"]))
        selected.extend(members[:count])
    if len(selected) != n_tiles or len({r["sample_id"] for r in selected}) != n_tiles:
        raise RuntimeError("Selection count/duplicate check failed")
    checks = allocate_checks({key: count for key, count in allocation.items()}, n_tiles - N_FIT)
    roles: dict[str, str] = {}
    for key, count in allocation.items():
        members = sorted(strata[key], key=lambda r: selection_key(r["sample_id"]))
        n_check = checks.get(key, 0)
        for i, rec in enumerate(members[:count]):
            roles[rec["sample_id"]] = "check" if i < n_check else "fit"
    if sum(v == "fit" for v in roles.values()) != N_FIT:
        raise RuntimeError("Probe role allocation does not match 96/32")
    selected.sort(key=lambda r: (r["event_id"], r["size_class"], r["sample_id"]))
    tiles = []
    for rec in selected:
        info = rec["object"]
        tiles.append(
            {
                "sample_id": rec["sample_id"],
                "event_id": rec["event_id"],
                "date": rec["date"],
                "time_index": rec["time_index"],
                "split": "TRAIN",
                "cache": rec["cache"],
                "cache_sha256": rec["cache_sha256"],
                "mask_px": int(info["mask_px"]),
                "n_objects": int(info["n_objects"]),
                "max_object_px": int(info["max_object_px"]),
                "median_object_px": float(info["median_object_px"]),
                "size_class": rec["size_class"],
                "stratum": f"{rec['event_id']}::{rec['size_class']}",
                "probe_role": roles[rec["sample_id"]],
                "selection_key": selection_key(rec["sample_id"]),
            }
        )
    payload = {
        "schema_version": 1,
        "purpose": "fixed TRAIN diagnostic subset for USE interface diagnostics; not generalisation evidence",
        "protocol": "event_ood_v4r1",
        "split": "TRAIN",
        "seed": SELECT_SEED,
        "n_tiles": n_tiles,
        "n_fit": N_FIT,
        "n_check": n_tiles - N_FIT,
        "selection_rule": (
            "stratum = TRAIN event x connected-object size class of the largest object "
            "(1-4/5-25/26-100/101+ px); proportional allocation with >=1 per non-empty stratum; "
            "within-stratum order by SHA256(seed:sample_id)"
        ),
        "probe_role_rule": "per-stratum proportional 25% check tiles (32 total), >=1 fit per stratum",
        "source_manifest": str(MANIFEST).replace("\\", "/"),
        "source_manifest_sha256": hash_file(MANIFEST),
        "object_stats_rule": "scipy.ndimage.label, 8-connectivity, on the frozen npz mask",
        "tiles": tiles,
    }
    payload["content_sha256"] = content_sha256(canonical_json([(t["sample_id"], t["probe_role"]) for t in tiles]))
    path = out_dir / "selection/diagnostic_tiles.json"
    write_json(path, payload)
    digest = hash_file(path)
    (out_dir / "selection/diagnostic_tiles.sha256").write_text(digest + "\n", encoding="utf-8")
    summary = {
        "n_tiles": n_tiles,
        "n_fit": N_FIT,
        "n_check": n_tiles - N_FIT,
        "by_event": dict(sorted(Counter(t["event_id"] for t in tiles).items())),
        "by_size_class": dict(sorted(Counter(t["size_class"] for t in tiles).items())),
        "by_stratum": {f"{k[0]}::{k[1]}": v for k, v in sorted(Counter((t["event_id"], t["size_class"]) for t in tiles).items())},
        "file_sha256": digest,
        "content_sha256": payload["content_sha256"],
        "max_object_px_quantiles": np.percentile([t["max_object_px"] for t in tiles], [0, 10, 25, 50, 75, 90, 100]).round(2).tolist(),
    }
    write_json(out_dir / "selection/sampling_summary.json", summary)
    return payload


def load_tiles(out_dir: Path = DIAG_ROOT) -> dict:
    return read_json(out_dir / "selection/diagnostic_tiles.json")


def select_diagnostic_tiles(records: list[dict], count: int) -> list[dict]:
    """Deterministic, event/size-mixed subset for gradient or validity passes."""
    ordered = sorted(records, key=lambda r: selection_key(r["sample_id"]))
    buckets: dict[str, list[dict]] = defaultdict(list)
    for rec in ordered:
        buckets[rec["event_id"]].append(rec)
    picked: list[dict] = []
    events = sorted(buckets)
    while len(picked) < min(count, len(records)):
        progressed = False
        for event in events:
            if buckets[event] and len(picked) < count:
                picked.append(buckets[event].pop(0))
                progressed = True
        if not progressed:
            break
    return picked


# ---------------------------------------------------------------------------
# feature accumulators
# ---------------------------------------------------------------------------
class FeatureAccumulator:
    """Per-layer activation statistics accumulated over batches (channels first)."""

    def __init__(self) -> None:
        self.count = 0
        self.sum: np.ndarray | None = None
        self.sumsq: np.ndarray | None = None
        self.cov: np.ndarray | None = None
        self.diff_sum: np.ndarray | None = None
        self.diff_count = 0
        self.h = {"cross": None, "left": None, "right": None, "left_sq": None, "right_sq": None, "count": 0}
        self.v = {"cross": None, "left": None, "right": None, "left_sq": None, "right_sq": None, "count": 0}
        self.shape: tuple[int, int, int] | None = None

    def _init(self, channels: int, height: int, width: int) -> None:
        self.sum = np.zeros(channels, dtype=np.float64)
        self.sumsq = np.zeros(channels, dtype=np.float64)
        self.cov = np.zeros((channels, channels), dtype=np.float64)
        self.diff_sum = np.zeros(channels, dtype=np.float64)
        self.shape = (channels, height, width)
        for axis in (self.h, self.v):
            for key in ("cross", "left", "right", "left_sq", "right_sq"):
                axis[key] = np.zeros(channels, dtype=np.float64)

    @torch.no_grad()
    def add(self, tensor: torch.Tensor) -> None:
        x = tensor.detach().float()
        if x.ndim != 4:
            raise ValueError(f"Expected B,C,H,W, got {tuple(x.shape)}")
        b, c, h, w = x.shape
        if self.sum is None:
            self._init(c, h, w)
        elif self.shape != (c, h, w):
            raise ValueError(f"Layer shape changed: {self.shape} != {(c, h, w)}")
        n = b * h * w
        flat = x.permute(1, 0, 2, 3).reshape(c, n)
        self.count += n
        self.sum += flat.sum(1).double().cpu().numpy()
        self.sumsq += flat.square().sum(1).double().cpu().numpy()
        self.cov += (flat @ flat.t()).double().cpu().numpy()
        for axis, a, b_ in ((self.h, x[:, :, :-1, :], x[:, :, 1:, :]), (self.v, x[:, :, :, :-1], x[:, :, :, 1:])):
            left = a.permute(1, 0, 2, 3).reshape(c, -1)
            right = b_.permute(1, 0, 2, 3).reshape(c, -1)
            if left.shape[1] == 0:
                continue
            axis["cross"] += (left * right).sum(1).double().cpu().numpy()
            axis["left"] += left.sum(1).double().cpu().numpy()
            axis["right"] += right.sum(1).double().cpu().numpy()
            axis["left_sq"] += left.square().sum(1).double().cpu().numpy()
            axis["right_sq"] += right.square().sum(1).double().cpu().numpy()
            axis["count"] += int(left.shape[1])
        self.diff_sum += (x[:, :, :-1, :] - x[:, :, 1:, :]).abs().sum((0, 2, 3)).double().cpu().numpy()
        self.diff_count += b * max(h - 1, 0) * w

    def finalize(self) -> dict:
        if self.sum is None or self.count == 0:
            raise RuntimeError("Empty accumulator")
        mean = self.sum / self.count
        var = np.maximum(self.sumsq / self.count - mean**2, 0.0)
        std = np.sqrt(var)
        cov = self.cov / self.count - np.outer(mean, mean)
        eig = np.linalg.eigvalsh((cov + cov.T) / 2)
        eig = np.maximum(eig, 0.0)
        total = float(eig.sum())
        eff_rank = float(total**2 / max(float((eig**2).sum()), 1e-30))
        stable_rank = float(float((eig**2).sum()) / max(float(eig.max()), 1e-30))
        out = {
            "shape": list(self.shape),
            "n_pixels": int(self.count),
            "mean": float(mean.mean()),
            "std": float(std.mean()),
            "channel_mean": mean.tolist(),
            "channel_std": std.tolist(),
            "channel_std_min": float(std.min()),
            "channel_zero_fraction": float((std < 1e-8).mean()),
            "zero_fraction": float((np.abs(mean) < 1e-8).mean()),
            "effective_rank": eff_rank,
            "effective_rank_ratio": float(eff_rank / self.shape[0]),
            "stable_rank": stable_rank,
            "roughness": float(self.diff_sum.sum() / max(self.diff_count, 1) / max(std.mean(), 1e-8)),
        }
        for name, axis in (("horizontal", self.h), ("vertical", self.v)):
            cnt = axis["count"]
            if not cnt:
                out[f"lag1_{name}"] = None
                continue
            ml, mr = axis["left"] / cnt, axis["right"] / cnt
            vl = np.maximum(axis["left_sq"] / cnt - ml**2, 0.0)
            vr = np.maximum(axis["right_sq"] / cnt - mr**2, 0.0)
            cov_lr = axis["cross"] / cnt - ml * mr
            corr = cov_lr / np.sqrt(np.maximum(vl * vr, 1e-30))
            out[f"lag1_{name}"] = float(np.nanmean(corr))
        return out


class PairAccumulator:
    """Cross-covariance between two channel stacks on the same grid."""

    def __init__(self) -> None:
        self.count = 0
        self.sa = self.sb = self.saa = self.sbb = None
        self.sab = None

    def add(self, a: torch.Tensor, b: torch.Tensor) -> None:
        if a.shape[-2:] != b.shape[-2:]:
            raise ValueError("Pair grids differ")
        ca, cb = a.shape[1], b.shape[1]
        if self.sa is None:
            self.sa = np.zeros(ca)
            self.sb = np.zeros(cb)
            self.saa = np.zeros(ca)
            self.sbb = np.zeros(cb)
            self.sab = np.zeros((ca, cb))
        fa = a.detach().float().permute(1, 0, 2, 3).reshape(ca, -1)
        fb = b.detach().float().permute(1, 0, 2, 3).reshape(cb, -1)
        n = fa.shape[1]
        self.count += n
        self.sa += fa.sum(1).double().cpu().numpy()
        self.sb += fb.sum(1).double().cpu().numpy()
        self.saa += fa.square().sum(1).double().cpu().numpy()
        self.sbb += fb.square().sum(1).double().cpu().numpy()
        self.sab += (fa @ fb.t()).double().cpu().numpy()

    def finalize(self) -> dict:
        if not self.count:
            raise RuntimeError("Empty pair accumulator")
        ma, mb = self.sa / self.count, self.sb / self.count
        va = np.maximum(self.saa / self.count - ma**2, 0.0)
        vb = np.maximum(self.sbb / self.count - mb**2, 0.0)
        cov = self.sab / self.count - np.outer(ma, mb)
        corr = cov / np.sqrt(np.maximum(np.outer(va, vb), 1e-30))
        diag = np.diag(corr)
        off = corr.copy()
        np.fill_diagonal(off, 0.0)
        return {
            "corr": corr.tolist(),
            "diag": diag.tolist(),
            "diag_mean": float(np.mean(diag)),
            "diag_min": float(np.min(diag)),
            "offdiag_abs_max": float(np.abs(off).max()),
            "offdiag_abs_mean": float(np.abs(off).mean()),
            "input_channel_std": np.sqrt(va).tolist(),
            "feature_channel_std": np.sqrt(vb).tolist(),
        }


def spatial_energy_radius(energy: np.ndarray, center: tuple[int, int]) -> dict:
    """Radius (px) containing 50%/90% of the spatial gradient energy."""
    e = np.maximum(np.asarray(energy, dtype=np.float64), 0.0)
    total = float(e.sum())
    if total <= 0:
        return {"r50": None, "r90": None, "center_share": None}
    h, w = e.shape
    cy, cx = center
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    order = np.argsort(radius.ravel())
    cumulative = np.cumsum(e.ravel()[order]) / total
    sorted_r = radius.ravel()[order]
    return {
        "r50": float(sorted_r[int(np.searchsorted(cumulative, 0.5))]),
        "r90": float(sorted_r[int(np.searchsorted(cumulative, 0.9))]),
        "center_share": float(e[cy, cx] / total),
    }


# ---------------------------------------------------------------------------
# model / data plumbing
# ---------------------------------------------------------------------------
def arm_dataset(arm: str, records: list[dict], *, augment: bool, preload: bool = True):
    stats = load_stats(STATS)
    if arm == "use":
        ds = USESegDataset(records, stats, augment=augment, seed=SEED, preload=preload)
        ds.mixed = False
        return ds
    if arm == "direct":
        return S2SegDataset(records, stats, augment=augment, seed=SEED, preload=preload)
    raise ValueError(arm)


def load_arm(arm: str, device: torch.device):
    path = DIRECT_CKPT if arm == "direct" else USE_CKPT
    model, ckpt = load_checkpoint(path, str(device))
    model.eval()
    return model, ckpt


def visual_of(model, arm: str):
    return model if arm == "direct" else model.visual


def activation_specs(model, arm: str) -> list[tuple[str, torch.nn.Module]]:
    visual = visual_of(model, arm)
    specs = [
        ("mit.stage1", visual.model.segformer.encoder.layer_norm[0]),
        ("mit.stage2", visual.model.segformer.encoder.layer_norm[1]),
        ("detail.stem", visual.stem),
    ]
    if arm == "use":
        specs = [("use.features", model.use), ("bridge", model.bridge)] + specs
    return specs


def _tensor_from_output(output):
    if isinstance(output, dict):
        value = output.get("features", output)
    elif isinstance(output, tuple):
        value = output[0]
    else:
        value = output
    if torch.is_tensor(value) and value.ndim == 3:
        # MiT stage layer-norm emits [B, H*W, C]; the diagnostic works in BCHW.
        batch, tokens, channels = value.shape
        side = int(round(math.sqrt(tokens)))
        if side * side == tokens:
            value = value.transpose(1, 2).reshape(batch, channels, side, side)
    return value


def capture(model, x, arm: str) -> tuple[dict, dict]:
    store: dict[str, torch.Tensor] = {}
    handles = []
    for name, module in activation_specs(model, arm):
        def hook(_module, _inputs, output, _name=name):
            value = _tensor_from_output(output)
            if torch.is_tensor(value):
                store[_name] = value
        handles.append(module.register_forward_hook(hook))
    try:
        out = model(x)
    finally:
        for handle in handles:
            handle.remove()
    visual = visual_of(model, arm)
    if visual._decoder_feat is not None:
        store["decoder"] = visual._decoder_feat
    return out, store


def batch_input(batch: dict, arm: str):
    image = batch["image"]
    return image if arm == "direct" else image


def clone_batch(batch: dict) -> dict:
    clone = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            inner = {}
            for k, v in value.items():
                if torch.is_tensor(v):
                    inner[k] = v.clone()
                elif isinstance(v, list):
                    inner[k] = [list(x) if isinstance(x, list) else x for x in v]
                else:
                    inner[k] = v
            clone[key] = type(value)(inner) if isinstance(value, OpticalBatch) else inner
        elif torch.is_tensor(value):
            clone[key] = value.clone()
        else:
            clone[key] = value
    return clone


def probability(model, batch_or_input, arm: str) -> torch.Tensor:
    out = model(batch_or_input)
    return torch.sigmoid(out["logits"].float()[:, 0])


def permute_batch(batch: dict, perm: list[int], mode: str, arm: str) -> dict:
    """Joint / band-only / metadata-only permutation of a collated batch."""
    out = clone_batch(batch)
    index = torch.as_tensor(perm, device=out["image"].device if arm == "direct" else out["image"]["x"].device)
    if arm == "direct":
        if mode in ("joint", "band"):
            out["image"] = out["image"][:, index]
        return out
    image = out["image"]
    if mode in ("joint", "band"):
        for key in ("x", "band_valid", "pixel_valid"):
            image[key] = image[key][:, index]
        if mode == "joint":
            for key in ("band_metadata", "known_mask"):
                image[key] = image[key][:, index]
            image["band_ids"] = [[image["band_ids"][i][j] for j in perm] for i in range(len(image["band_ids"]))]
    elif mode == "metadata":
        for key in ("band_metadata", "known_mask"):
            image[key] = image[key][:, index]
    else:
        raise ValueError(mode)
    return out


def subset_batch(batch: dict, bands: tuple[str, ...]) -> dict:
    """Physical band removal for a collated USE batch (no new observation object)."""
    index = [S2_10_BANDS.index(b) for b in bands]
    image = batch["image"]
    new = OpticalBatch(
        {
            "x": image["x"][:, index],
            "band_metadata": image["band_metadata"][:, index],
            "known_mask": image["known_mask"][:, index],
            "band_valid": image["band_valid"][:, index],
            "pixel_valid": image["pixel_valid"][:, index],
            "band_ids": [[names[i] for i in index] for names in image["band_ids"]],
            "spatial_metadata": image["spatial_metadata"],
        }
    )
    out = {key: value for key, value in batch.items() if key != "image"}
    out["image"] = new
    return out


def masked_batch(batch: dict, remove: Iterable[int]) -> dict:
    out = clone_batch(batch)
    image = out["image"]
    index = torch.as_tensor(list(remove), device=image["x"].device)
    image["band_valid"][:, index] = False
    image["pixel_valid"][:, index] = False
    image["x"][:, index] = float("nan")
    return out


def prob_delta(p0: torch.Tensor, p1: torch.Tensor, valid: torch.Tensor) -> dict:
    v = valid.bool()
    diff = (p0 - p1).abs()
    if not v.any():
        return {"mean_abs_diff": None, "max_abs_diff": None, "iou_agreement": None}
    pred0 = (p0 >= 0.5).cpu().numpy()
    pred1 = (p1 >= 0.5).cpu().numpy()
    vv = v.cpu().numpy()
    c = confusion(pred1, pred0, vv)
    denom = c["tp"] + c["fp"] + c["fn"]
    return {
        "mean_abs_diff": float(diff[v].mean()),
        "max_abs_diff": float(diff[v].max()),
        "iou_agreement": float(c["tp"] / denom) if denom else None,
    }


def group_name(name: str, arm: str) -> str:
    if name.startswith("use.tokenizer"):
        return "use.tokenizer"
    if name.startswith("use.metadata"):
        return "use.metadata"
    if name.startswith("use.attention"):
        return "use.attention"
    if name.startswith("use.pool") or name.startswith("use.project"):
        return "use.pool"
    if name.startswith("use.fuse"):
        return "use.fuse"
    if name.startswith("bridge"):
        return "bridge"
    if "patch_embeddings.0" in name:
        return "mit.embed1"
    if "encoder.block.0" in name or "encoder.layer_norm.0" in name:
        return "mit.stage1"
    if "encoder.block.1" in name or "encoder.layer_norm.1" in name:
        return "mit.stage2"
    if "encoder.block.2" in name or "encoder.block.3" in name:
        return "mit.stage3_4"
    if "stem" in name:
        return "detail.stem"
    if "decode_head" in name:
        return "decoder"
    return "other"


def grad_group_norms(model, arm: str, cfg: dict) -> dict:
    optim = cfg["segformer"]["optim"]
    encoder_lr = float(optim["encoder_lr"])
    head_lr = float(optim["head_lr"])
    groups: dict[str, dict] = defaultdict(lambda: {"grad_sq": 0.0, "param_sq": 0.0, "n": 0})
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        key = group_name(name, arm)
        groups[key]["param_sq"] += float(param.detach().double().square().sum())
        groups[key]["n"] += int(param.numel())
        if param.grad is not None:
            groups[key]["grad_sq"] += float(param.grad.detach().double().square().sum())
    total_grad = sum(v["grad_sq"] for v in groups.values())
    out = {}
    for key, value in groups.items():
        grad_norm = math.sqrt(value["grad_sq"])
        param_norm = math.sqrt(value["param_sq"])
        lr = encoder_lr if key.startswith("mit.") else head_lr
        out[key] = {
            "grad_norm": grad_norm,
            "param_norm": param_norm,
            "n_params": value["n"],
            "grad_share": (value["grad_sq"] / total_grad) if total_grad else None,
            "relative_update": (lr * grad_norm / param_norm) if param_norm else None,
        }
    return out


# ---------------------------------------------------------------------------
# representation diagnostics
# ---------------------------------------------------------------------------
def valid_agreement(batch: dict, arm: str) -> dict | None:
    if arm == "direct":
        return None
    image = batch["image"]
    cache = batch["valid"].bool()
    quality = (image["pixel_valid"].bool() & image["band_valid"][:, :, None, None]).any(1)
    inter = (cache & quality).sum().item()
    union = (cache | quality).sum().item()
    return {
        "iou": inter / union if union else None,
        "cache_only_px": int((cache & ~quality).sum()),
        "quality_only_px": int((quality & ~cache).sum()),
    }


@torch.no_grad()
def analyse_activations(model, arm: str, records: list[dict], device: torch.device, out_dir: Path) -> dict:
    model.eval()
    ds = arm_dataset(arm, records, augment=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate, num_workers=0)
    accumulators: dict[str, FeatureAccumulator] = {}
    pair = PairAccumulator() if arm == "use" else None
    agreements = []
    for batch in loader:
        x = batch_input(batch, arm).to(device)
        _, store = capture(model, x, arm)
        input_tensor = x if arm == "direct" else x["x"]
        for name, tensor in [("input", input_tensor)] + list(store.items()):
            if name == "logits" or not torch.is_tensor(tensor):
                continue
            accumulators.setdefault(name, FeatureAccumulator()).add(tensor)
        if pair is not None and "bridge" in store:
            pair.add(input_tensor, store["bridge"])
        agreement = valid_agreement(batch, arm)
        if agreement is not None:
            agreements.append(agreement)
    stats = {name: acc.finalize() for name, acc in accumulators.items()}
    result = {
        "arm": arm,
        "checkpoint": str(DIRECT_CKPT if arm == "direct" else USE_CKPT).replace("\\", "/"),
        "layers": stats,
        "valid_agreement": {
            key: (float(np.mean([a[key] for a in agreements if a[key] is not None])) if agreements and any(a[key] is not None for a in agreements) else None)
            for key in ("iou",)
        } if agreements else None,
    }
    if pair is not None:
        result["input_bridge_pair"] = pair.finalize()
    # Bridge weight geometry.
    if arm == "use":
        weight = model.bridge.weight.detach().float().cpu().numpy()[:, :, 0, 0]
        bias = model.bridge.bias.detach().float().cpu().numpy()
        singular = np.linalg.svd(weight, compute_uv=False)
        result["bridge_weight"] = {
            "shape": list(weight.shape),
            "sha256": tensor_sha256(model.bridge.weight) + ":" + tensor_sha256(model.bridge.bias),
            "singular_values": singular.tolist(),
            "condition_number": float(singular[0] / singular[-1]) if singular[-1] > 0 else None,
            "effective_rank": float(singular.sum() ** 2 / max(float((singular**2).sum()), 1e-30)),
            "frobenius_norm": float(np.linalg.norm(weight)),
            "bias_abs_mean": float(np.abs(bias).mean()),
            "bias_abs_max": float(np.abs(bias).max()),
        }
        input_std = np.asarray(stats["input"]["channel_std"])
        bridge_std = np.asarray(stats["bridge"]["channel_std"])
        ratio = bridge_std / np.maximum(input_std, 1e-8)
        result["bridge_scale"] = {
            "channel_std_ratio": ratio.tolist(),
            "log2_ratio": np.log2(np.maximum(ratio, 1e-8)).tolist(),
            "mean_abs_log2_ratio": float(np.abs(np.log2(np.maximum(ratio, 1e-8))).mean()),
            "channels_below_half_input_std": int((ratio < 0.5).sum()),
            "channels_above_double_input_std": int((ratio > 2.0).sum()),
        }
    write_json(out_dir / "interface" / f"layer_stats_{arm}.json", result)
    return result


@torch.no_grad()
def valid_mask_summary(arm: str, records: list[dict], out_dir: Path) -> dict:
    ds = arm_dataset(arm, records, augment=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate, num_workers=0)
    rows = []
    for batch in loader:
        agreement = valid_agreement(batch, arm)
        if agreement:
            rows.append(agreement)
    if not rows:
        result = {"arm": arm, "supported": False}
    else:
        result = {
            "arm": arm,
            "supported": True,
            "mean_iou": float(np.mean([r["iou"] for r in rows])),
            "min_iou": float(np.min([r["iou"] for r in rows])),
            "cache_only_px": int(sum(r["cache_only_px"] for r in rows)),
            "quality_only_px": int(sum(r["quality_only_px"] for r in rows)),
        }
    write_json(out_dir / "interface" / f"valid_masks_{arm}.json", result)
    return result


def analyse_gradients(model, arm: str, records: list[dict], device: torch.device, out_dir: Path, cfg: dict) -> dict:
    model.eval()
    subset = select_diagnostic_tiles(records, 32)
    ds = arm_dataset(arm, subset, augment=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate, num_workers=0)
    input_acc = FeatureAccumulator()
    input_band_norms = []
    input_band_nonzero = []
    top10_share = []
    internal_acc: dict[str, FeatureAccumulator] = {}
    group_totals: dict[str, dict] = defaultdict(lambda: {"grad_sq": 0.0, "param_sq": 0.0, "n": 0})
    for batch in loader:
        model.zero_grad(set_to_none=True)
        image = batch_input(batch, arm).to(device)
        target_tensor = image if arm == "direct" else image["x"]
        target_tensor.requires_grad_(True)
        retain: dict[str, torch.Tensor] = {}
        handles = []
        for name, module in activation_specs(model, arm):
            def hook(_module, _inputs, output, _name=name):
                value = _tensor_from_output(output)
                if torch.is_tensor(value):
                    value.retain_grad()
                    retain[_name] = value
            handles.append(module.register_forward_hook(hook))
        try:
            out = model(image)
        finally:
            for handle in handles:
                handle.remove()
        mask = batch["mask"].to(device)
        valid = batch["valid"].to(device)
        loss = _seg_loss(cfg, out["logits"], mask, valid, boundary=True)
        loss.backward()
        grad = target_tensor.grad
        if grad is not None:
            input_acc.add(grad)
            per_band = grad.detach().double().square().sum((0, 2, 3)).sqrt().cpu().numpy()
            denom = per_band.sum()
            input_band_norms.append((per_band / denom).tolist() if denom else per_band.tolist())
            input_band_nonzero.append((grad.detach().abs() > 0).float().mean((0, 2, 3)).cpu().numpy().tolist())
            flat = grad.detach().abs().flatten(1)
            k = max(1, int(flat.shape[1] * 0.1))
            top = torch.topk(flat, k, dim=1).values.sum(1)
            top10_share.append((top / flat.sum(1).clamp_min(1e-30)).cpu().numpy().tolist())
        for name, tensor in retain.items():
            if tensor.grad is not None:
                internal_acc.setdefault(name, FeatureAccumulator()).add(tensor.grad)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            key = group_name(name, arm)
            group_totals[key]["param_sq"] += float(param.detach().double().square().sum())
            group_totals[key]["n"] += int(param.numel())
            if param.grad is not None:
                group_totals[key]["grad_sq"] += float(param.grad.detach().double().square().sum())
    total_grad = sum(v["grad_sq"] for v in group_totals.values())
    optim = cfg["segformer"]["optim"]
    param_groups = {}
    for key, value in group_totals.items():
        grad_norm = math.sqrt(value["grad_sq"])
        param_norm = math.sqrt(value["param_sq"])
        lr = float(optim["encoder_lr"]) if key.startswith("mit.") else float(optim["head_lr"])
        param_groups[key] = {
            "grad_norm": grad_norm,
            "param_norm": param_norm,
            "n_params": value["n"],
            "grad_share": (value["grad_sq"] / total_grad) if total_grad else None,
            "relative_update": (lr * grad_norm / param_norm) if param_norm else None,
        }
    result = {
        "arm": arm,
        "n_tiles": len(subset),
        "input_gradient": input_acc.finalize(),
        "input_band_share": np.mean(input_band_norms, axis=0).tolist() if input_band_norms else None,
        "input_band_nonzero_fraction": np.mean(input_band_nonzero, axis=0).tolist() if input_band_nonzero else None,
        "top10pixel_energy_share": float(np.mean(top10_share)) if top10_share else None,
        "internal_gradient": {name: acc.finalize() for name, acc in internal_acc.items()},
        "parameter_groups": param_groups,
    }
    write_json(out_dir / "interface" / f"gradient_stats_{arm}.json", result)
    return result


def select_erf_tiles(tiles: list[dict]) -> list[dict]:
    chosen = []
    for size in ("1_4", "5_25", "26_100", "101_plus"):
        candidates = [t for t in tiles if t["size_class"] == size]
        if candidates:
            chosen.append(sorted(candidates, key=lambda t: t["selection_key"])[0])
    return chosen


def analyse_erf(model, arm: str, tiles: list[dict], device: torch.device, out_dir: Path, cfg: dict) -> dict:
    model.eval()
    records_by_id = {r["sample_id"]: r for r in read_manifest(MANIFEST)}
    rows = []
    for tile in select_erf_tiles(tiles):
        record = records_by_id[tile["sample_id"]]
        ds = arm_dataset(arm, [record], augment=False)
        batch = collate([ds[0]])
        model.zero_grad(set_to_none=True)
        image = batch_input(batch, arm).to(device)
        target = image if arm == "direct" else image["x"]
        target.requires_grad_(True)
        retain = {}
        handles = []
        for name, module in activation_specs(model, arm):
            def hook(_module, _inputs, output, _name=name):
                value = _tensor_from_output(output)
                if torch.is_tensor(value):
                    value.retain_grad()
                    retain[_name] = value
            handles.append(module.register_forward_hook(hook))
        try:
            out = model(image)
        finally:
            for handle in handles:
                handle.remove()
        cy, cx = out["logits"].shape[-2] // 2, out["logits"].shape[-1] // 2
        out["logits"][0, 0, cy, cx].backward()
        grad = target.grad
        energy = grad.detach().double().square().sum(1)[0].cpu().numpy()
        row = {
            "sample_id": tile["sample_id"],
            "event_id": tile["event_id"],
            "size_class": tile["size_class"],
            "input": spatial_energy_radius(energy, (cy, cx)),
            "interface": {},
        }
        for name, tensor in retain.items():
            if tensor.grad is not None:
                e = tensor.grad.detach().double().square().sum(1)[0].cpu().numpy()
                row["interface"][name] = spatial_energy_radius(e, (cy, cx))
        rows.append(row)
    result = {"arm": arm, "center": "64,64 (single tile per size class)", "tiles": rows}
    write_json(out_dir / "interface" / f"erf_{arm}.json", result)
    return result


@torch.no_grad()
def analyse_permutation(arm: str, records: list[dict], device: torch.device, out_dir: Path) -> dict:
    model, _ = load_arm(arm, device)
    model.eval()
    ds = arm_dataset(arm, records, augment=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate, num_workers=0)
    rng = random.Random(SELECT_SEED)
    perms = [rng.sample(range(10), 10) for _ in range(3)]
    rows = []
    for batch in loader:
        image = batch_input(batch, arm).to(device)
        with torch.no_grad():
            out = model(image)
            p0 = torch.sigmoid(out["logits"].float()[:, 0])
        valid = batch["valid"].to(device)
        record = {"n": int(valid.sum())}
        for mode in ("joint", "band", "metadata"):
            mode_rows = []
            for perm in perms:
                permuted = permute_batch(batch, perm, mode, arm)
                p1 = probability(model, batch_input(permuted, arm).to(device), arm)
                mode_rows.append(prob_delta(p0, p1, valid))
            record[mode] = {
                key: float(np.mean([r[key] for r in mode_rows if r[key] is not None]))
                if any(r[key] is not None for r in mode_rows) else None
                for key in ("mean_abs_diff", "max_abs_diff", "iou_agreement")
            }
        rows.append(record)
    total = sum(r["n"] for r in rows)
    result = {
        "arm": arm,
        "permutations": perms,
        "n_pixels": total,
        "joint": {k: float(np.average([r["joint"][k] for r in rows], weights=[r["n"] for r in rows])) for k in ("mean_abs_diff", "max_abs_diff", "iou_agreement")},
        "band_only": {k: float(np.average([r["band"][k] for r in rows], weights=[r["n"] for r in rows])) for k in ("mean_abs_diff", "max_abs_diff", "iou_agreement")},
        "metadata_only": {k: float(np.average([r["metadata"][k] for r in rows], weights=[r["n"] for r in rows])) for k in ("mean_abs_diff", "max_abs_diff", "iou_agreement")},
    }
    write_json(out_dir / "interface" / f"permutation_{arm}.json", result)
    return result


@torch.no_grad()
def analyse_missing_bands(arm: str, records: list[dict], device: torch.device, out_dir: Path) -> dict:
    model, _ = load_arm(arm, device)
    model.eval()
    ds = arm_dataset(arm, records, augment=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate, num_workers=0)
    if arm == "direct":
        result = {
            "arm": arm,
            "supported": False,
            "reason": "direct10 replaces the first patch-embedding with exactly 10 input channels; it has no variable-band or band-valid interface",
        }
        write_json(out_dir / "interface" / f"missing_bands_{arm}.json", result)
        return result
    results = {name: [] for name in ("rgb", "rgb_nir", "six", "unseen5")}
    masked = []
    for batch in loader:
        image = batch_input(batch, arm).to(device)
        with torch.no_grad():
            p_full = probability(model, image, arm)
        valid = batch["valid"].to(device)
        for name in results:
            sub = subset_batch(batch, SUBSETS[name])
            p_sub = probability(model, batch_input(sub, arm).to(device), arm)
            results[name].append(prob_delta(p_full, p_sub, valid))
        remove = [S2_10_BANDS.index(b) for b in ("B05", "B06", "B07", "B8A", "B11", "B12")]
        masked_batch_ = masked_batch(batch, remove)
        p_mask = probability(model, batch_input(masked_batch_, arm).to(device), arm)
        p_absent = probability(model, batch_input(subset_batch(batch, ("B02", "B03", "B04", "B08")), arm).to(device), arm)
        masked.append({
            "masked_vs_full": prob_delta(p_full, p_mask, valid),
            "masked_vs_absent": prob_delta(p_absent, p_mask, valid),
            "finite": bool(torch.isfinite(p_mask).all()),
        })
    out = {"arm": arm, "supported": True, "subsets": {}}
    for name, rows in results.items():
        out["subsets"][name] = {
            key: float(np.mean([r[key] for r in rows if r[key] is not None]))
            for key in ("mean_abs_diff", "max_abs_diff", "iou_agreement")
        }
    out["band_valid_masking"] = {
        "masked_vs_full": {k: float(np.mean([r["masked_vs_full"][k] for r in masked])) for k in ("mean_abs_diff", "iou_agreement")},
        "masked_vs_absent": {k: float(np.mean([r["masked_vs_absent"][k] for r in masked])) for k in ("mean_abs_diff", "iou_agreement")},
        "all_finite": bool(all(r["finite"] for r in masked)),
    }
    write_json(out_dir / "interface" / f"missing_bands_{arm}.json", out)
    return out


@torch.no_grad()
def analyse_valid_region(arm: str, records: list[dict], device: torch.device, out_dir: Path) -> dict:
    model, _ = load_arm(arm, device)
    model.eval()
    ds = arm_dataset(arm, select_diagnostic_tiles(records, 32), augment=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate, num_workers=0)
    rows = []
    for batch in loader:
        base = clone_batch(batch)
        region = (slice(None), slice(None), slice(32, 64), slice(32, 64))
        image = base["image"]
        if arm == "direct":
            image[region] = 0.0
        else:
            image["pixel_valid"][region] = False
            image["x"][region] = 0.0
        p_zero = probability(model, batch_input(base, arm).to(device), arm)
        nan_batch = clone_batch(base)
        if arm == "direct":
            nan_batch["image"][region] = float("nan")
        else:
            nan_batch["image"]["x"][region] = float("nan")
        out_nan = model(batch_input(nan_batch, arm).to(device))
        p_nan = torch.sigmoid(out_nan["logits"].float()[:, 0])
        valid = batch["valid"].to(device).clone()
        valid[:, 32:64, 32:64] = False
        # pixels far from the injected rectangle: exclude a 4 px dilation
        far = valid.clone()
        far[:, 28:68, 28:68] = False
        rows.append({
            "valid_px": int(valid.sum()),
            "far_px": int(far.sum()),
            "nan_finite": bool(torch.isfinite(p_nan).all()),
            "nan_max_diff_valid": float((p_zero - p_nan).abs()[valid].max()) if valid.any() else None,
            "nan_max_diff_far": float((p_zero - p_nan).abs()[far].max()) if far.any() else None,
            "input_valid_supported": "input_valid" in out_nan,
            "input_valid_matches": bool(
                (out_nan["input_valid"].detach().cpu()
                 == (nan_batch["image"]["pixel_valid"].bool()
                     & nan_batch["image"]["band_valid"][:, :, None, None]).any(1)).all()
            ) if arm == "use" else None,
        })
    result = {
        "arm": arm,
        "region": "channels all, rows 32:64, cols 32:64; same synthetic pixel_valid mask in both runs; only values differ (0 vs NaN)",
        "nan_finite_all": bool(all(r["nan_finite"] for r in rows)),
        "nan_max_diff_valid_max": float(max(r["nan_max_diff_valid"] for r in rows if r["nan_max_diff_valid"] is not None)),
        "nan_max_diff_far_max": float(max(r["nan_max_diff_far"] for r in rows if r["nan_max_diff_far"] is not None)),
        "input_valid_supported": bool(all(r["input_valid_supported"] for r in rows)),
        "input_valid_matches": bool(all(r["input_valid_matches"] for r in rows)) if arm == "use" else None,
    }
    # fully-invalid sample behaviour
    one = clone_batch(collate([ds[0]]))
    if arm == "direct":
        one["image"][:] = float("nan")
        try:
            out = model(one["image"].to(device))
            result["fully_invalid"] = {"accepted": True, "finite": bool(torch.isfinite(out["logits"]).all())}
        except Exception as exc:  # pragma: no cover - diagnostic record
            result["fully_invalid"] = {"accepted": False, "error": type(exc).__name__}
    else:
        one["image"]["pixel_valid"][:] = False
        one["image"]["x"][:] = float("nan")
        try:
            model(one["image"].to(device))
            result["fully_invalid"] = {"accepted": True}
        except Exception as exc:
            result["fully_invalid"] = {"accepted": False, "error": type(exc).__name__}
    write_json(out_dir / "interface" / f"valid_region_{arm}.json", result)
    return result


def run_represent(out_dir: Path = DIAG_ROOT, device_name: str | None = None) -> dict:
    device = _device(device_name)
    tiles = load_tiles(out_dir)
    records = read_manifest(MANIFEST)
    by_id = {r["sample_id"]: r for r in records}
    selected = [by_id[t["sample_id"]] for t in tiles["tiles"]]
    out_dir.joinpath("interface").mkdir(parents=True, exist_ok=True)
    identity = {}
    for arm, path, config in (("direct", DIRECT_CKPT, DIRECT_CONFIG), ("use", USE_CKPT, USE_CONFIG)):
        model, ckpt = load_arm(arm, device)
        run_dir = path.parent
        run_summary = read_json(run_dir / "train_summary.json") if (run_dir / "train_summary.json").is_file() else {}
        identity[arm] = {
            "checkpoint": str(path).replace("\\", "/"),
            "checkpoint_sha256": hash_file(path),
            "config": str(config).replace("\\", "/"),
            "config_sha256": hash_file(config),
            "checkpoint_step": ckpt.get("global_step"),
            "checkpoint_best_step": ckpt.get("best_step"),
            "checkpoint_best_val_iou": ckpt.get("best_val_iou"),
            "run_stop_step": run_summary.get("global_step"),
            "run_best_step": run_summary.get("best_step"),
            "run_best_val_iou": run_summary.get("best_val_iou"),
            "params": model.count_params(),
            "model_identity": model_identity(ckpt["config"]["segformer"], model),
        }
        if arm == "use":
            identity[arm]["bridge_weight_sha256"] = tensor_sha256(model.bridge.weight)
            identity[arm]["bridge_bias_sha256"] = tensor_sha256(model.bridge.bias)
        else:
            identity[arm]["first_conv_sha256"] = tensor_sha256(model.model.segformer.encoder.patch_embeddings[0].proj.weight)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_json(out_dir / "interface/checkpoint_identity.json", identity)

    cfg_direct = load_segformer_config(DIRECT_CONFIG)
    cfg_use = load_segformer_config(USE_CONFIG)
    results = {}
    for arm, cfg in (("direct", cfg_direct), ("use", cfg_use)):
        model, _ = load_arm(arm, device)
        results[f"activations_{arm}"] = analyse_activations(model, arm, selected, device, out_dir)
        results[f"valid_masks_{arm}"] = valid_mask_summary(arm, selected, out_dir)
        results[f"gradients_{arm}"] = analyse_gradients(model, arm, selected, device, out_dir, cfg)
        results[f"erf_{arm}"] = analyse_erf(model, arm, tiles["tiles"], device, out_dir, cfg)
        results[f"permutation_{arm}"] = analyse_permutation(arm, selected, device, out_dir)
        results[f"missing_bands_{arm}"] = analyse_missing_bands(arm, selected, device, out_dir)
        results[f"valid_region_{arm}"] = analyse_valid_region(arm, selected, device, out_dir)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = build_summary(out_dir)
    write_json(out_dir / "summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# 300-update fitting probes
# ---------------------------------------------------------------------------
def probe_settings(cfg: dict) -> dict:
    optim = cfg["segformer"]["optim"]
    return {
        "updates": PROBE_UPDATES,
        "batch_size": PROBE_BATCH,
        "schedule_updates": PROBE_UPDATES,
        "warmup_updates": max(1, int(PROBE_UPDATES * float(optim["warmup_frac"]))),
        "eval_every_updates": PROBE_EVAL_EVERY,
        "encoder_lr": float(optim["encoder_lr"]),
        "head_lr": float(optim["head_lr"]),
        "weight_decay": float(optim["weight_decay"]),
        "grad_clip": float(optim["grad_clip"]),
        "seed": SEED,
        "note": "fitting probe with a 300-update horizon; not the formal 2000-update run and not a generalisation score",
    }


@torch.no_grad()
def evaluate_subset(model, ds, cfg: dict, device: torch.device, batch_size: int = 8) -> dict:
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    threshold = float(cfg["segformer"]["eval"]["threshold"])
    counters = []
    by_event = defaultdict(list)
    for batch in loader:
        out = model(batch["image"].to(device))
        prob = torch.sigmoid(out["logits"].float()[:, 0]).cpu().numpy()
        gt = batch["mask"].numpy().astype(bool)
        valid = batch["valid"].numpy().astype(bool)
        for i, event in enumerate(batch["event_id"]):
            counter = confusion(prob[i] >= threshold, gt[i], valid[i])
            counters.append(counter)
            by_event[event].append(counter)
    events = {event: summarize(cs) for event, cs in sorted(by_event.items())}
    values = [m["iou"] for m in events.values() if m["iou"] is not None and np.isfinite(m["iou"])]
    return {"pooled": summarize(counters), "by_event": events, "event_mean_iou": float(np.mean(values)) if values else None}


def freeze_bn(model) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def probe_arm(arm: str, out_dir: Path = DIAG_ROOT, *, force: bool = False, device_name: str | None = None) -> dict:
    cfg_path = DIRECT_CONFIG if arm == "direct" else USE_CONFIG
    cfg = load_segformer_config(cfg_path)
    settings = probe_settings(cfg)
    device = _device(device_name)
    tiles = load_tiles(out_dir)
    by_id = {r["sample_id"]: r for r in read_manifest(MANIFEST)}
    fit_records = [by_id[t["sample_id"]] for t in tiles["tiles"] if t["probe_role"] == "fit"]
    check_records = [by_id[t["sample_id"]] for t in tiles["tiles"] if t["probe_role"] == "check"]
    run_dir = out_dir / "probes" / f"{arm}_{PROBE_UPDATES}"
    if run_dir.exists() and any(run_dir.iterdir()):
        if not force:
            raise FileExistsError(f"Probe directory is not empty (use --force for the diagnostic dir): {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = copy.deepcopy(cfg)
    resolved["segformer"]["train"].update({
        "stop_updates": PROBE_UPDATES,
        "schedule_updates": PROBE_UPDATES,
        "max_steps": PROBE_UPDATES,
        "budget_mode": "updates",
    })
    write_json(run_dir / "resolved_config.json", resolved)
    set_seed(SEED, True)
    model = build_model(cfg["segformer"]).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    param_count = model.count_params()
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    init_hash = state_dict_sha256(init_state)
    torch.save({"state_dict": init_state, "seed": SEED, "arm": arm, "config": cfg}, run_dir / "init.pt")
    optimizer = _make_optimizer(model, cfg)
    scheduler = _make_scheduler(optimizer, cfg, PROBE_UPDATES)
    fit_ds = arm_dataset(arm, fit_records, augment=True, preload=True)
    check_ds = arm_dataset(arm, check_records, augment=False, preload=True)
    fit_eval_ds = arm_dataset(arm, fit_records, augment=False, preload=True)
    started = time.time()
    history, update_trace = [], []
    best_fit, best_fit_step, best_check, best_check_step = -1.0, -1, -1.0, -1
    step, epoch = 0, 0
    log = lambda text: print(f"[probe:{arm}] {text}", flush=True)
    while step < PROBE_UPDATES:
        fit_ds.set_epoch(epoch)
        loader = DataLoader(
            fit_ds,
            batch_size=PROBE_BATCH,
            shuffle=True,
            generator=torch.Generator().manual_seed(SEED * 100_003 + epoch),
            collate_fn=collate,
            num_workers=0,
        )
        for batch in loader:
            if step >= PROBE_UPDATES:
                break
            model.train()
            freeze_bn(model)
            optimizer.zero_grad(set_to_none=True)
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            valid = batch["valid"].to(device)
            out = model(image)
            loss = _seg_loss(cfg, out["logits"], mask, valid, boundary=True)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite probe loss")
            loss.backward()
            groups = grad_group_norms(model, arm, cfg)
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            update_trace.append({
                "global_step": step,
                "epoch": epoch,
                "train_loss": float(loss.detach()),
                "lr": [float(g["lr"]) for g in optimizer.param_groups],
                "grad_norm_total": float(math.sqrt(sum(v["grad_norm"] ** 2 for v in groups.values()))),
                "grad_norm_bridge": groups.get("bridge", {}).get("grad_norm"),
                "grad_share_use": float(sum((v["grad_share"] or 0.0) for k, v in groups.items() if k.startswith("use."))) if arm == "use" else None,
                "grad_share_bridge": groups.get("bridge", {}).get("grad_share"),
                "grad_share_mit1": groups.get("mit.embed1", {}).get("grad_share"),
                "grad_share_stem": groups.get("detail.stem", {}).get("grad_share"),
                "seconds": time.time() - started,
            })
            if step % PROBE_EVAL_EVERY == 0 or step == PROBE_UPDATES:
                fit_metrics = evaluate_subset(model, fit_eval_ds, cfg, device)
                check_metrics = evaluate_subset(model, check_ds, cfg, device)
                row = {
                    "global_step": step,
                    "epoch": epoch,
                    "train_loss": float(loss.detach()),
                    "fit": fit_metrics,
                    "check": check_metrics,
                    "seconds": time.time() - started,
                }
                history.append(row)
                improved_fit = fit_metrics["pooled"]["iou"] > best_fit
                improved_check = check_metrics["pooled"]["iou"] > best_check
                if improved_fit:
                    best_fit, best_fit_step = float(fit_metrics["pooled"]["iou"]), step
                if improved_check:
                    best_check, best_check_step = float(check_metrics["pooled"]["iou"]), step
                payload = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "global_step": step,
                    "best_fit_iou": best_fit,
                    "best_check_iou": best_check,
                    "config": cfg,
                    "probe": settings,
                }
                torch.save(payload, run_dir / "last.pt")
                if improved_fit:
                    torch.save({**payload, "model": {k: v.detach().clone() for k, v in model.state_dict().items()}}, run_dir / "best_fit.pt")
                if improved_check:
                    torch.save({**payload, "model": {k: v.detach().clone() for k, v in model.state_dict().items()}}, run_dir / "best_check.pt")
                log(f"step={step} loss={row['train_loss']:.4f} fit_iou={fit_metrics['pooled']['iou']:.4f} check_iou={check_metrics['pooled']['iou']:.4f}")
                write_json(run_dir / "metrics.json", history)
        epoch += 1
    final_fit = evaluate_subset(model, fit_eval_ds, cfg, device)
    final_check = evaluate_subset(model, check_ds, cfg, device)
    torch.save({"model": model.state_dict(), "config": cfg, "probe": settings, "seed": SEED}, run_dir / "last.pt")
    (run_dir / "per_update.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in update_trace), encoding="utf-8")
    write_json(run_dir / "metrics.json", history)
    meta = {
        "arm": arm,
        "experiment": cfg["segformer"]["experiment"],
        "config": str(cfg_path).replace("\\", "/"),
        "config_sha256": hash_file(cfg_path),
        "resolved_config": str(run_dir / "resolved_config.json").replace("\\", "/"),
        "checkpoint_used_for_representation": str(DIRECT_CKPT if arm == "direct" else USE_CKPT).replace("\\", "/"),
        "checkpoint_sha256": hash_file(DIRECT_CKPT if arm == "direct" else USE_CKPT),
        "tiles_manifest": str(out_dir / "selection/diagnostic_tiles.json").replace("\\", "/"),
        "tiles_sha256": hash_file(out_dir / "selection/diagnostic_tiles.json"),
        "stats_sha256": hash_file(STATS),
        "quality_index_sha256": hash_file(QUALITY_INDEX),
        "n_fit": len(fit_records),
        "n_check": len(check_records),
        "fit_samples": [r["sample_id"] for r in fit_records],
        "check_samples": [r["sample_id"] for r in check_records],
        "params": param_count,
        "init_state_sha256": init_hash,
        "probe": settings,
        "seed_info": {"seed": SEED, "deterministic": True},
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "model_identity": model_identity(cfg["segformer"], model),
    }
    write_json(run_dir / "run_meta.json", meta)
    summary = {
        **meta,
        "started_at": started,
        "seconds": time.time() - started,
        "final_fit": final_fit,
        "final_check": final_check,
        "best_fit_iou": best_fit,
        "best_fit_step": best_fit_step,
        "best_check_iou": best_check,
        "best_check_step": best_check_step,
        "history": history,
        "per_update_count": len(update_trace),
        "peak_alloc_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
        "not_generalisation": True,
    }
    write_json(run_dir / "train_summary.json", summary)
    log(f"done fit_iou={final_fit['pooled']['iou']:.4f} check_iou={final_check['pooled']['iou']:.4f} seconds={summary['seconds']:.1f}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


# ---------------------------------------------------------------------------
# summary / figures / report
# ---------------------------------------------------------------------------
def build_summary(out_dir: Path = DIAG_ROOT) -> dict:
    summary: dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "not_generalisation": True}
    selection = read_json(out_dir / "selection/diagnostic_tiles.json")
    summary["selection"] = {
        "n_tiles": selection["n_tiles"],
        "n_fit": selection["n_fit"],
        "n_check": selection["n_check"],
        "file_sha256": hash_file(out_dir / "selection/diagnostic_tiles.json"),
        "content_sha256": selection["content_sha256"],
        "by_event": dict(sorted(Counter(t["event_id"] for t in selection["tiles"]).items())),
        "by_size_class": dict(sorted(Counter(t["size_class"] for t in selection["tiles"]).items())),
    }
    path = out_dir / "interface/checkpoint_identity.json"
    if path.is_file():
        summary["checkpoint_identity"] = read_json(path)
    for arm in ("direct", "use"):
        for stem in ("layer_stats", "gradient_stats", "erf", "permutation", "missing_bands", "valid_region"):
            arm_path = out_dir / "interface" / f"{stem}_{arm}.json"
            if arm_path.is_file():
                summary[f"{stem}_{arm}"] = read_json(arm_path)
    for arm in ("direct", "use"):
        path = out_dir / "probes" / f"{arm}_{PROBE_UPDATES}" / "train_summary.json"
        if path.is_file():
            probe = read_json(path)
            summary[f"probe_{arm}"] = {
                key: probe.get(key)
                for key in (
                    "experiment", "n_fit", "n_check", "params", "seconds", "init_state_sha256",
                    "best_fit_iou", "best_fit_step", "best_check_iou", "best_check_step",
                    "final_fit", "final_check", "peak_alloc_mib", "config_sha256", "checkpoint_sha256",
                )
            }
            summary[f"probe_{arm}"]["history"] = [
                {"global_step": h["global_step"], "train_loss": h["train_loss"],
                 "fit_iou": h["fit"]["pooled"]["iou"], "check_iou": h["check"]["pooled"]["iou"]}
                for h in probe["history"]
            ]
    return summary


def write_arm_jsons(out_dir: Path = DIAG_ROOT) -> None:
    """Compatibility no-op: per-arm JSONs are written directly by the analyses."""
    return None


def run_probes(out_dir: Path = DIAG_ROOT, arms: str = "both", force: bool = False, device_name: str | None = None) -> dict:
    results = {}
    if arms in ("both", "direct"):
        results["direct"] = probe_arm("direct", out_dir, force=force, device_name=device_name)
    if arms in ("both", "use"):
        results["use"] = probe_arm("use", out_dir, force=force, device_name=device_name)
    summary = build_summary(out_dir)
    write_json(out_dir / "summary.json", summary)
    return summary


def make_figures(out_dir: Path = DIAG_ROOT, doc_dir: Path = DOC_ROOT) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    doc_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    selection = read_json(out_dir / "selection/diagnostic_tiles.json")
    tiles = selection["tiles"]
    files: list[Path] = []

    # 1. sampling design
    events = sorted({t["event_id"] for t in tiles})
    sizes = [name for _, _, name in SIZE_BINS]
    matrix = np.zeros((len(events), len(sizes)))
    for t in tiles:
        matrix[events.index(t["event_id"]), sizes.index(t["size_class"])] += 1
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1.4, 1]})
    bottom = np.zeros(len(events))
    for j, size in enumerate(sizes):
        axes[0].bar(events, matrix[:, j], bottom=bottom, label=SIZE_LABELS[size])
        bottom += matrix[:, j]
    axes[0].set_ylabel("图块数", fontproperties=font)
    axes[0].set_title("128块TRAIN诊断子集：按事件与最大对象尺寸", fontproperties=font)
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].legend(prop=font, fontsize=8)
    axes[1].hist([t["max_object_px"] for t in tiles], bins=24, color="#4e7fa8")
    axes[1].set_xlabel("最大连通对象面积（像素）", fontproperties=font)
    axes[1].set_ylabel("图块数", fontproperties=font)
    axes[1].set_title("对象尺寸抽样分布", fontproperties=font)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    path = doc_dir / "sampling.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    files.append(path)

    # 2. activations
    def load_arm(stem: str) -> dict:
        return read_json(out_dir / "interface" / f"{stem}_direct.json"), read_json(out_dir / "interface" / f"{stem}_use.json")

    layer_direct, layer_use = load_arm("layer_stats")
    layers = ["input", "bridge", "mit.stage1", "mit.stage2", "detail.stem", "use.features", "decoder"]
    metrics = [("effective_rank_ratio", "有效秩/通道数"), ("lag1_horizontal", "水平滞后1相关"), ("roughness", "空间粗糙度")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (key, label) in zip(axes, metrics):
        values_d = [layer_direct["layers"].get(l, {}).get(key) for l in layers]
        values_u = [layer_use["layers"].get(l, {}).get(key) for l in layers]
        x = np.arange(len(layers))
        ax.bar(x - 0.2, [np.nan if v is None else v for v in values_d], 0.4, label="直入参照", color=COLORS["direct"])
        ax.bar(x + 0.2, [np.nan if v is None else v for v in values_u], 0.4, label="USE", color=COLORS["use"])
        ax.set_xticks(x, layers, rotation=35, fontsize=8)
        ax.set_title(label, fontproperties=font)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(prop=font, fontsize=8)
    fig.suptitle("接口各层激活：有效秩、空间相关与粗糙度", fontproperties=font)
    fig.tight_layout()
    path = doc_dir / "activations.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    files.append(path)

    # 3. gradients and ERF
    grad_direct, grad_use = load_arm("gradient_stats")
    erf_direct, erf_use = load_arm("erf")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    groups = ["use.tokenizer", "use.metadata", "use.attention", "use.pool", "use.fuse", "bridge",
              "mit.embed1", "mit.stage1", "mit.stage2", "mit.stage3_4", "detail.stem", "decoder", "other"]
    for arm, data in (("direct", grad_direct), ("use", grad_use)):
        values = [data["parameter_groups"].get(g, {}).get("grad_share") or 0.0 for g in groups]
        axes[0].plot(groups, values, marker="o", label="直入参照" if arm == "direct" else "USE", color=COLORS[arm])
    axes[0].set_xticks(range(len(groups)), groups, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("参数梯度份额", fontproperties=font)
    axes[0].set_title("接口参数梯度分配（32块，一次损失反传）", fontproperties=font)
    axes[0].grid(alpha=0.2)
    axes[0].legend(prop=font, fontsize=8)
    def erf_mean(arm: str, key: str = "r90", interface: str | None = None) -> float | None:
        data = erf_direct if arm == "direct" else erf_use
        values = []
        for tile in data.get("tiles", []):
            source = tile["input"] if interface is None else tile.get("interface", {}).get(interface, {})
            if source.get(key) is not None:
                values.append(source[key])
        return float(np.mean(values)) if values else None

    erf_rows = [
        ("直入 输入", [erf_mean("direct")], COLORS["direct"]),
        ("USE 输入", [erf_mean("use")], COLORS["use"]),
        ("USE 桥接特征", [erf_mean("use", interface="bridge")], "#7a7a7a"),
        ("USE 隐含特征", [erf_mean("use", interface="use.features")], "#b58a3a"),
    ]
    labels = [row[0] for row in erf_rows if row[1][0] is not None]
    values = [row[1][0] for row in erf_rows if row[1][0] is not None]
    colors = [row[2] for row in erf_rows if row[1][0] is not None]
    axes[1].bar(range(len(values)), values, color=colors)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, fontproperties=font, fontsize=8)
    axes[1].set_ylabel("中心输出的梯度能量r90（像素）", fontproperties=font)
    axes[1].set_title("中心输出对各接口的感受野半径", fontproperties=font)
    axes[1].grid(axis="y", alpha=0.2)
    for i, value in enumerate(values):
        axes[1].text(i, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = doc_dir / "gradients_erf.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    files.append(path)

    # 4. permutation / missing / valid
    perm_direct, perm_use = load_arm("permutation")
    miss_direct, miss_use = load_arm("missing_bands")
    valid_direct, valid_use = load_arm("valid_region")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    modes = [("joint", "联合置换"), ("band_only", "仅波段"), ("metadata_only", "仅元数据")]
    x = np.arange(len(modes))
    for arm, data in (("direct", perm_direct), ("use", perm_use)):
        axes[0].bar(x + (-0.2 if arm == "direct" else 0.2), [data[m]["mean_abs_diff"] for m, _ in modes], 0.4,
                    label="直入参照" if arm == "direct" else "USE", color=COLORS[arm])
    axes[0].set_xticks(x, [label for _, label in modes], fontsize=9)
    axes[0].set_ylabel("平均绝对概率变化", fontproperties=font)
    axes[0].set_title("波段/元数据置换敏感性", fontproperties=font)
    axes[0].legend(prop=font, fontsize=8)
    subsets = ["rgb", "rgb_nir", "six", "unseen5"]
    axes[1].bar(np.arange(len(subsets)), [miss_use["subsets"][s]["iou_agreement"] for s in subsets], color=COLORS["use"])
    axes[1].set_xticks(range(len(subsets)), subsets)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("与全十波段预测的IoU一致度", fontproperties=font)
    axes[1].set_title("USE缺波段：与全输入预测一致度", fontproperties=font)
    axes[2].axis("off")
    axes[2].set_title("有效区值处理（同一无效掩膜，值0→NaN）", fontproperties=font)
    axes[2].text(
        0.02, 0.80,
        "直入参照\n"
        f"  NaN后输出有限：{valid_direct.get('nan_finite_all')}\n"
        "  没有 input_valid 接口\n"
        "  无效像元值直接进入前向",
        fontproperties=font, fontsize=9, va="top",
    )
    axes[2].text(
        0.02, 0.42,
        "USE\n"
        f"  NaN后输出有限：{valid_use.get('nan_finite_all')}\n"
        f"  远处有效像素最大变化：{valid_use.get('nan_max_diff_far_max')}\n"
        f"  input_valid 与掩膜一致：{valid_use.get('input_valid_matches')}\n"
        f"  全无效样本：{valid_use.get('fully_invalid')}",
        fontproperties=font, fontsize=9, va="top",
    )
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = doc_dir / "permutation_valid.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    files.append(path)

    # 5. probes
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for arm, color, label in (("direct", COLORS["direct"], "直入参照"), ("use", COLORS["use"], "USE")):
        path_json = out_dir / "probes" / f"{arm}_{PROBE_UPDATES}" / "train_summary.json"
        if not path_json.is_file():
            continue
        probe = read_json(path_json)
        steps = [h["global_step"] for h in probe["history"]]
        axes[0].plot(steps, [h["fit"]["pooled"]["iou"] for h in probe["history"]], marker="o", color=color, label=f"{label} 拟合96块")
        axes[0].plot(steps, [h["check"]["pooled"]["iou"] for h in probe["history"]], ls="--", marker=".", color=color, label=f"{label} TRAIN内32块")
        axes[1].plot([r["global_step"] for r in map(json.loads, (out_dir / "probes" / f"{arm}_{PROBE_UPDATES}" / "per_update.jsonl").read_text(encoding="utf-8").splitlines())],
                     [r["train_loss"] for r in map(json.loads, (out_dir / "probes" / f"{arm}_{PROBE_UPDATES}" / "per_update.jsonl").read_text(encoding="utf-8").splitlines())],
                     color=color, alpha=0.35, lw=0.7)
    axes[0].set_xlabel("成功更新", fontproperties=font)
    axes[0].set_ylabel("IoU（拟合探针，非泛化）", fontproperties=font)
    axes[0].set_title("300步拟合探针：同固定TRAIN子集", fontproperties=font)
    axes[0].grid(alpha=0.2)
    axes[0].legend(prop=font, fontsize=7)
    axes[1].set_xlabel("成功更新", fontproperties=font)
    axes[1].set_ylabel("训练损失", fontproperties=font)
    axes[1].set_title("逐更新损失", fontproperties=font)
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    path = doc_dir / "probes.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    files.append(path)
    return files


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "非有限"
        if value != 0.0 and abs(value) < 1e-3:
            return f"{value:.2e}"
        return f"{value:.{digits}f}"
    return str(value)


def write_report(out_dir: Path = DIAG_ROOT, doc_dir: Path = DOC_ROOT) -> Path:
    summary = build_summary(out_dir)
    write_json(out_dir / "summary.json", summary)
    selection = summary["selection"]
    queue_lines: list[str] = []
    queue_path = out_dir / "queue_check.json"
    if queue_path.is_file():
        queue = read_json(queue_path)
        acq = queue.get("acquisition", {})
        comp = queue.get("completion", {})
        queue_lines = [
            "## 0. 既有USE队列核验（只读）",
            "",
            f"- 核验时间 `{queue.get('checked_at')}`；只读，未启动/停止/修改任何外部数据或完成队列进程、配置或权重。",
            f"- 外部扩量 `outputs/use/external/acquisition_status.json`：status `{acq.get('status')}`，"
            f"S2 {acq.get('progress', {}).get('s2')}、Landsat {acq.get('progress', {}).get('landsat')}，"
            f"errors {acq.get('errors_count')}，updated `{acq.get('updated_at')}`。",
            f"- completion `outputs/use/completion_status.json`：status `{comp.get('status')}`，jobs {comp.get('jobs')}。",
            f"- 实测存活进程 {len(queue.get('live_processes', []))} 个（`prepare_use_external.py`/`run_use.py`）；"
            "PID、启动时间与完整命令见 `outputs/use_interface_diagnostics/queue_check.json`。",
            "",
        ]
    lines = [
        "# USE接口首批诊断：固定128块TRAIN子集",
        "",
        "> 本页是**诊断证据**，不是正式训练或泛化成绩。300步探针只回答“同一固定TRAIN子集上能否拟合”，",
        "> 不能作为跨事件性能、显著性结论或模型选择依据。",
        "",
        "## 结论摘要",
        "",
        "- 同一固定TRAIN子集、同一300步短日程：直入在96块拟合集上IoU约0.386、TRAIN内32块约0.234；USE约0.202、约0.046。两组都是拟合探针，不是泛化成绩。",
        "- USE桥接输出与十波段输入的对角相关均值-0.025，10个通道中4个标准差不足输入一半；联合置换输出不变（约2e-6，数值零），仅元数据置换改变0.0582，说明物理描述确实进入前向。",
        "- USE细节分支的有效秩为2.05/32（直入7.08/32），参数梯度份额0.0012（直入0.0127）；USE编码器约9.9%、桥接约1.0%，直入首层67.8%。这些是接口层的实际差异，不等同于架构上限。",
        "- 缺波段：USE对“移除通道”与“band_valid=False+NaN”给出一致预测（概率差3.7e-6）；无效区NaN对USE远处有效像素最大变化0.0000，直入输出非有限。",
        "- 因此下一步优先做桥接波段/尺度标定与细节分支辅助监督；在完成标定对照前，不把本次300步差异当作USE不可用的证据。",
        "",
        *queue_lines,
        "## 1. 数据与范围",
        "",
        "- 协议：`event_ood_v4r1`；只取 `split=TRAIN` 的2223块，按事件×最大连通对象尺寸分层抽128块。",
        f"- 事件分布：`{selection['by_event']}`；尺寸分布：`{selection['by_size_class']}`。",
        f"- 探针划分：拟合96块、TRAIN内检查32块（两者都来自TRAIN，后者不是DEV/泛化集）。",
        f"- 清单：`outputs/use_interface_diagnostics/selection/diagnostic_tiles.json`，文件SHA256 `{selection['file_sha256']}`，",
        f"  内容SHA256 `{selection['content_sha256']}`。",
        "",
        "## 2. 权重、配置与复现",
        "",
        "| 用途 | 配置 | 权重 | SHA256 | 权重步/最佳步 | 正式运行止步 |",
        "|---|---|---|---|---|---|",
    ]
    identity = summary.get("checkpoint_identity", {})
    for arm, label in (("direct", "直入参照（十波段直入＋原网格细节）"), ("use", "USE（共享编码＋物理描述＋桥接）")):
        row = identity.get(arm, {})
        lines.append(
            f"| {label} | `{row.get('config', '')}` | `{row.get('checkpoint', '')}` | `{row.get('checkpoint_sha256', '')}` | "
            f"{row.get('checkpoint_step', '')}/{row.get('checkpoint_best_step', '')} | {row.get('run_stop_step', '')} |"
        )
    lines += [
        "",
        f"接口诊断使用DEV选模得到的best权重（直入@{identity.get('direct', {}).get('checkpoint_step')}、"
        f"USE@{identity.get('use', {}).get('checkpoint_step')}；两轮正式运行分别止步"
        f"{identity.get('direct', {}).get('run_stop_step')}和{identity.get('use', {}).get('run_stop_step')}），",
        "因此这些激活/梯度反映“被选中的开发权重”，不是训练末态，也不是无DEV接触的权重；不用于性能比较。",
        "",
    ]
    lines += [
        "",
        "复现命令（项目根目录，Python 环境见研究入口）：",
        "",
        "```powershell",
        "python -X utf8 scripts/diagnose_use_interface.py all",
        "# 或分步：",
        "python -X utf8 scripts/diagnose_use_interface.py select",
        "python -X utf8 scripts/diagnose_use_interface.py represent",
        "python -X utf8 scripts/diagnose_use_interface.py probe --arm both",
        "python -X utf8 scripts/diagnose_use_interface.py figures",
        "python -X utf8 scripts/diagnose_use_interface.py report",
        "```",
        "",
        "复现性实测：同一命令重跑，`diagnostic_tiles.json`字节级一致（SHA256 `c4749296…`）；探针初始化state_dict SHA256一致；",
        "300步直入拟合IoU在0.3856与0.3858之间浮动，说明GPU算子不是逐位确定，但差异约2e-4。表格以最新一次产物为准，不把末位差异当作方法差异。",
        "",
        "## 3. 接口激活与空间信息",
        "",
        "| 层 | 直入：有效秩/通道 | USE：有效秩/通道 | 直入：水平滞后1相关 | USE：水平滞后1相关 | 直入：粗糙度 | USE：粗糙度 |",
        "|---|---|---|---|---|---|---|",
    ]
    layers = ["input", "bridge", "mit.stage1", "mit.stage2", "detail.stem", "use.features", "decoder"]
    layer_direct, layer_use = summary.get("layer_stats_direct", {}), summary.get("layer_stats_use", {})
    for layer in layers:
        d = layer_direct.get("layers", {}).get(layer, {})
        u = layer_use.get("layers", {}).get(layer, {})
        lines.append(
            f"| {layer} | {_fmt(d.get('effective_rank_ratio'))} | {_fmt(u.get('effective_rank_ratio'))} | "
            f"{_fmt(d.get('lag1_horizontal'))} | {_fmt(u.get('lag1_horizontal'))} | {_fmt(d.get('roughness'))} | {_fmt(u.get('roughness'))} |"
        )
    bridge = layer_use.get("bridge_weight", {})
    scale = layer_use.get("bridge_scale", {})
    pair = layer_use.get("input_bridge_pair", {})
    lines += [
        "",
        "桥接几何：",
        "",
        f"- `Conv2d(32→10,1)` 奇异值条件数 {_fmt(bridge.get('condition_number'), 2)}；奇异值有效秩 {_fmt(bridge.get('effective_rank'), 2)}。",
        f"- 桥接输出/输入逐通道标准差比：均值绝对 log2 = {_fmt(scale.get('mean_abs_log2_ratio'), 3)}；"
        f"低于输入一半的通道 {scale.get('channels_below_half_input_std')}/10，高于两倍的通道 {scale.get('channels_above_double_input_std')}/10。",
        f"- 桥接输出与十波段输入的逐通道相关（对角线均值）{_fmt(pair.get('diag_mean'), 3)}；"
        f"非对角绝对最大 {_fmt(pair.get('offdiag_abs_max'), 3)}。",
        "",
        "## 4. 梯度与感受野",
        "",
        "| 组 | 直入：梯度份额 | USE：梯度份额 | 直入：相对更新 | USE：相对更新 |",
        "|---|---|---|---|---|",
    ]
    grad_direct, grad_use = summary.get("gradient_stats_direct", {}), summary.get("gradient_stats_use", {})
    for group in ["use.tokenizer", "use.metadata", "use.attention", "use.pool", "use.fuse", "bridge",
                  "mit.embed1", "mit.stage1", "mit.stage2", "mit.stage3_4", "detail.stem", "decoder", "other"]:
        d = grad_direct.get("parameter_groups", {}).get(group, {})
        u = grad_use.get("parameter_groups", {}).get(group, {})
        lines.append(
            f"| {group} | {_fmt(d.get('grad_share'))} | {_fmt(u.get('grad_share'))} | "
            f"{_fmt(d.get('relative_update'), 5)} | {_fmt(u.get('relative_update'), 5)} |"
        )
    erf_d = summary.get("erf_direct", {})
    erf_u = summary.get("erf_use", {})
    lines += [
        "",
        "中心像素感受野（输入梯度能量50%/90%半径，单位像素）：",
        "",
        "| 模型 | 输入 r50 | 输入 r90 | 桥接特征 r90 |",
        "|---|---|---|---|",
    ]
    for arm, data in (("direct", erf_d), ("use", erf_u)):
        rows = [t["input"] for t in data.get("tiles", []) if t["input"].get("r90") is not None]
        bridge_rows = [t.get("interface", {}).get("bridge", {}) for t in data.get("tiles", []) if arm == "use"]
        bridge_r90 = [b.get("r90") for b in bridge_rows if b.get("r90") is not None]
        lines.append(
            f"| {'直入' if arm == 'direct' else 'USE'} | {_fmt(np.mean([r['r50'] for r in rows]) if rows else None, 1)} | "
            f"{_fmt(np.mean([r['r90'] for r in rows]) if rows else None, 1)} | "
            f"{_fmt(np.mean(bridge_r90) if bridge_r90 else None, 1)} |"
        )
    lines += [
        "",
        "## 5. 波段/元数据置换、缺波段与有效区",
        "",
        "| 模型 | 联合置换平均概率差 | 仅波段 | 仅元数据 | 缺波段支持 | NaN无效区最大概率变化（远处有效像素） |",
        "|---|---|---|---|---|---|",
    ]
    for arm in ("direct", "use"):
        perm = summary.get(f"permutation_{arm}", {})
        miss = summary.get(f"missing_bands_{arm}", {})
        valid = summary.get(f"valid_region_{arm}", {})
        lines.append(
            f"| {'直入' if arm == 'direct' else 'USE'} | {_fmt(perm.get('joint', {}).get('mean_abs_diff'))} | "
            f"{_fmt(perm.get('band_only', {}).get('mean_abs_diff'))} | {_fmt(perm.get('metadata_only', {}).get('mean_abs_diff'))} | "
            f"{'是' if miss.get('supported') else '否'} | {_fmt(valid.get('nan_max_diff_far_max'))} |"
        )
    use_missing = summary.get("missing_bands_use", {})
    lines += [
        "",
        "USE缺波段与全十波段预测的一致度：",
        "",
        "| 子集 | 波段数 | 平均概率差 | IoU一致度 |",
        "|---|---|---|---|",
    ]
    for name in ("rgb", "rgb_nir", "six", "unseen5"):
        row = use_missing.get("subsets", {}).get(name, {})
        lines.append(f"| {name} | {len(SUBSETS[name])} | {_fmt(row.get('mean_abs_diff'))} | {_fmt(row.get('iou_agreement'))} |")
    lines += [
        "",
        "## 6. 300步拟合探针（同一固定TRAIN子集）",
        "",
        "| 模型 | 拟合96块最终IoU | TRAIN内32块最终IoU | 最佳拟合步 | 参数量 | 用时(s) | 初始化SHA256 |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in ("direct", "use"):
        probe = summary.get(f"probe_{arm}", {})
        lines.append(
            f"| {'直入' if arm == 'direct' else 'USE'} | {_fmt(probe.get('final_fit', {}).get('pooled', {}).get('iou'))} | "
            f"{_fmt(probe.get('final_check', {}).get('pooled', {}).get('iou'))} | {probe.get('best_fit_step', '')} | "
            f"{probe.get('params', {}).get('total', '')} | {_fmt(probe.get('seconds'), 1)} | `{str(probe.get('init_state_sha256', ''))[:16]}…` |"
        )
    interpretation_path = doc_dir / "interpretation.md"
    interpretation = (
        interpretation_path.read_text(encoding="utf-8").strip()
        if interpretation_path.is_file()
        else "（数值判读尚未写入；保存为同目录 `interpretation.md` 后重跑 `report` 即可并入。）"
    )
    lines += [
        "",
        "探针只用于检查接口在固定小样本上的可拟合性；两组都从头初始化，使用正式配置的优化器/损失与300步短日程，",
        "不是2000步正式训练，不能据此判断事件泛化或显著性。实际探针设置见",
        "`probes/direct_300/resolved_config.json`、`probes/use_300/resolved_config.json`与各自`run_meta.json`的`probe`块；",
        "正式配置里的`stop_updates=2000`/`schedule_updates=3600`没有被用于探针。",
        "",
        "## 7. 限制",
        "",
        "- 单seed、128块固定TRAIN子集，约375万参数模型中仅300步探针；结论只覆盖“该子集上的接口行为”，不证明其他子集或其他seed。",
        "- 代表层用DEV选模得到的best权重（直入1600步、USE1400步；两轮正式运行均止步2000）做一次反传/统计；梯度不是训练全程轨迹，也不是训练末态。",
        "- 置换检验使用3个固定随机排列；缺波段只覆盖4个预定义子集（含未见5波段组合），不覆盖任意缺波段组合。",
        "- 有效区检验用合成矩形与既有质量侧表，未修改源数组或标签；不证明真实云/阴影边界的行为。",
        "- 本页不新增DEV评价、不使用FINAL/Italy，也不启动或修改任何2000步正式队列；best权重本身来自既有DEV选模，这一点已在第2节标明。",
        "- 300步拟合差距可能来自优化难度、随机初始化、桥接标定或数据顺序；未做多种子、学习率、初始化或架构消融，不能据此做因果归因。",
        "",
        "## 8. 判读与方案",
        "",
        interpretation,
        "",
        "桥接标定与特征辅助监督的具体实现方案见同目录 `PLAN.md`。",
        "",
        "## 9. 诊断图",
        "",
        "![抽样设计](sampling.png)",
        "",
        "左图每根柱子是一个TRAIN事件、按最大连通对象尺寸分段堆叠；右图是128块的最大对象面积分布。看三点：9个事件是否都有覆盖、尺寸档是否按TRAIN比例保留（1–4 px在TRAIN仅25块，本次7块已属过采样）、有没有事件/尺寸完全缺失。",
        "",
        "![接口激活](activations.png)",
        "",
        "三栏分别是各层“有效秩/通道数”“水平滞后1相关”“空间粗糙度”；蓝=直入、橙=USE，横轴从输入到解码器。看桥接（bridge）与USE特征（use.features）是否比直入同层更低秩、更平滑，以及细节分支（detail.stem）是否明显掉秩。直入没有bridge/use.features柱是接口本身不存在。",
        "",
        "![梯度与感受野](gradients_erf.png)",
        "",
        "左图是各参数组的梯度份额（32块一次反传），看USE的编码器/桥接/细节分支是否远低于直入首层；右图是中心输出对输入、桥接特征、USE特征的梯度能量r90（像素），看两条路径的空间上下文是否相近。r90只说明“中心输出能受多远输入影响”，不是有效感受野的完整测量。",
        "",
        "![置换、缺波段与有效区](permutation_valid.png)",
        "",
        "左图：联合/仅波段/仅元数据置换的平均绝对概率变化，USE联合置换应接近0、仅元数据应非0；中图：USE缺波段预测与全十波段预测的IoU一致度，越低表示缺波段影响越大；右图文字：同一无效掩膜下把值0改成NaN的行为，USE应远处不变、直入应出现非有限。",
        "",
        "![300步拟合探针](probes.png)",
        "",
        "左图实线是96块拟合IoU、虚线是TRAIN内32块检查IoU，横轴是成功更新；右图是逐更新训练损失。两组都只说明“在固定小样本上拟合得多快”，不是DEV/泛化成绩；曲线末端仍在上升说明300步不是收敛点。",
        "",
        "## 10. 证据索引",
        "",
        "- `outputs/use_interface_diagnostics/selection/`：清单、对象统计、抽样汇总。",
        "- `outputs/use_interface_diagnostics/interface/`：各层激活、桥接几何、梯度、ERF、置换、缺波段、有效区。",
        "- `outputs/use_interface_diagnostics/probes/`：两组300步探针的逐更新日志、指标、初始化/末态权重与元数据。",
        "- `outputs/use_interface_diagnostics/summary.json`、`provenance.json`：汇总与哈希。",
    ]
    path = doc_dir / "REPORT.md"
    doc_dir.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_provenance(out_dir: Path = DIAG_ROOT) -> dict:
    sources = [DIRECT_CONFIG, USE_CONFIG, DIRECT_CKPT, USE_CKPT, MANIFEST, STATS, QUALITY_INDEX, Path(__file__)]
    files = []
    for path in sources:
        if path.is_file():
            files.append({"path": str(path).replace("\\", "/"), "sha256": hash_file(path), "size_bytes": path.stat().st_size})
    outputs = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name not in {"provenance.json"}:
            outputs.append({"path": str(path).replace("\\", "/"), "sha256": hash_file(path), "size_bytes": path.stat().st_size})
    documents = []
    for path in sorted(DOC_ROOT.rglob("*")):
        if path.is_file():
            documents.append({"path": str(path).replace("\\", "/"), "sha256": hash_file(path), "size_bytes": path.stat().st_size})
    payload = {
        "command": "python -X utf8 scripts/diagnose_use_interface.py all",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sources": files,
        "outputs": outputs,
        "documents": documents,
        "note": "outputs include checkpoints; hashes identify the exact diagnostic artefacts",
    }
    write_json(out_dir / "provenance.json", payload)
    return payload


def run_all(out_dir: Path = DIAG_ROOT, device_name: str | None = None, force: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    build_selection(out_dir)
    run_represent(out_dir, device_name)
    run_probes(out_dir, "both", force=force, device_name=device_name)
    make_figures(out_dir, DOC_ROOT)
    write_report(out_dir, DOC_ROOT)
    return write_provenance(out_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select", help="build the fixed 128-tile TRAIN manifest")
    select.add_argument("--out", type=Path, default=DIAG_ROOT)
    select.add_argument("--tiles", type=int, default=N_TILES)
    represent = sub.add_parser("represent", help="interface activation/gradient/permutation diagnostics")
    represent.add_argument("--out", type=Path, default=DIAG_ROOT)
    represent.add_argument("--device", default=None)
    probe = sub.add_parser("probe", help="300-update fitting probes")
    probe.add_argument("--arm", choices=["direct", "use", "both"], default="both")
    probe.add_argument("--out", type=Path, default=DIAG_ROOT)
    probe.add_argument("--device", default=None)
    probe.add_argument("--force", action="store_true")
    figures = sub.add_parser("figures", help="render diagnostic figures")
    figures.add_argument("--out", type=Path, default=DIAG_ROOT)
    report = sub.add_parser("report", help="write REPORT.md from the collected JSON")
    report.add_argument("--out", type=Path, default=DIAG_ROOT)
    all_cmd = sub.add_parser("all", help="select -> represent -> probes -> figures -> report")
    all_cmd.add_argument("--out", type=Path, default=DIAG_ROOT)
    all_cmd.add_argument("--device", default=None)
    all_cmd.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "select":
        payload = build_selection(args.out, args.tiles)
        print(json.dumps({"n_tiles": payload["n_tiles"], "content_sha256": payload["content_sha256"]}, ensure_ascii=False))
    elif args.command == "represent":
        summary = run_represent(args.out, args.device)
        write_arm_jsons(args.out)
        write_json(args.out / "summary.json", summary)
        print(json.dumps({"summary": str(args.out / "summary.json")}, ensure_ascii=False))
    elif args.command == "probe":
        write_arm_jsons(args.out)
        run_probes(args.out, args.arm, args.force, args.device)
        write_arm_jsons(args.out)
        print(json.dumps({"probes": str(args.out / "probes")}, ensure_ascii=False))
    elif args.command == "figures":
        write_arm_jsons(args.out)
        files = make_figures(args.out, DOC_ROOT)
        print(json.dumps([str(p) for p in files], ensure_ascii=False))
    elif args.command == "report":
        write_arm_jsons(args.out)
        path = write_report(args.out, DOC_ROOT)
        write_provenance(args.out)
        print(str(path))
    elif args.command == "all":
        run_all(args.out, args.device, args.force)
        print(str(DOC_ROOT / "REPORT.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
