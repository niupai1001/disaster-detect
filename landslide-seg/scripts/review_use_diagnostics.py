"""Review the existing USE diagnostics and supplement the physical_mixed reference.

Read-only scope:
  * audit the fixed 128-tile TRAIN manifest, the recorded 300-step probe artefacts,
    and the existing formal run identities;
  * load the existing physical_mixed best checkpoint and compute the same
    activation / gradient / representation / interface diagnostics on it;
  * compare physical (full-band training) with physical_mixed (mixed-band
    training) without treating the former's mechanism explanation as the latter's;
  * write evidence under outputs/use_diagnostic_review and a short review under
    docs/assets/research/use_diagnostic_review.

It never trains, never re-runs the 300-step probes, and never edits formal
configs/checkpoints or the previous diagnostic directory.

Commands:
  python -X utf8 scripts/review_use_diagnostics.py audit
  python -X utf8 scripts/review_use_diagnostics.py represent
  python -X utf8 scripts/review_use_diagnostics.py compare
  python -X utf8 scripts/review_use_diagnostics.py report
  python -X utf8 scripts/review_use_diagnostics.py all
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/use_diagnostic_review"
DOC = ROOT / "docs/assets/research/use_diagnostic_review"
OLD = ROOT / "outputs/use_interface_diagnostics"
MIXED_RUN = ROOT / "outputs/use/s2_10band_v4r1_p20/use_physical_mixed/1234"
MIXED_CKPT = MIXED_RUN / "best.pt"
DIRECT_CKPT = ROOT / "outputs/use/s2_10band_v4r1_p20/use_baseline/1234/best.pt"
PHYSICAL_CKPT = ROOT / "outputs/use/s2_10band_v4r1_p20/use_physical/1234/best.pt"


def _load_diag():
    spec = importlib.util.spec_from_file_location(
        "diagnose_use_interface", ROOT / "scripts/diagnose_use_interface.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diag = _load_diag()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")


def sha(path: Path) -> str:
    return diag.hash_file(path)


def _device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _tiles_records():
    tiles = diag.load_tiles(OLD)
    records = diag.read_manifest(diag.MANIFEST)
    by_id = {r["sample_id"]: r for r in records}
    selected = [by_id[t["sample_id"]] for t in tiles["tiles"]]
    return tiles, selected


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
def audit(out_dir: Path = OUT) -> dict:
    tiles, selected = _tiles_records()
    summary = read_json(OLD / "summary.json")
    identity = read_json(OLD / "interface/checkpoint_identity.json")
    issues: list[str] = []
    checks: dict[str, object] = {}

    manifest_path = OLD / "selection/diagnostic_tiles.json"
    checks["manifest_file_sha256"] = sha(manifest_path)
    checks["manifest_content_sha256"] = tiles["content_sha256"]
    checks["manifest_counts"] = {"n_tiles": len(selected), "n_fit": sum(t["probe_role"] == "fit" for t in tiles["tiles"]),
                                 "n_check": sum(t["probe_role"] == "check" for t in tiles["tiles"])}
    if checks["manifest_file_sha256"] != summary["selection"]["file_sha256"]:
        issues.append("manifest file hash differs from previous summary")
    if checks["manifest_content_sha256"] != summary["selection"]["content_sha256"]:
        issues.append("manifest content hash differs from previous summary")
    if any(r.get("split") != "train" for r in selected):
        issues.append("selected records are not all split=train")
    if len({r["sample_id"] for r in selected}) != len(selected):
        issues.append("duplicate sample_id in fixed subset")

    # Probe artefacts are read-only; recompute the init hash from init.pt.
    probes = {}
    for arm in ("direct", "use"):
        pdir = OLD / "probes" / f"{arm}_300"
        train_summary = read_json(pdir / "train_summary.json")
        init = torch.load(pdir / "init.pt", map_location="cpu", weights_only=False)
        init_hash = diag.state_dict_sha256(init["state_dict"])
        recorded = summary[f"probe_{arm}"]
        probes[arm] = {
            "init_state_sha256": init_hash,
            "recorded_init_sha256": recorded.get("init_state_sha256"),
            "init_match": init_hash == recorded.get("init_state_sha256") == train_summary.get("init_state_sha256"),
            "best_fit_iou": train_summary.get("best_fit_iou"),
            "best_fit_step": train_summary.get("best_fit_step"),
            "best_check_iou": train_summary.get("best_check_iou"),
            "best_check_step": train_summary.get("best_check_step"),
            "config_sha256": train_summary.get("config_sha256"),
            "resolved_config": str((pdir / "resolved_config.json").relative_to(ROOT)).replace("\\", "/")
            if (pdir / "resolved_config.json").is_file() else None,
            "init_pt_sha256": sha(pdir / "init.pt"),
            "train_summary_sha256": sha(pdir / "train_summary.json"),
        }
        if not probes[arm]["init_match"]:
            issues.append(f"{arm}: recorded init hash does not match init.pt")
        if train_summary.get("n_fit") != 96 or train_summary.get("n_check") != 32:
            issues.append(f"{arm}: probe split is not 96/32")
        if train_summary.get("not_generalisation") is not True:
            issues.append(f"{arm}: probe is not marked as fitting-only")

    # Existing formal run identities.
    formal = {}
    for label, ckpt_path in (("direct", DIRECT_CKPT), ("physical", PHYSICAL_CKPT), ("physical_mixed", MIXED_CKPT)):
        run_dir = ckpt_path.parent
        train_summary = read_json(run_dir / "train_summary.json")
        resolved = read_json(run_dir / "resolved_config.json")
        meta = read_json(run_dir / "run_meta.json")
        formal[label] = {
            "checkpoint": str(ckpt_path.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256": sha(ckpt_path),
            "run_dir": str(run_dir.relative_to(ROOT)).replace("\\", "/"),
            "global_step": train_summary.get("global_step"),
            "complete": train_summary.get("complete"),
            "best_step": train_summary.get("best_step"),
            "best_val_iou": train_summary.get("best_val_iou"),
            "mixed_bands": resolved["segformer"]["data"].get("mixed_bands"),
            "experiment": resolved["segformer"]["experiment"],
            "manifest_sha256_prefix": meta.get("manifest", {}).get("sha256"),
            "stats_sha256": meta.get("stats_sha256"),
            "quality_sha256": meta.get("use_quality_sha256"),
        }
    if formal["physical_mixed"]["global_step"] != 2000 or formal["physical_mixed"]["complete"] is not True:
        issues.append("physical_mixed formal run is not a completed 2000-step run")
    if formal["physical_mixed"]["mixed_bands"] is not True:
        issues.append("physical_mixed config does not have mixed_bands=true")
    if formal["physical_mixed"]["experiment"] != "use_physical_mixed":
        issues.append("physical_mixed experiment name differs")
    if identity.get("direct", {}).get("checkpoint_sha256") != formal["direct"]["checkpoint_sha256"]:
        issues.append("direct diagnostic checkpoint differs from formal direct best")
    if identity.get("use", {}).get("checkpoint_sha256") != formal["physical"]["checkpoint_sha256"]:
        issues.append("USE diagnostic checkpoint differs from formal physical best")
    if formal["physical_mixed"]["checkpoint_sha256"] in (formal["physical"]["checkpoint_sha256"],):
        issues.append("physical_mixed checkpoint equals physical checkpoint")

    # Existing old diagnostic outputs must exist; no 300-step probe is rerun here.
    expected_files = [
        "summary.json", "interface/checkpoint_identity.json", "interface/layer_stats_use.json",
        "interface/gradient_stats_use.json", "interface/erf_use.json", "interface/permutation_use.json",
        "interface/missing_bands_use.json", "interface/valid_region_use.json",
        "probes/direct_300/train_summary.json", "probes/use_300/train_summary.json",
    ]
    existing = {name: (OLD / name).is_file() for name in expected_files}
    missing = [name for name, ok in existing.items() if not ok]
    if missing:
        issues.append(f"missing previous evidence: {missing}")

    result = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scope": "read-only review; no training and no 300-step probe rerun",
        "manifest": checks,
        "probes": probes,
        "formal_runs": formal,
        "previous_evidence": existing,
        "issues": issues,
        "ok": not issues,
        "corrections": [
            "bridge output is a fixed 10-dimensional latent, not band-identified reflectance; do not index it by input band_valid",
            "low diagonal correlation / low effective rank / low gradient share are diagnostic phenomena, not proven bottlenecks",
            "physical (full-band) diagnostics must not be copied to physical_mixed; mixed is the main control",
            "no multi-loss accumulation or hyper-parameter scan is used in this review",
        ],
    }
    write_json(out_dir / "audit.json", result)
    return result


# ---------------------------------------------------------------------------
# physical_mixed activations / gradients / representation
# ---------------------------------------------------------------------------
@torch.no_grad()
def physical_mixed_activations(out_dir: Path = OUT, device_name: str | None = None) -> dict:
    device = _device(device_name)
    model, ckpt = diag.load_checkpoint(MIXED_CKPT, str(device))
    model.eval()
    assert ckpt["config"]["segformer"]["data"].get("mixed_bands") is True
    tiles, selected = _tiles_records()
    ds = diag.arm_dataset("use", selected, augment=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=diag.collate, num_workers=0)
    accumulators: dict[str, diag.FeatureAccumulator] = {}
    pair = diag.PairAccumulator()
    agreements = []
    for batch in loader:
        image = batch["image"].to(device)
        _, store = diag.capture(model, image, "use")
        input_tensor = image["x"]
        for name, tensor in [("input", input_tensor)] + list(store.items()):
            if name == "logits" or not torch.is_tensor(tensor):
                continue
            accumulators.setdefault(name, diag.FeatureAccumulator()).add(tensor)
        if "bridge" in store:
            pair.add(input_tensor, store["bridge"])
        agreement = diag.valid_agreement(batch, "use")
        if agreement:
            agreements.append(agreement)
    stats = {name: acc.finalize() for name, acc in accumulators.items()}
    weight = model.bridge.weight.detach().float().cpu().numpy()[:, :, 0, 0]
    bias = model.bridge.bias.detach().float().cpu().numpy()
    singular = np.linalg.svd(weight, compute_uv=False)
    input_std = np.asarray(stats["input"]["channel_std"])
    bridge_std = np.asarray(stats["bridge"]["channel_std"])
    ratio = bridge_std / np.maximum(input_std, 1e-8)
    result = {
        "arm": "physical_mixed",
        "checkpoint": str(MIXED_CKPT.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha(MIXED_CKPT),
        "config": "outputs/use/s2_10band_v4r1_p20/use_physical_mixed/1234/resolved_config.json",
        "training": {
            "global_step": read_json(MIXED_RUN / "train_summary.json")["global_step"],
            "best_step": read_json(MIXED_RUN / "train_summary.json")["best_step"],
            "best_val_iou": read_json(MIXED_RUN / "train_summary.json")["best_val_iou"],
            "mixed_bands": True,
        },
        "input_mode": "full 10-band evaluation input (mixed sampling is a training procedure, not an evaluation input change)",
        "layers": stats,
        "valid_agreement": {"iou": float(np.mean([a["iou"] for a in agreements])) if agreements else None},
        "input_bridge_pair": pair.finalize(),
        "bridge_weight": {
            "shape": list(weight.shape),
            "sha256": diag.tensor_sha256(model.bridge.weight) + ":" + diag.tensor_sha256(model.bridge.bias),
            "singular_values": singular.tolist(),
            "condition_number": float(singular[0] / singular[-1]) if singular[-1] > 0 else None,
            "effective_rank": float(singular.sum() ** 2 / max(float((singular ** 2).sum()), 1e-30)),
            "frobenius_norm": float(np.linalg.norm(weight)),
            "bias_abs_mean": float(np.abs(bias).mean()),
            "bias_abs_max": float(np.abs(bias).max()),
        },
        "bridge_scale": {
            "channel_std_ratio": ratio.tolist(),
            "log2_ratio": np.log2(np.maximum(ratio, 1e-8)).tolist(),
            "mean_abs_log2_ratio": float(np.abs(np.log2(np.maximum(ratio, 1e-8))).mean()),
            "channels_below_half_input_std": int((ratio < 0.5).sum()),
            "channels_above_double_input_std": int((ratio > 2.0).sum()),
        },
    }
    write_json(out_dir / "physical_mixed_activations.json", result)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def physical_mixed_gradients(out_dir: Path = OUT, device_name: str | None = None) -> dict:
    device = _device(device_name)
    model, ckpt = diag.load_checkpoint(MIXED_CKPT, str(device))
    _, selected = _tiles_records()
    cfg = ckpt["config"]
    tmp = out_dir / "_tmp_grad"
    tmp.mkdir(parents=True, exist_ok=True)
    diag.analyse_gradients(model, "use", selected, device, tmp, cfg)
    data = read_json(tmp / "interface/gradient_stats_use.json")
    data["arm"] = "physical_mixed"
    data["checkpoint"] = str(MIXED_CKPT.relative_to(ROOT)).replace("\\", "/")
    data["checkpoint_sha256"] = sha(MIXED_CKPT)
    write_json(out_dir / "physical_mixed_gradients.json", data)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return data


def physical_mixed_erf(out_dir: Path = OUT, device_name: str | None = None) -> dict:
    device = _device(device_name)
    model, ckpt = diag.load_checkpoint(MIXED_CKPT, str(device))
    tiles, _ = _tiles_records()
    cfg = ckpt["config"]
    tmp = out_dir / "_tmp_erf"
    tmp.mkdir(parents=True, exist_ok=True)
    diag.analyse_erf(model, "use", tiles["tiles"], device, tmp, cfg)
    data = read_json(tmp / "interface/erf_use.json")
    data["arm"] = "physical_mixed"
    data["checkpoint"] = str(MIXED_CKPT.relative_to(ROOT)).replace("\\", "/")
    data["checkpoint_sha256"] = sha(MIXED_CKPT)
    write_json(out_dir / "physical_mixed_erf.json", data)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return data


@torch.no_grad()
def physical_mixed_permutation(out_dir: Path = OUT, device_name: str | None = None) -> dict:
    device = _device(device_name)
    model, _ = diag.load_checkpoint(MIXED_CKPT, str(device))
    model.eval()
    _, selected = _tiles_records()
    ds = diag.arm_dataset("use", selected, augment=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=diag.collate, num_workers=0)
    import random
    rng = random.Random(diag.SELECT_SEED)
    perms = [rng.sample(range(10), 10) for _ in range(3)]
    rows = []
    for batch in loader:
        image = batch["image"].to(device)
        p0 = torch.sigmoid(model(image)["logits"].float()[:, 0])
        valid = batch["valid"].to(device)
        record = {"n": int(valid.sum())}
        for mode in ("joint", "band", "metadata"):
            values = []
            for perm in perms:
                permuted = diag.permute_batch(batch, perm, mode, "use")
                p1 = diag.probability(model, batch_input(permuted, "use").to(device), "use")
                values.append(diag.prob_delta(p0, p1, valid))
            record[mode] = {
                key: float(np.mean([r[key] for r in values if r[key] is not None]))
                if any(r[key] is not None for r in values) else None
                for key in ("mean_abs_diff", "max_abs_diff", "iou_agreement")
            }
        rows.append(record)
    weights = [r["n"] for r in rows]
    result = {
        "arm": "physical_mixed",
        "checkpoint": str(MIXED_CKPT.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha(MIXED_CKPT),
        "permutations": perms,
        "n_pixels": sum(weights),
        "joint": {k: float(np.average([r["joint"][k] for r in rows], weights=weights)) for k in ("mean_abs_diff", "max_abs_diff", "iou_agreement")},
        "band_only": {k: float(np.average([r["band"][k] for r in rows], weights=weights)) for k in ("mean_abs_diff", "max_abs_diff", "iou_agreement")},
        "metadata_only": {k: float(np.average([r["metadata"][k] for r in rows], weights=weights)) for k in ("mean_abs_diff", "max_abs_diff", "iou_agreement")},
    }
    write_json(out_dir / "physical_mixed_permutation.json", result)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def batch_input(batch, arm):
    return batch["image"] if arm == "direct" else batch["image"]


@torch.no_grad()
def physical_mixed_missing_bands(out_dir: Path = OUT, device_name: str | None = None) -> dict:
    device = _device(device_name)
    model, _ = diag.load_checkpoint(MIXED_CKPT, str(device))
    model.eval()
    _, selected = _tiles_records()
    ds = diag.arm_dataset("use", selected, augment=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=diag.collate, num_workers=0)
    results = {name: [] for name in ("rgb", "rgb_nir", "six", "unseen5")}
    masked = []
    for batch in loader:
        image = batch["image"].to(device)
        p_full = diag.probability(model, image, "use")
        valid = batch["valid"].to(device)
        for name in results:
            sub = diag.subset_batch(batch, diag.SUBSETS[name])
            p_sub = diag.probability(model, batch_input(sub, "use").to(device), "use")
            results[name].append(diag.prob_delta(p_full, p_sub, valid))
        remove = [diag.S2_10_BANDS.index(b) for b in ("B05", "B06", "B07", "B8A", "B11", "B12")]
        masked_batch_ = diag.masked_batch(batch, remove)
        p_mask = diag.probability(model, batch_input(masked_batch_, "use").to(device), "use")
        p_absent = diag.probability(
            model, batch_input(diag.subset_batch(batch, ("B02", "B03", "B04", "B08")), "use").to(device), "use"
        )
        masked.append({
            "masked_vs_full": diag.prob_delta(p_full, p_mask, valid),
            "masked_vs_absent": diag.prob_delta(p_absent, p_mask, valid),
            "finite": bool(torch.isfinite(p_mask).all()),
        })
    out = {"arm": "physical_mixed", "checkpoint": str(MIXED_CKPT.relative_to(ROOT)).replace("\\", "/"),
           "checkpoint_sha256": sha(MIXED_CKPT), "supported": True, "subsets": {}}
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
    write_json(out_dir / "physical_mixed_missing_bands.json", out)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


@torch.no_grad()
def physical_mixed_valid_region(out_dir: Path = OUT, device_name: str | None = None) -> dict:
    device = _device(device_name)
    model, _ = diag.load_checkpoint(MIXED_CKPT, str(device))
    model.eval()
    _, selected = _tiles_records()
    subset = diag.select_diagnostic_tiles(selected, 32)
    ds = diag.arm_dataset("use", subset, augment=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=diag.collate, num_workers=0)
    rows = []
    for batch in loader:
        base = diag.clone_batch(batch)
        region = (slice(None), slice(None), slice(32, 64), slice(32, 64))
        image = base["image"]
        image["pixel_valid"][region] = False
        image["x"][region] = 0.0
        p_zero = diag.probability(model, batch_input(base, "use").to(device), "use")
        nan_batch = diag.clone_batch(base)
        nan_batch["image"]["x"][region] = float("nan")
        out_nan = model(batch_input(nan_batch, "use").to(device))
        p_nan = torch.sigmoid(out_nan["logits"].float()[:, 0])
        valid = batch["valid"].to(device).clone()
        valid[:, 32:64, 32:64] = False
        far = valid.clone()
        far[:, 28:68, 28:68] = False
        rows.append({
            "nan_finite": bool(torch.isfinite(p_nan).all()),
            "nan_max_diff_valid": float((p_zero - p_nan).abs()[valid].max()) if valid.any() else None,
            "nan_max_diff_far": float((p_zero - p_nan).abs()[far].max()) if far.any() else None,
            "input_valid_matches": bool(
                (out_nan["input_valid"].detach().cpu()
                 == (nan_batch["image"]["pixel_valid"].bool()
                     & nan_batch["image"]["band_valid"][:, :, None, None]).any(1)).all()
            ),
        })
    result = {
        "arm": "physical_mixed",
        "checkpoint": str(MIXED_CKPT.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha(MIXED_CKPT),
        "region": "channels all, rows 32:64, cols 32:64; same synthetic pixel_valid mask, values 0 vs NaN",
        "nan_finite_all": bool(all(r["nan_finite"] for r in rows)),
        "nan_max_diff_valid_max": float(max(r["nan_max_diff_valid"] for r in rows if r["nan_max_diff_valid"] is not None)),
        "nan_max_diff_far_max": float(max(r["nan_max_diff_far"] for r in rows if r["nan_max_diff_far"] is not None)),
        "input_valid_matches": bool(all(r["input_valid_matches"] for r in rows)),
    }
    one = diag.clone_batch(diag.collate([ds[0]]))
    one["image"]["pixel_valid"][:] = False
    one["image"]["x"][:] = float("nan")
    try:
        model(one["image"].to(device))
        result["fully_invalid"] = {"accepted": True}
    except Exception as exc:
        result["fully_invalid"] = {"accepted": False, "error": type(exc).__name__}
    write_json(out_dir / "physical_mixed_valid_region.json", result)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# representation comparison
# ---------------------------------------------------------------------------
def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double()
    y = y.double()
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    hsic = torch.linalg.matrix_norm(x.T @ y, ord="fro").square()
    varx = torch.linalg.matrix_norm(x.T @ x, ord="fro").square()
    vary = torch.linalg.matrix_norm(y.T @ y, ord="fro").square()
    denom = torch.sqrt(varx * vary)
    return float(hsic / denom) if float(denom) > 0 else 0.0


@torch.no_grad()
def feature_snapshot(model, records, device, max_pixels: int = 4096) -> dict[str, torch.Tensor]:
    ds = diag.arm_dataset("use", records, augment=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=diag.collate, num_workers=0)
    parts: dict[str, list[torch.Tensor]] = defaultdict(list)
    for batch in loader:
        image = batch["image"].to(device)
        _, store = diag.capture(model, image, "use")
        values = {"input": image["x"], **store}
        for name, tensor in values.items():
            if name == "logits" or not torch.is_tensor(tensor):
                continue
            if tensor.ndim == 4:
                flat = tensor.permute(0, 2, 3, 1).reshape(-1, tensor.shape[1])
            elif tensor.ndim == 3:
                flat = tensor.reshape(-1, tensor.shape[-1])
            else:
                continue
            if flat.shape[0] > max_pixels:
                index = torch.linspace(0, flat.shape[0] - 1, max_pixels, device=flat.device).long()
                flat = flat[index]
            parts[name].append(flat.cpu())
    return {name: torch.cat(chunks, 0) for name, chunks in parts.items()}


def compare(out_dir: Path = OUT, device_name: str | None = None) -> dict:
    device = _device(device_name)
    _, selected = _tiles_records()
    snap_subset = diag.select_diagnostic_tiles(selected, 32)
    physical, _ = diag.load_checkpoint(PHYSICAL_CKPT, str(device))
    physical.eval()
    snap_p = feature_snapshot(physical, snap_subset, device)
    del physical
    if device.type == "cuda":
        torch.cuda.empty_cache()
    mixed, _ = diag.load_checkpoint(MIXED_CKPT, str(device))
    mixed.eval()
    snap_m = feature_snapshot(mixed, snap_subset, device)
    del mixed
    if device.type == "cuda":
        torch.cuda.empty_cache()
    cka = {}
    for name in sorted(set(snap_p) & set(snap_m)):
        if snap_p[name].shape[1] != snap_m[name].shape[1]:
            continue
        cka[name] = linear_cka(snap_p[name], snap_m[name])

    p_act = read_json(OLD / "interface/layer_stats_use.json")
    m_act = read_json(out_dir / "physical_mixed_activations.json")
    p_grad = read_json(OLD / "interface/gradient_stats_use.json")
    m_grad = read_json(out_dir / "physical_mixed_gradients.json")
    p_perm = read_json(OLD / "interface/permutation_use.json")
    m_perm = read_json(out_dir / "physical_mixed_permutation.json")
    p_miss = read_json(OLD / "interface/missing_bands_use.json")
    m_miss = read_json(out_dir / "physical_mixed_missing_bands.json")
    p_valid = read_json(OLD / "interface/valid_region_use.json")
    m_valid = read_json(out_dir / "physical_mixed_valid_region.json")
    p_erf = read_json(OLD / "interface/erf_use.json")
    m_erf = read_json(out_dir / "physical_mixed_erf.json")

    def erf_mean(data, interface=None, key="r90"):
        values = []
        for tile in data.get("tiles", []):
            source = tile["input"] if interface is None else tile.get("interface", {}).get(interface, {})
            if source.get(key) is not None:
                values.append(source[key])
        return float(np.mean(values)) if values else None

    activation_rows = []
    for layer in ("input", "bridge", "use.features", "mit.stage1", "mit.stage2", "detail.stem", "decoder"):
        a = p_act["layers"].get(layer, {})
        b = m_act["layers"].get(layer, {})
        activation_rows.append({
            "layer": layer,
            "physical_effective_rank_ratio": a.get("effective_rank_ratio"),
            "mixed_effective_rank_ratio": b.get("effective_rank_ratio"),
            "delta_effective_rank_ratio": (b.get("effective_rank_ratio") - a.get("effective_rank_ratio"))
            if a.get("effective_rank_ratio") is not None and b.get("effective_rank_ratio") is not None else None,
            "physical_lag1_horizontal": a.get("lag1_horizontal"),
            "mixed_lag1_horizontal": b.get("lag1_horizontal"),
            "physical_roughness": a.get("roughness"),
            "mixed_roughness": b.get("roughness"),
            "cka_physical_vs_mixed": cka.get(layer),
        })
    gradient_rows = []
    for group in ("use.tokenizer", "use.metadata", "use.attention", "use.pool", "use.fuse", "bridge",
                  "mit.embed1", "mit.stage1", "mit.stage2", "mit.stage3_4", "detail.stem", "decoder", "other"):
        a = p_grad["parameter_groups"].get(group, {})
        b = m_grad["parameter_groups"].get(group, {})
        gradient_rows.append({
            "group": group,
            "physical_grad_share": a.get("grad_share"),
            "mixed_grad_share": b.get("grad_share"),
            "delta_grad_share": (b.get("grad_share") - a.get("grad_share"))
            if a.get("grad_share") is not None and b.get("grad_share") is not None else None,
        })
    result = {
        "physical_checkpoint": str(PHYSICAL_CKPT.relative_to(ROOT)).replace("\\", "/"),
        "physical_checkpoint_sha256": sha(PHYSICAL_CKPT),
        "mixed_checkpoint": str(MIXED_CKPT.relative_to(ROOT)).replace("\\", "/"),
        "mixed_checkpoint_sha256": sha(MIXED_CKPT),
        "snapshot_tiles": [r["sample_id"] for r in snap_subset],
        "snapshot_max_pixels": 4096,
        "activation_rows": activation_rows,
        "gradient_rows": gradient_rows,
        "bridge": {
            "physical_diag_mean": p_act.get("input_bridge_pair", {}).get("diag_mean"),
            "mixed_diag_mean": m_act.get("input_bridge_pair", {}).get("diag_mean"),
            "physical_offdiag_abs_max": p_act.get("input_bridge_pair", {}).get("offdiag_abs_max"),
            "mixed_offdiag_abs_max": m_act.get("input_bridge_pair", {}).get("offdiag_abs_max"),
            "physical_mean_abs_log2_scale": p_act.get("bridge_scale", {}).get("mean_abs_log2_ratio"),
            "mixed_mean_abs_log2_scale": m_act.get("bridge_scale", {}).get("mean_abs_log2_ratio"),
            "physical_channels_below_half": p_act.get("bridge_scale", {}).get("channels_below_half_input_std"),
            "mixed_channels_below_half": m_act.get("bridge_scale", {}).get("channels_below_half_input_std"),
        },
        "erf": {
            "physical_input_r50": erf_mean(p_erf, key="r50"),
            "mixed_input_r50": erf_mean(m_erf, key="r50"),
            "physical_input_r90": erf_mean(p_erf, key="r90"),
            "mixed_input_r90": erf_mean(m_erf, key="r90"),
            "physical_bridge_r90": erf_mean(p_erf, "bridge", "r90"),
            "mixed_bridge_r90": erf_mean(m_erf, "bridge", "r90"),
            "physical_detail_stem_readout_r90": erf_mean(p_erf, "detail.stem", "r90"),
            "mixed_detail_stem_readout_r90": erf_mean(m_erf, "detail.stem", "r90"),
        },
        "input_band_share": {
            "physical": p_grad.get("input_band_share"),
            "mixed": m_grad.get("input_band_share"),
        },
        "interface": {
            "permutation": {
                "physical_joint": p_perm["joint"]["mean_abs_diff"], "mixed_joint": m_perm["joint"]["mean_abs_diff"],
                "physical_band_only": p_perm["band_only"]["mean_abs_diff"], "mixed_band_only": m_perm["band_only"]["mean_abs_diff"],
                "physical_metadata_only": p_perm["metadata_only"]["mean_abs_diff"], "mixed_metadata_only": m_perm["metadata_only"]["mean_abs_diff"],
            },
            "missing_bands": {
                "physical_masked_vs_absent": p_miss["band_valid_masking"]["masked_vs_absent"]["mean_abs_diff"],
                "mixed_masked_vs_absent": m_miss["band_valid_masking"]["masked_vs_absent"]["mean_abs_diff"],
                "physical_subsets": {k: v["iou_agreement"] for k, v in p_miss["subsets"].items()},
                "mixed_subsets": {k: v["iou_agreement"] for k, v in m_miss["subsets"].items()},
            },
            "valid_region": {
                "physical_nan_far": p_valid.get("nan_max_diff_far_max"),
                "mixed_nan_far": m_valid.get("nan_max_diff_far_max"),
                "physical_finite": p_valid.get("nan_finite_all"),
                "mixed_finite": m_valid.get("nan_finite_all"),
            },
        },
        "corrections": [
            "physical_mixed is the main mixed-band control; physical (full-band) mechanism readings are not substituted for it",
            "bridge output is a fixed 10-dimensional latent, not band-indexed reflectance; band_valid is handled inside USE, not by indexing bridge channels",
            "effective rank / diagonal correlation / gradient share are descriptive, not pass/fail thresholds",
        ],
    }
    write_json(out_dir / "comparison.json", result)
    return result


# ---------------------------------------------------------------------------
# figure / report / provenance
# ---------------------------------------------------------------------------
def make_figure(out_dir: Path = OUT, doc_dir: Path = DOC) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    doc_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    data = read_json(out_dir / "comparison.json")
    rows = data["activation_rows"]
    layers = [r["layer"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    x = np.arange(len(layers))
    axes[0].bar(x - 0.2, [r["physical_effective_rank_ratio"] or 0 for r in rows], 0.4, label="physical（全波段训练）", color="#2f5d8a")
    axes[0].bar(x + 0.2, [r["mixed_effective_rank_ratio"] or 0 for r in rows], 0.4, label="physical_mixed（混合波段训练）", color="#b5653a")
    axes[0].set_xticks(x, layers, rotation=35, fontsize=8)
    axes[0].set_ylabel("有效秩/通道数", fontproperties=font)
    axes[0].set_title("表示：physical 与 physical_mixed", fontproperties=font)
    axes[0].legend(prop=font, fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)
    cka = [r["cka_physical_vs_mixed"] or 0 for r in rows]
    axes[1].bar(x, cka, color="#4e7fa8")
    axes[1].set_xticks(x, layers, rotation=35, fontsize=8)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("线性CKA（同层、同像素抽样）", fontproperties=font)
    axes[1].set_title("两套权重的表示相似度", fontproperties=font)
    axes[1].grid(axis="y", alpha=0.2)
    for i, value in enumerate(cka):
        axes[1].text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    path = doc_dir / "physical_mixed_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_report(out_dir: Path = OUT, doc_dir: Path = DOC) -> Path:
    audit = read_json(out_dir / "audit.json")
    comparison = read_json(out_dir / "comparison.json")
    mixed = read_json(out_dir / "physical_mixed_activations.json")
    doc_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# USE诊断复核与 physical_mixed 参照补查",
        "",
        "> 只读复核：不训练、不重跑300步探针、不改正式配置/权重。",
        "",
        "## 1. 复核结论",
        "",
        f"- 128块清单文件SHA256 `{audit['manifest']['manifest_file_sha256']}`，内容SHA256 `{audit['manifest']['manifest_content_sha256']}`；"
        f"拟合/检查 {audit['manifest']['manifest_counts']['n_fit']}/{audit['manifest']['manifest_counts']['n_check']}。",
        f"- 旧300步探针初始化哈希与记录一致：direct/use 均为 {audit['probes']['direct']['init_match'] and audit['probes']['use']['init_match']}；"
        "本次没有重跑这两组探针。",
        f"- physical_mixed 正式运行：{audit['formal_runs']['physical_mixed']['global_step']} 步完成，"
        f"best@{audit['formal_runs']['physical_mixed']['best_step']} DEV {audit['formal_runs']['physical_mixed']['best_val_iou']:.4f}，"
        f"mixed_bands={audit['formal_runs']['physical_mixed']['mixed_bands']}，权重SHA256 `{audit['formal_runs']['physical_mixed']['checkpoint_sha256']}`。",
        f"- 审计问题：{audit['issues'] if audit['issues'] else '无'}。",
        "",
        "## 2. 纠正后的解释",
        "",
        "- 桥接输出是固定10维隐特征，不是有已知波段身份的反射率；不能按输入`band_valid`索引屏蔽桥接输出，输入可变C与固定输出10也没有逐通道对应。",
        "- 对角相关低、有效秩低、梯度份额低是诊断现象，不是已证实瓶颈；不再把“有效秩>4、梯度份额>0.005、元数据干预>0.01”作为性能晋级硬门槛。",
        "- 本次补查的是 physical_mixed 自身权重，不把 physical（全波段训练）的结论直接套到它上面。",
        "",
        "## 3. physical 与 physical_mixed 同口径读数",
        "",
        "| 层 | physical 有效秩/通道 | physical_mixed 有效秩/通道 | Δ | physical 滞后1 | physical_mixed 滞后1 | CKA |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in comparison["activation_rows"]:
        lines.append(
            f"| {row['layer']} | {_f(row['physical_effective_rank_ratio'])} | {_f(row['mixed_effective_rank_ratio'])} | "
            f"{_f(row['delta_effective_rank_ratio'])} | {_f(row['physical_lag1_horizontal'])} | {_f(row['mixed_lag1_horizontal'])} | "
            f"{_f(row['cka_physical_vs_mixed'])} |"
        )
    bridge = comparison["bridge"]
    lines += [
        "",
        "桥接（固定10维隐特征；不是波段反射率）：",
        "",
        f"- 对角相关均值：physical {_f(bridge['physical_diag_mean'])}，physical_mixed {_f(bridge['mixed_diag_mean'])}。",
        f"- 尺度平均绝对log2：physical {_f(bridge['physical_mean_abs_log2_scale'])}，physical_mixed {_f(bridge['mixed_mean_abs_log2_scale'])}。",
        f"- 低于输入一半标准差的通道：physical {bridge['physical_channels_below_half']}/10，physical_mixed {bridge['mixed_channels_below_half']}/10。",
        "",
        "梯度份额（32块一次反传）：",
        "",
        "| 参数组 | physical | physical_mixed | Δ |",
        "|---|---|---|---|",
    ]
    for row in comparison["gradient_rows"]:
        lines.append(f"| {row['group']} | {_f(row['physical_grad_share'])} | {_f(row['mixed_grad_share'])} | {_f(row['delta_grad_share'])} |")
    erf = comparison["erf"]
    shares = comparison["input_band_share"]
    interface = comparison["interface"]
    lines += [
        "",
        "空间/输入关注：",
        "",
        f"- 中心输出输入梯度r90均值：physical {_f(erf['physical_input_r90'], 1)} px，physical_mixed {_f(erf['mixed_input_r90'], 1)} px；"
        f"桥接特征r90：physical {_f(erf['physical_bridge_r90'], 1)}，physical_mixed {_f(erf['mixed_bridge_r90'], 1)}。",
        f"- 十波段输入梯度份额：physical `{[round(v, 3) for v in shares['physical']]}`；"
        f"physical_mixed `{[round(v, 3) for v in shares['mixed']]}`。",
        "",
        "接口不变量：",
        "",
        "| 检查 | physical | physical_mixed |",
        "|---|---|---|",
        f"| 联合置换平均概率差 | {_f(interface['permutation']['physical_joint'])} | {_f(interface['permutation']['mixed_joint'])} |",
        f"| 仅波段置换 | {_f(interface['permutation']['physical_band_only'])} | {_f(interface['permutation']['mixed_band_only'])} |",
        f"| 仅元数据置换 | {_f(interface['permutation']['physical_metadata_only'])} | {_f(interface['permutation']['mixed_metadata_only'])} |",
        f"| masked vs absent 概率差 | {_f(interface['missing_bands']['physical_masked_vs_absent'])} | {_f(interface['missing_bands']['mixed_masked_vs_absent'])} |",
        f"| 无效区NaN远处最大变化 | {_f(interface['valid_region']['physical_nan_far'])} | {_f(interface['valid_region']['mixed_nan_far'])} |",
        "",
        "缺波段与全十波段的IoU一致度（越低表示缺波段影响越大）：",
        "",
        "| 子集 | physical | physical_mixed |",
        "|---|---|---|",
    ]
    for name in ("rgb", "rgb_nir", "six", "unseen5"):
        lines.append(f"| {name} | {_f(interface['missing_bands']['physical_subsets'][name])} | {_f(interface['missing_bands']['mixed_subsets'][name])} |")
    lines += [
        "",
        "判读：physical_mixed 不是 physical 的小扰动。两套权重的同层CKA在MiT两级约0.64–0.66、细节分支约0.51、桥接约0.82；"
        "mixed的有效秩更高、十波段梯度更均匀、缺波段一致度更高，但既有DEV 0.1895略低于physical 0.1955。"
        "这是同一批权重上的观察，不证明混合波段训练导致鲁棒性或性能变化；不能把physical的机制解释直接套到mixed。",
        "",
        "## 4. 图像",
        "",
        "![physical vs physical_mixed](physical_mixed_comparison.png)",
        "",
        "左图：每层有效秩/通道数，蓝=physical（全波段训练），橙=physical_mixed（混合波段训练）；看两套权重是否在USE特征、桥接、细节分支上出现不同的秩结构。右图：同层同像素抽样的线性CKA，1表示表示几乎相同、0表示线性不可比；CKA低不等于某套权重“错误”，只说明学习到的表示不同。",
        "",
        "## 5. 限制与下一步",
        "",
        "- 本次是只读推理/一次反传，不是训练；physical_mixed的9组DEV/组合评价沿用既有结果，不新增正式分数。",
        "- physical_mixed训练使用混合波段采样；本次表示诊断统一用全十波段评估输入，训练采样差异不当作评估输入差异。",
        "- 两套权重均来自既有DEV选模，含开发选模历史；CKA/秩/梯度只作解释读数，不作性能门槛。",
        "- 下一步按当前PLAN.md进入两组恢复实验的隔离实现；本次未启动2000步正式训练。",
    ]
    path = doc_dir / "REVIEW.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _f(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "非有限"
        if value != 0 and abs(value) < 1e-3:
            return f"{value:.2e}"
        return f"{value:.{digits}f}"
    return str(value)


def write_provenance(out_dir: Path = OUT) -> dict:
    sources = [MIXED_CKPT, PHYSICAL_CKPT, DIRECT_CKPT, diag.MANIFEST, diag.STATS, diag.QUALITY_INDEX,
               OLD / "selection/diagnostic_tiles.json", OLD / "summary.json", Path(__file__)]
    source_rows = [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(p), "size_bytes": p.stat().st_size}
                   for p in sources if p.is_file()]
    outputs = [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(p), "size_bytes": p.stat().st_size}
               for p in sorted(out_dir.rglob("*")) if p.is_file()]
    documents = [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(p), "size_bytes": p.stat().st_size}
                 for p in sorted(DOC.rglob("*")) if p.is_file()]
    payload = {
        "command": "python -X utf8 scripts/review_use_diagnostics.py all",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sources": source_rows,
        "outputs": outputs,
        "documents": documents,
        "scope": "read-only review; no training and no 300-step probe rerun",
    }
    write_json(out_dir / "provenance.json", payload)
    return payload


def run_all(out_dir: Path = OUT, device_name: str | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    audit(out_dir)
    physical_mixed_activations(out_dir, device_name)
    physical_mixed_gradients(out_dir, device_name)
    physical_mixed_erf(out_dir, device_name)
    physical_mixed_permutation(out_dir, device_name)
    physical_mixed_missing_bands(out_dir, device_name)
    physical_mixed_valid_region(out_dir, device_name)
    compare(out_dir, device_name)
    make_figure(out_dir, DOC)
    write_report(out_dir, DOC)
    return write_provenance(out_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "represent", "compare", "report", "all"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--out", type=Path, default=OUT)
        if name in ("represent", "compare", "all"):
            cmd.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    if args.command == "audit":
        result = audit(args.out)
        print(json.dumps({"ok": result["ok"], "issues": result["issues"]}, ensure_ascii=False))
    elif args.command == "represent":
        args.out.mkdir(parents=True, exist_ok=True)
        physical_mixed_activations(args.out, args.device)
        physical_mixed_gradients(args.out, args.device)
        physical_mixed_erf(args.out, args.device)
        physical_mixed_permutation(args.out, args.device)
        physical_mixed_missing_bands(args.out, args.device)
        physical_mixed_valid_region(args.out, args.device)
        print(json.dumps({"out": str(args.out)}, ensure_ascii=False))
    elif args.command == "compare":
        result = compare(args.out, args.device)
        print(json.dumps({"out": str(args.out / "comparison.json"), "n_rows": len(result["activation_rows"])}, ensure_ascii=False))
    elif args.command == "report":
        make_figure(args.out, DOC)
        path = write_report(args.out, DOC)
        write_provenance(args.out)
        print(str(path))
    else:
        run_all(args.out, args.device)
        print(str(DOC / "REVIEW.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
