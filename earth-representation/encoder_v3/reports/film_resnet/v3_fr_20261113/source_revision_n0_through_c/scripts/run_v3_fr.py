"""Read-only baseline audit and explicit V3-FR engineering stages."""
import argparse
from collections import Counter
import csv
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import ssl  # Load the standard library module before adding the frozen project path.
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def hash_file(path):
    with path.open("rb") as stream:
        from hashlib import file_digest
        return file_digest(stream, "sha256").hexdigest()


def table(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0])
        writer.writeheader()
        writer.writerows(rows)


def mask_sha(masks):
    h = sha256()
    for mask in masks:
        arr = mask.detach().cpu().numpy()
        h.update(np.asarray(arr.shape, dtype="<i8").tobytes())
        h.update(np.packbits(arr.reshape(-1), bitorder="little").tobytes())
    return h.hexdigest()


def bind_runtime(release):
    runtime = release / "runtime"
    # Add only after standard library imports so the historical src/ssl cannot shadow SSL.
    sys.path[:0] = [str(runtime / "scripts"), str(runtime / "src")]
    import encoder_v3
    encoder_v3.__path__.append(str(Path(__file__).resolve().parents[1] / "src/encoder_v3"))


def identity(args):
    from encoder_v3.data import sha256 as old_sha
    from encoder_v3.latent_response.checkpoint import digest, parameter_sha
    from encoder_v3.shared_mae.data import identity as data_identity, load_data
    from encoder_v3.shared_mae.geometry import metric_geometry
    from encoder_v3.latent_response.types import NativeGrid
    from train_patch_latent import Config, Pretraining

    out, release, prior = args.output, args.release, args.prior
    status, config, binding, manifest = [read(release / f"{key}.json")
                                        for key in ("status", "config", "identity", "source_manifest")]
    previous = read(prior / "baseline_identity.json")
    for path, expected in manifest.items():
        assert old_sha(release / "runtime" / path) == expected, path
    assert len(manifest) == previous["source_count"] and digest(manifest) == previous["source_snapshot_hash"]
    assert hash_file(release / "config.json") == previous["config_file_sha256"]
    assert status["run_id"] == previous["parent_id"] and status["cursor"] == 6990
    data, resolver = load_data()
    current = data_identity(config, data, resolver)
    assert dict(current, frozen_source=digest(manifest), architecture=config["architecture"]) == binding
    allowed = [np._core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType]
    model_config = Config(**config["model"])
    model = Pretraining(model_config)
    for name, key in (("best.pt", "checkpoint_hash"), ("last.pt", "last_checkpoint_hash")):
        assert hash_file(release / name) == previous[key]
        with torch.serialization.safe_globals(allowed):
            state = torch.load(release / name, map_location="cpu", weights_only=True)
        assert state["identity"] == binding and state["runner"]["run_id"] == status["run_id"]
        model.load_state_dict(state["model"], strict=True)
        assert parameter_sha(model) == previous["parameter_sha256"]
    torch.manual_seed(config["training"]["seed"])
    assert parameter_sha(Pretraining(model_config)) == status["initial_parameter_sha"]
    assert tuple(model_config.widths) == (32,64,96) and tuple(model_config.reductions) == (8,4,2)
    print("S0 frozen source, weights, initialization and data binding verified", flush=True)

    layouts = {}
    counts, census, errors = Counter(), [], []
    for role, ds in data.items():
        for row in ds.rows:
            counts[(role, row["sensor"])] += 1
            try:
                specs = tuple(dict.fromkeys((tuple(b["transform"]), tuple(b["shape"]), b["crs"]) for b in row["bands"]))
                if specs not in layouts:
                    grids = [NativeGrid(*spec) for spec in specs]
                    layouts[specs] = metric_geometry([SimpleNamespace(grid=g) for g in grids], 40.)
                geometry = layouts[specs]
                gsd = Counter()
                pack = native = edges = 0
                for b in row["bands"]:
                    affine, (h,w) = b["transform"], b["shape"]
                    a,e = affine[0], -affine[4]
                    factor = max((r for r in (1,2,4) if r*max(a,e)<=40.), default=1)
                    gsd[f"{a:g}x{e:g}@{factor}"] += 1
                    native += h*w
                    pack += math.ceil(h/factor)*math.ceil(w/factor)
                    def overlaps(n, offset, step):
                        lo = (offset+np.arange(n)*step)/20.
                        hi = (offset+(np.arange(n)+1)*step)/20.
                        lo = np.where(abs(lo-lo.round()) < 1e-8, lo.round(), lo)
                        hi = np.where(abs(hi-hi.round()) < 1e-8, hi.round(), hi)
                        return int((np.ceil(hi)-np.floor(lo)).sum())
                    edges += overlaps(w, affine[2]-geometry.origin[0], a)*overlaps(
                        h, affine[5]-h*e-geometry.origin[1], e)
                census.append(dict(role=role, sensor=row["sensor"], id=row["id"], bands=len(row["bands"]),
                                   gsd_factors=json.dumps(gsd, sort_keys=True), shapes=json.dumps(sorted({tuple(b["shape"]) for b in row["bands"]})),
                                   native_values=native, packed_positions=pack, parent_h=geometry.shape[0],
                                   parent_w=geometry.shape[1], slots=math.prod(geometry.shape)*4, overlap_edges=edges,
                                   units="projected metres", front_MACs_lower_bound=pack*36*32**2))
            except (ValueError, KeyError) as error:
                errors.append(dict(id=row["id"], role=role, error=str(error)))
    if census:
        table(out / "geometry_census.csv", census)
    write(out / "geometry_census_summary.json",
          dict(status="BLOCKED_GEOMETRY" if errors else "PASS", failures=errors,
               counts={f"{r}/{s}": n for (r,s),n in sorted(counts.items())},
               maximum={s:{k:max((r for r in census if r["role"]=="TRAIN" and r["sensor"]==s), key=lambda r:r[k])
                           for k in ("native_values","bands","slots","overlap_edges")}
                        for s in sorted({r["sensor"] for r in census if r["role"]=="TRAIN"})}))
    assert not errors, errors[:3]
    old_panel = read(prior / "p2_information/panel.json")["rows"]
    fit = [max((r for r in old_panel if r["sensor"]==sensor and r["role"]=="fit"), key=lambda r:r["texture"])
           for sensor in ("S2","L8")]
    held = [max((r for r in old_panel if r["sensor"]==sensor and r["role"]=="read_only"), key=lambda r:r["texture"])
            for sensor in ("S2","L8","L9","EnMAP")]
    assert len({r["id"] for r in fit+held}) == 6
    assert all(data["TRAIN"].rows[r["index"]]["id"] == r["id"] for r in fit+held)
    fixed = read(prior / "acceptance_policy.json")["P0_fixed_samples"]
    paired = read(prior / "summary/paired_comparisons.json")["B-DETAIL-OFF"]["B1/all"]["selected"]
    write(out / "data_panel.json", dict(fit=fit, read_only=held, fixed_val=fixed, historic_display=list(paired.values()),
                                      selection="Reuse metadata/texture-only frozen TRAIN panel; choose highest texture per role and sensor. Prior B1 display panel is read-only."))
    write(out / "baseline_identity.json", previous | dict(status="PASS", initial_parameter_hash=status["initial_parameter_sha"],
          design_sha256=hash_file(args.design), plan_sha256=hash_file(args.plan),
          reference_git_commit="d330714874274c69eb07bf2466f1d8b74f708661",
          prior_results_commit="521ace47907a458f309133320e6dcc66de88850c"))
    write(out / "dataset_binding.json", current)
    write(out / "environment.json", dict(executable=sys.executable, python=sys.version, torch=str(torch.__version__),
          cuda=torch.version.cuda, devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))
    for src, name in ((args.design, "DESIGN_FROZEN.md"), (args.plan, "PLAN_FROZEN.md"),
                      (release / "source_manifest.json", "source_manifest.json"),
                      (release / "config.json", "baseline_config.json"),
                      (prior / "p0_baseline/worktree_vs_HEAD.patch", "frozen_baseline.patch")):
        shutil.copyfile(src, out / name)
    print("S0 geometry and sample panels frozen", len(census), flush=True)


def compare(expected, actual, tolerance, path="root"):
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys(), path
        return max((compare(v, actual[k], tolerance, path+"/"+k) for k,v in expected.items()), default=0.)
    if isinstance(expected, list):
        return max((compare(a,b,tolerance,path+f"/{i}") for i,(a,b) in enumerate(zip(expected,actual,strict=True))), default=0.)
    if isinstance(expected, float):
        delta = abs(expected-actual)
        assert math.isfinite(actual) and delta <= tolerance["atol"]+tolerance["rtol"]*abs(expected), (path, expected, actual)
        return delta
    assert expected == actual, (path, expected, actual)
    return 0.


def replay(args):
    from encoder_v3.shared_mae.data import load_data, sample_from_dataset
    from encoder_v3.hierarchical_revised.tasks import task_masks
    from train_patch_latent import Config, Pretraining, validate
    from probe_patch_latent import texture_mse
    out, release = args.output, args.release
    report = read(out / "baseline_identity.json")
    assert report["status"] == "PASS"
    policy = read(out / "acceptance_policy.json")
    stage = out / "s1_baseline"
    stage.mkdir(exist_ok=False)
    settings = read(release / "config.json")["training"]
    device, seed = settings["device"], settings["validation_seed"]
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    data, resolver = load_data()
    val = data["VAL"]
    cfg = Config(**read(release / "config.json")["model"])
    allowed = [np._core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(release / "best.pt", map_location="cpu", weights_only=True)
    model = Pretraining(cfg).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    panel = read(out / "data_panel.json")
    reference = read(release / "validation_epoch_30.json")
    ids = set(panel["fixed_val"]) | set(panel["historic_display"])
    raw, checks, display = {}, [], []
    reference_metrics = policy["baseline_replay"]
    for index, row in enumerate(val.rows):
        if row["id"] not in ids:
            continue
        sample = sample_from_dataset(val, index, resolver, device)
        for kind in ("A","B1"):
            keeps, targets = task_masks(sample, cfg, kind, seed, 0)
            with torch.no_grad():
                result = model(sample, kind=kind, seed=seed)
                main = model.decoder(result.encoded, [c.grid for c in sample.channels], targets)
                expected = reference["members"][index]["tasks"][kind]
                compare(expected, dict(mse=float(result.mse), texture_gradient_mse=texture_mse(sample,result),
                        targets=result.predictions.numel(),
                        channel_mse={sample.channels[i].name:float(v) for i,v in result.channel_loss.items()}),
                        reference_metrics, sample.identifier+"/"+kind)
                for label, mask in (("target", targets), ("qa_invalid", [~c.valid for c in sample.channels])):
                    changed = replace(sample, channels=[replace(c, values=torch.where(m, 1024-c.values,c.values))
                                                         for c,m in zip(sample.channels,mask,strict=True)])
                    other = model(changed, kind=kind, seed=seed)
                    torch.testing.assert_close(result.predictions, other.predictions, **policy["gpu_prediction"])
                    assert mask_sha(result.targets)==mask_sha(other.targets)==mask_sha(targets)
                    delta = abs(float(result.loss-other.loss))
                    assert delta > 0 if label=="target" else delta <= policy["gpu_prediction"]["atol"]
                    checks.append(dict(id=sample.identifier, task=kind, mutation=label, loss_delta=delta,
                                       max_prediction_delta=float((result.predictions-other.predictions).abs().max())))
            band = max(range(len(targets)), key=lambda i:int(targets[i].sum()))
            offset = sum(int(m.sum()) for m in targets[:band])
            c, keep, target = sample.channels[band], keeps[band], targets[band]
            total = result.predictions[offset:offset+int(target.sum())]
            prefix = f"{sample.identifier}_{kind}"
            full = torch.full_like(c.values, torch.nan)
            for key,tensor in dict(truth=c.values, qa=c.valid, visible_mask=keep, target_mask=target,
                                   visible_values=torch.where(keep,c.values,0.), total_prediction=full.masked_scatter(target,total),
                                   main_prediction=full.masked_scatter(target,main[band]),
                                   detail_contribution=full.masked_scatter(target,total-main[band])).items():
                raw[prefix+"_"+key] = tensor.cpu().numpy()
            actual = total.cpu().numpy().astype(np.float64)
            truth = c.values[target].cpu().numpy().astype(np.float64)
            lo, hi = np.quantile(c.values[c.valid].cpu().numpy(),[.02,.98]).tolist()
            ref_band = next((i for i,m in enumerate(keeps) if bool(m.any())),None)
            if ref_band is not None:
                bc = sample.channels[ref_band]
                raw[prefix+"_visible_reference"] = torch.where(keeps[ref_band],bc.values,torch.nan).cpu().numpy()
            clipped = [float(np.mean(actual < lo)),float(np.mean(actual > hi))]
            stats = lambda x: dict(mean=float(x.mean()), std=float(x.std()), minimum=float(x.min()),
                                   maximum=float(x.max()), p01_p05_p50_p95_p99=np.quantile(x,[.01,.05,.5,.95,.99]).tolist())
            pred_map, truth_map, keep_mask = [raw[prefix+"_"+k] for k in ("total_prediction","truth","target_mask")]
            signed = {}
            for name,field in (("prediction",pred_map),("truth",truth_map)):
                terms = [np.diff(field,axis=0)[keep_mask[1:] & keep_mask[:-1]],
                         np.diff(field,axis=1)[keep_mask[:,1:] & keep_mask[:,:-1]]]
                signed[name] = np.concatenate(terms).astype(np.float64)
            display.append(dict(id=sample.identifier, task=kind, band=c.name, affine=c.grid.affine, crs=c.grid.crs,
                reference_band=sample.channels[ref_band].name if ref_band is not None else None,
                reference_affine=sample.channels[ref_band].grid.affine if ref_band is not None else None,
                target_count=len(truth), display_low=lo, display_high=hi, clipped_low=clipped[0], clipped_high=clipped[1],
                truth=stats(truth), prediction=stats(actual), constant_zero_mse=float(np.mean(truth**2)),
                variance_ratio=float(actual.var()/truth.var()) if truth.var()>1e-12 else None,
                correlation=float(np.corrcoef(actual,truth)[0,1]) if min(actual.std(),truth.std())>1e-6 else None,
                gradient_energy={k:float(np.mean(v**2)) if len(v) else None for k,v in signed.items()},
                signed_gradient_mse=float(np.mean((signed["prediction"]-signed["truth"])**2)) if len(signed["truth"]) else None,
                detail_energy=float((total-main[band]).square().mean()),
                display_saturation=max(clipped) >= .95))
    np.savez_compressed(stage / "raw_predictions.npz", **raw)
    write(stage / "display_audit.json", display)
    write(stage / "leakage.json", checks)
    print("S1 fixed samples, raw predictions and separate interventions passed", len(checks), flush=True)
    result = validate(model, val, resolver, settings, lambda: False)
    fields = ("observations","metric","tasks","sensors","bands","members")
    errors = {k:compare(reference[k],result[k],reference_metrics,k) for k in fields}
    old_obs = {(r["observation"],r["task"]):r for r in csv.DictReader(
        (args.prior/"summary/metrics_by_observation_p0.csv").open(encoding="utf-8-sig"))}
    old_bands = {(r["observation"],r["task"],r["band"]):r for r in csv.DictReader(
        (args.prior/"summary/metrics_by_band_p0.csv").open(encoding="utf-8-sig"))}
    observations, bands = [], []
    for index, member in enumerate(result["members"]):
        sample = sample_from_dataset(val,index,resolver,"cpu")
        assert sample.identifier==member["id"]
        for kind,row in member["tasks"].items():
            keeps, targets = task_masks(sample,cfg,kind,seed,0)
            key = (member["id"],kind)
            hashes = dict(target_mask_sha256=mask_sha(targets), visible_mask_sha256=mask_sha(keeps))
            expected = old_obs.pop(key)
            assert all(expected[k]==v for k,v in hashes.items()) and int(expected["targets"])==row["targets"], key
            observations.append(dict(observation=key[0], task=kind, sensor=member["sensor"], mse=row["mse"],
                                     texture_gradient_mse=row["texture_gradient_mse"], targets=row["targets"], **hashes))
            for channel, keep, target in zip(sample.channels,keeps,targets,strict=True):
                if channel.name not in row["channel_mse"]:
                    continue
                band = old_bands.pop((*key,channel.name))
                assert mask_sha([target]) == band["target_mask_sha256"] and mask_sha([keep])==band["visible_mask_sha256"]
                assert int(target.sum())==int(band["targets"])
                bands.append(band | dict(experiment="S1-REPLAY", mse=row["channel_mse"][channel.name]))
    assert not old_obs and not old_bands
    table(stage / "metrics_by_observation.csv",observations)
    table(stage / "metrics_by_band.csv",bands)
    write(stage / "baseline_replay.json", dict(status="PASS", tasks=result["tasks"], observations=len(observations)//2,
          bands=len(bands), reference_max_absolute_errors=errors, all_masks_match=True, elapsed_seconds=result["seconds"]))
    print("S1 full historical replay passed", result["tasks"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inspect","replay","executor","candidate","resume-gate","fit","ridge","compare"))
    for name in ("release","prior","output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--arms", nargs="+", default=["B0","BP","N0","N1","N2","C"])
    parser.add_argument("--seed", type=int, choices=(17,29,43), default=17)
    args = parser.parse_args()
    torch.set_num_threads(2)
    bind_runtime(args.release)
    try:
        if args.stage in ("executor","candidate","resume-gate","fit","ridge","compare"):
            if args.stage=="executor":
                from executor_gate import run
            elif args.stage=="candidate":
                from candidate_gate import run
            elif args.stage=="resume-gate":
                from candidate_gate import update_and_resume as run
            elif args.stage=="fit":
                from fit_diagnostics import fit as run
            elif args.stage=="ridge":
                from ridge_probe import run
            else:
                from compare_trials import compare as run
            run(args)
        else:
            {"inspect":identity,"replay":replay}[args.stage](args)
    except Exception as error:
        write(args.output / (args.stage+"_failure.json"),dict(status="BLOCKED", error=repr(error)))
        raise


if __name__ == "__main__":
    main()
