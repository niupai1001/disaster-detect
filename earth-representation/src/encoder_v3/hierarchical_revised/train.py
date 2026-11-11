"""Independent, bounded A/B1 formal runner; frozen-source release required."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import signal
import tempfile
import threading
import time
import uuid

import numpy as np
import torch

from encoder_v3.data import sha256
from encoder_v3.shared_mae.data import load_data, sample_from_dataset, EpochPlan, identity as data_identity, ROOT
from encoder_v3.shared_mae.train import write, StopRequested
from encoder_v3.latent_response.checkpoint import snapshot, restore, restore_rng, atomic_replace, digest, parameter_sha, validation_transaction
from encoder_v3.latent_response.run_control import append_event, reconcile_logs
from encoder_v3.latent_response.tvlp_writer import RunWriter
from .model import Config
from .tasks import Pretraining, available_tasks


def validate_config(config):
    if (config.get("architecture") != "hierarchical_v3_h_revised"
            or config.get("formal_training_authorized") is not True
            or config.get("decoder_variant") != "per_pixel"
            or config.get("authorization", {}).get("source") != "user_in_current_chat"):
        raise ValueError("Explicitly authorized per-pixel V3-H-R formal configuration required")
    Config(**config["model"])
    t = config["training"]
    for name in ("epochs", "batch_size", "threads", "offload_native_values"):
        if type(t[name]) is not int or t[name] < 1:
            raise ValueError("Positive integer required: "+name)
    if type(t["warmup_steps"]) is not int or t["warmup_steps"] < 0:
        raise ValueError("Nonnegative warmup required")
    if t["precision"] != "float32" or t["optimizer"] != "AdamW":
        raise ValueError("Only validated FP32 AdamW is enabled")
    if not (0 < t["min_lr"] <= t["lr"] and math.isfinite(t["lr"]) and
            math.isfinite(t["weight_decay"]) and t["weight_decay"] >= 0):
        raise ValueError("Invalid optimizer settings")
    if set(t["task_weights"]) != {"A", "B1"} or any(
        not math.isfinite(w) or w <= 0 for w in t["task_weights"].values()
    ):
        raise ValueError("Positive A/B1 weights required")


def learning_rate(index, total, settings):
    warmup = settings["warmup_steps"]
    if not 0 <= index < total or warmup >= total:
        raise ValueError("Invalid schedule position or warmup")
    if index < warmup:
        return settings["lr"]*(index+1)/warmup
    progress = (index-warmup)/max(1, total-1-warmup)
    return settings["min_lr"] + .5*(settings["lr"]-settings["min_lr"])*(1+math.cos(math.pi*progress))


def binding_for(config, datasets, resolver):
    binding = data_identity(config, datasets, resolver)
    for folder in ("hierarchical", "hierarchical_revised"):
        for path in (ROOT/"src/encoder_v3"/folder).glob("*.py"):
            binding["code_files"][str(path.relative_to(ROOT))] = sha256(path)
    path = ROOT/"scripts/train_v3_h_revised.py"
    binding["code_files"][str(path.relative_to(ROOT))] = sha256(path)
    binding.update(code=digest(binding["code_files"]), architecture=config["architecture"],
                   model=config["model"], decoder_variant="per_pixel")
    return binding


def save(path, model, optimizer, state, binding):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(dict(schema="v3_h_revised_formal_v1", identity=binding,
                        **snapshot(model, optimizer, state)), temporary)
        atomic_replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def resume(path, model, optimizer, state, binding):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "v3_h_revised_formal_v1" or payload.get("identity") != binding:
        raise ValueError("Checkpoint architecture/code/data/config binding mismatch")
    previous = snapshot(model, optimizer, state)
    try:
        restore(payload, model, optimizer, state)
    except Exception:
        restore(previous, model, optimizer, state)
        raise


def continue_from_checkpoint(model, optimizer, config, binding):
    continuation = config.get("continuation")
    if not continuation:
        return None
    parent = Path(continuation["parent_release"])
    checkpoint = parent/"formal_run/last.pt"
    report_path = Path(continuation["acceptance_report"])
    if sha256(checkpoint) != continuation["checkpoint_sha256"] or sha256(report_path) != continuation["acceptance_sha256"]:
        raise ValueError("Parent checkpoint or acceptance evidence changed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report["status"] != "PASS" or not report.get("checks") or not all(report["checks"].values())
            or not all(report["checks"].get(key) is True for key in ("A_improved","B1_improved",
                        "train_full_coverage","validation_full_coverage","checkpoint_binding","real_recovery_gate","tests_pass","optimizer_journal"))
            or report["final_checkpoint_sha256"] != continuation["checkpoint_sha256"]
            or report["run_id"] != continuation["source_run_id"]):
        raise ValueError("Continuation requires matching passed acceptance")
    payload = torch.load(checkpoint,map_location="cpu",weights_only=False)
    if payload.get("schema") != "v3_h_revised_formal_v1" or payload["runner"]["status"] != "COMPLETE":
        raise ValueError("Completed V3-H-R parent required")
    previous = payload["identity"]
    for key in ("data","members","physical_catalog","model","decoder_variant","architecture"):
        if previous[key] != binding[key]:
            raise ValueError("Continuation changed "+key)
    for name, expected in previous["code_files"].items():
        if name.replace("\\","/") not in ("src/encoder_v3/hierarchical_revised/train.py",
                                           "scripts/train_v3_h_revised.py"):
            if binding["code_files"].get(name) != expected:
                raise ValueError("Continuation changed model/data code: "+name)
    parent_config = json.loads((parent/"config.json").read_text(encoding="utf-8"))
    if digest(parent_config) != previous["config"]:
        raise ValueError("Parent configuration changed")
    for name,value in parent_config["training"].items():
        if name != "epochs" and config["training"][name] != value:
            raise ValueError("Continuation changed training setting: "+name)
    expected_epoch = payload["runner"].get("sampling_start_epoch",0)+parent_config["training"]["epochs"]
    if continuation["start_epoch"] != expected_epoch or payload["runner"]["viewer"]["run_id"] != continuation["source_run_id"]:
        raise ValueError("Continuation epoch or Run ID mismatch")
    model.load_state_dict(payload["model"],strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload["rng"])
    if parameter_sha(model) != payload["runner"]["final_parameter_sha"]:
        raise ValueError("Parent parameters do not match committed state")
    return dict(run_id=continuation["source_run_id"],checkpoint_sha256=continuation["checkpoint_sha256"],
                acceptance_sha256=continuation["acceptance_sha256"],sampling_start_epoch=expected_epoch,
                inherited_optimizer_updates=payload["runner"]["spent_updates"])


def observation(model, sample, settings, epoch, backward=False, stopped=lambda: False):
    kinds = available_tasks(sample, model.config)
    weights = settings["task_weights"]
    total = sum(weights[kind] for kind in kinds)
    tasks, weighted = {}, 0.
    offload = backward and sum(c.values.numel() for c in sample.channels) >= settings["offload_native_values"]
    for kind in kinds:
        if stopped():
            raise StopRequested()
        context = torch.autograd.graph.save_on_cpu() if offload else nullcontext()
        with context:
            result = model(sample, kind=kind, seed=settings["seed"], epoch=epoch)
        if not bool(torch.isfinite(result.loss)):
            raise ValueError("Nonfinite "+kind+" objective: "+sample.identifier)
        coefficient = weights[kind]/total
        if backward:
            (result.loss*coefficient).backward()
        loss = float(result.loss.detach())
        tasks[kind] = dict(loss=loss, weight=coefficient, targets=len(result.predictions),
                          channel_mse={str(c): float(v.detach()) for c,v in result.channel_loss.items()})
        weighted += coefficient*loss
        del result
    return dict(loss=weighted if kinds else None, tasks=tasks, eligible=bool(kinds),
                eligibility={k:k in kinds for k in ("A", "B1")}, activation_offload=offload)


def update(model, dataset, rows, lookup, resolver, optimizer, state, settings, total, run=None,
           stopped=lambda: False, progress=None):
    if any(row["role"] != "TRAIN" for row in rows):
        raise ValueError("Held-out member cannot enter optimizer update")
    optimizer.zero_grad(set_to_none=True)
    records = []
    for row in rows:
        if stopped():
            raise StopRequested()
        before = time.perf_counter()
        sample = sample_from_dataset(dataset, lookup[row["id"]], resolver, settings["device"])
        if sample.role != "TRAIN":
            raise ValueError("Dataset/sample role mismatch")
        loaded = time.perf_counter()
        record = observation(model, sample, settings, state["epoch"], True, stopped)
        record.update(id=row["id"], sensor=row["sensor"], read_seconds=loaded-before,
                      compute_seconds=time.perf_counter()-loaded)
        records.append(record)
        del sample
        if progress:
            progress(dict(observations_in_batch=len(records), batch_observations=len(rows), current_member=row["id"]))
    eligible = sum(r["eligible"] for r in records)
    if not eligible:
        raise ValueError("Entire batch has no eligible task; no silent member removal")
    rate = learning_rate(state["cursor"], total, settings)
    norm = 0.
    if eligible:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(eligible)
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True))
        for group in optimizer.param_groups:
            group["lr"] = rate
        if stopped():
            raise StopRequested()
        sequence = state["spent_updates"]+1
        if run:
            append_event(Path(run)/"optimizer_attempts.jsonl",
                         dict(event="intent", sequence=sequence, cursor=state["cursor"]+1))
        state["spent_updates"] = sequence
        optimizer.step()
        if run:
            append_event(Path(run)/"optimizer_attempts.jsonl",
                         dict(event="applied", sequence=sequence, cursor=state["cursor"]+1))
    state["cursor"] += 1
    state["observations_seen"] += len(rows)
    return dict(cursor=state["cursor"], loss=sum(r["loss"] for r in records if r["eligible"])/eligible if eligible else None,
                eligible=eligible, batch_size=len(rows), gradient_norm=norm, lr=rate, members=records)


def evaluate(model, dataset, resolver, settings, stopped=lambda: False, progress=None):
    records = []
    if any(row["role"] != "VAL" for row in dataset.rows):
        raise ValueError("Evaluation requires frozen VAL only")
    fixed = dict(settings, seed=settings["validation_seed"])
    with torch.no_grad():
        model.eval()
        for index,row in enumerate(dataset.rows):
            if stopped():
                raise StopRequested()
            sample = sample_from_dataset(dataset, index, resolver, settings["device"])
            if sample.role != "VAL":
                raise ValueError("Dataset/sample role mismatch")
            record = observation(model, sample, fixed, 0)
            records.append(dict(id=row["id"], sensor=row["sensor"], **record))
            del sample
            if progress:
                progress(dict(validation_observations=index+1, validation_total=len(dataset.rows)))
    valid = [r["loss"] for r in records if r["eligible"]]
    by_task = {kind: [r["tasks"][kind]["loss"] for r in records if kind in r["tasks"]] for kind in ("A","B1")}
    by_sensor = {sensor: {} for sensor in sorted({r["sensor"] for r in records})}
    for sensor in by_sensor:
        for kind in by_task:
            values = [r["tasks"][kind]["loss"] for r in records if r["sensor"] == sensor and kind in r["tasks"]]
            by_sensor[sensor][kind] = dict(eligible=len(values), mse=sum(values)/len(values) if values else None)
    return dict(metric=sum(valid)/len(valid) if valid else None, observations=len(records),
                eligible=len(valid), tasks={k:dict(eligible=len(v), mse=sum(v)/len(v) if v else None) for k,v in by_task.items()},
                sensors=by_sensor, members=records)


def initialize(config):
    settings = config["training"]
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(settings["threads"])
    device = torch.device(settings["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = Pretraining(Config(**config["model"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    return model, optimizer


def preflight(config, datasets, resolver, binding, release):
    """Discard three bounded engineering updates; formal initialization remains fresh."""
    model, optimizer = initialize(config)
    rows = []
    for sensor in ("S2", "L8"):
        choices = sorted((sum(b["shape"][0]*b["shape"][1] for b in row["bands"]),row["id"],row)
                         for row in datasets["TRAIN"].rows if row["sensor"] == sensor)
        rows.append(choices[len(choices)//2][2])
    lookup = {r["id"]:i for i,r in enumerate(datasets["TRAIN"].rows)}
    parent = continue_from_checkpoint(model,optimizer,config,binding)
    state = dict(cursor=0, spent_updates=0, epoch=parent["sampling_start_epoch"] if parent else 0, observations_seen=0)
    total = len(EpochPlan(datasets["TRAIN"].rows, config["training"]["seed"],config["training"]["batch_size"]))
    update(model,datasets["TRAIN"],rows,lookup,resolver,optimizer,state,config["training"],total)
    checkpoint = release/"preflight_checkpoint.pt"
    save(checkpoint,model,optimizer,state,binding)
    update(model,datasets["TRAIN"],rows,lookup,resolver,optimizer,state,config["training"],total)
    expected = {n:p.detach().clone() for n,p in model.named_parameters()}
    expected_opt = optimizer.state_dict()
    restore_model, restore_opt = initialize(config)
    restored = {}
    resume(checkpoint,restore_model,restore_opt,restored,binding)
    update(restore_model,datasets["TRAIN"],rows,lookup,resolver,restore_opt,restored,config["training"],total)
    maximum = max(float((p.detach()-expected[n]).abs().max()) for n,p in restore_model.named_parameters())
    for name,p in restore_model.named_parameters():
        torch.testing.assert_close(p,expected[name],atol=2e-6,rtol=1e-5)
    for key,entry in restore_opt.state_dict()["state"].items():
        for name,value in entry.items():
            torch.testing.assert_close(value,expected_opt["state"][key][name],atol=2e-6,rtol=1e-4)
    assert restored == state
    write(release/"preflight.json",dict(status="PASS", identity=binding, parameter_max_abs=maximum,
                                      observations=[r["id"] for r in rows], engineering_updates=3,
                                      formal_weights_reused=False))
    print("FROZEN_RUNTIME_PREFLIGHT_PASS",maximum,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release",type=Path,required=True)
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    release = args.release.resolve()
    config = json.loads((release/"config.json").read_text(encoding="utf-8"))
    validate_config(config)
    manifest = json.loads((release/"source_manifest.json").read_text())
    for name, expected in manifest.items():
        if sha256(release/"runtime"/name) != expected:
            raise ValueError("Frozen source changed: "+name)
    if ROOT.resolve() != (release/"runtime").resolve():
        raise ValueError("Execute from the frozen runtime")
    os.environ["V3_H_R_GEOMETRY_CACHE"] = config["geometry_cache"]
    datasets, resolver = load_data()
    binding = binding_for(config,datasets,resolver)
    settings = config["training"]
    plan = EpochPlan(datasets["TRAIN"].rows,settings["seed"],settings["batch_size"])
    total = len(plan)*settings["epochs"]
    learning_rate(0,total,settings)
    if args.preflight:
        preflight(config,datasets,resolver,binding,release)
        return
    gate = json.loads((release/"preflight.json").read_text())
    if gate.get("status") != "PASS" or gate.get("identity") != binding:
        raise ValueError("Frozen runtime preflight does not match this run")
    run = release/"formal_run"
    if run.exists() and not args.resume:
        raise ValueError("New formal run requires a new directory")
    if args.resume and not (run/"last.pt").is_file():
        raise ValueError("Resume requires a committed checkpoint")
    run.mkdir(exist_ok=True)
    model, optimizer = initialize(config)
    parent = None if args.resume else continue_from_checkpoint(model,optimizer,config,binding)
    sampling_start = config.get("continuation",{}).get("start_epoch",0)
    state = dict(cursor=0, spent_updates=0, epoch=sampling_start, observations_seen=0, status="RUNNING",
                 sampling_start_epoch=sampling_start,parent=parent,last_validation_cursor=0,
                 total_batches=total, train_observations=len(datasets["TRAIN"].rows),
                 validation_observations=len(datasets["VAL"].rows), initial_parameter_sha=parameter_sha(model),
                 started_unix=time.time(), completed_update_seconds=0.)
    if args.resume:
        resume(run/"last.pt",model,optimizer,state,binding)
        # Refuse uncertain or already-finished budgets rather than silently replay.
        committed_spent = state["spent_updates"]
        reconcile_logs(run,state)
        if (state["spent_updates"] != committed_spent or state["status"] == "COMPLETE"
                or state.get("last_validation_cursor",0) == total):
            raise ValueError("Uncertain optimizer attempt or completed budget; explicit review required")
        state["status"] = "RUNNING"
    stop = threading.Event()
    for signum in (signal.SIGINT,signal.SIGTERM):
        signal.signal(signum,lambda *_:stop.set())
    stopped = lambda: stop.is_set() or (run/"STOP_REQUESTED").exists()
    lookup = {r["id"]:i for i,r in enumerate(datasets["TRAIN"].rows)}
    labels = dict(train_loss="训练A/B1观测等权损失",train_A="训练A",train_B1="训练B1",
                  val_loss="完整VAL综合损失",val_A="完整VAL A",val_B1="完整VAL B1",
                  learning_rate="学习率",gradient_norm="梯度范数",update_seconds="更新秒数")
    metrics = {k:dict(label=v,unit="",direction="min" if "loss" in k or k.endswith(("_A","_B1")) else "none",
                      role="loss" if "loss" in k or k.endswith(("_A","_B1")) else "other") for k,v in labels.items()}
    viewer = Path(state["viewer"]["directory"]) if args.resume else Path(config["viewer_root"])/("v3_h_r_"+uuid.uuid4().hex)
    with RunWriter(viewer,experiment=dict(id="encoder-v3-h-revised",name="Encoder V3-H-R"),
                   variant=dict(id="per-pixel-a-b1",name="逐像元A+B1正式训练",kind="baseline"),
                   metrics=metrics,charts=["train_loss","val_A","val_B1"],numbers=["learning_rate","gradient_norm","update_seconds"],
                   primary_metric="val_loss",config=dict(identity=digest(binding),training_directory=str(run),
                   budget_epochs=settings["epochs"]),resume=args.resume) as writer:
        state["viewer"] = dict(directory=str(viewer),run_id=writer.meta["run_id"])
        write(run/"identity.json",binding)
        write(run/"config.json",config)
        def progress(extra):
            write(run/"status.json",dict(state,**extra,last_heartbeat=time.time(),
                                        epoch_progress=plan.progress(state["cursor"]),
                                        elapsed_wall_seconds=time.time()-state["started_unix"]))
        if not args.resume:
            save(run/"initial.pt",model,optimizer,state,binding)
            save(run/"last.pt",model,optimizer,state,binding)
            append_event(run/"events.jsonl",dict(event="initialized",run_id=state["viewer"]["run_id"],
                                                parameter_sha=state["initial_parameter_sha"]))
        progress({})
        try:
            while True:
                if state["cursor"] and state["cursor"] % len(plan) == 0 and state.get("last_validation_cursor",0) != state["cursor"]:
                    state["status"] = "VALIDATING"
                    progress({})
                    validation_start = time.perf_counter()
                    with validation_transaction(model,optimizer,state):
                        validation = evaluate(model,datasets["VAL"],resolver,settings,stopped,progress)
                    state["validation_wall_seconds"] = state.get("validation_wall_seconds",0.)+time.perf_counter()-validation_start
                    write(run/("validation_epoch_"+str(state["cursor"]//len(plan))+".json"),validation)
                    write(run/"validation.json",validation)
                    values = {("val_loss" if task is None else "val_"+task): value for task,value in
                              [(None,validation["metric"]),*[(k,v["mse"]) for k,v in validation["tasks"].items()]] if value is not None}
                    writer.log(step=state["cursor"],epoch=plan.progress(state["cursor"]),values=values)
                    state.update(status="RUNNING",last_validation_cursor=state["cursor"],
                                 validation_summary={k:v for k,v in validation.items() if k!="members"})
                    if state.get("best") is None or validation["metric"] < state["best"]:
                        state["best"] = validation["metric"]
                        save(run/"best.pt",model,optimizer,state,binding)
                    save(run/"last.pt",model,optimizer,state,binding)
                    progress({})
                if state["cursor"] >= total:
                    break
                if stopped():
                    raise StopRequested()
                state["epoch"] = sampling_start+state["cursor"]//len(plan)
                rows = plan.batch(state["cursor"]+sampling_start*len(plan))
                started = time.perf_counter()
                record = update(model,datasets["TRAIN"],rows,lookup,resolver,optimizer,state,settings,total,
                                run,stopped,progress)
                record["seconds"] = time.perf_counter()-started
                state["completed_update_seconds"] += record["seconds"]
                append_event(run/"updates.jsonl",record)
                save(run/"last.pt",model,optimizer,state,binding)
                values = dict(learning_rate=record["lr"],gradient_norm=record["gradient_norm"],update_seconds=record["seconds"])
                if record["loss"] is not None:
                    values["train_loss"] = record["loss"]
                for task in ("A","B1"):
                    losses = [m["tasks"][task]["loss"] for m in record["members"] if task in m["tasks"]]
                    if losses:
                        values["train_"+task] = sum(losses)/len(losses)
                writer.log(step=state["cursor"],epoch=plan.progress(state["cursor"]),values=values)
                progress({})
                print("UPDATE",state["cursor"],"/",total,"loss",record["loss"],"seconds",record["seconds"],flush=True)
            state.update(status="COMPLETE",final_parameter_sha=parameter_sha(model))
            save(run/"last.pt",model,optimizer,state,binding)
            progress({})
            print("COMPLETE",state["viewer"]["run_id"],flush=True)
        except StopRequested:
            state["status"] = "STOPPED"
            save(run/"last.pt",model,optimizer,state,binding)
            progress({})
        except Exception as error:
            # Keep the last committed checkpoint; never save partially applied state.
            state.update(status="FAILED",error=repr(error))
            progress({})
            raise


if __name__ == "__main__":
    main()
