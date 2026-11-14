"""Run only the registered short A/B1 comparisons with the frozen executor."""
from dataclasses import asdict
import csv
from hashlib import sha256
import json
import time
from pathlib import Path

import torch

from candidate_gate import config_device, options
from evaluation import evaluate
from run_v3_fr import hash_file, mask_sha, read, write


def compare(args):
    from encoder_v3.film_resnet.model import FRPretraining
    from encoder_v3.hierarchical_revised import train as frozen_train
    from encoder_v3.hierarchical_revised.tasks import task_masks
    from encoder_v3.latent_response.checkpoint import digest, parameter_sha
    from encoder_v3.patch_latent.batching import ProportionalEpochPlan
    from encoder_v3.patch_latent.parallel import ParallelBatch
    from encoder_v3.shared_mae.data import load_data

    assert read(args.output/"s5_capacity_fit/fit_gate.json")["status"]=="PASS"
    cfg,settings,device = config_device(args)
    frozen = read(args.output/"baseline_identity.json")
    policy = read(args.output/"acceptance_policy.json")
    data,resolver = load_data()
    train,val = data["TRAIN"],data["VAL"]
    other,other_resolver = load_data()
    settings = settings | {"offload_native_values":float("inf"),"spatial_cuda_graphs":False}
    plan = ProportionalEpochPlan(train.rows,17,settings["batch_size"])
    assert len(plan)==233
    order = [[row["id"] for row in plan.batch(i)] for i in range(len(plan))]
    panel = read(args.output/"data_panel.json")
    root = args.output/("s6_component_trials" if args.seed==17 else "s7_seed_confirmation")
    root.mkdir(exist_ok=True)
    candidate_root = Path(__file__).resolve().parents[1]
    files = {p.relative_to(candidate_root).as_posix():hash_file(p)
             for folder in ("scripts","src") for p in (candidate_root/folder).rglob("*.py")}
    parents = dict(B0=None,BP="B0",N0="BP",N1="N0",N2="N1",C="N2",PD="BP")
    baseline_obs = args.output/"s1_baseline/metrics_by_observation.csv"
    with baseline_obs.open(encoding="utf-8-sig",newline="") as stream:
        val_masks = {(row["observation"],row["task"]):(row["target_mask_sha256"],row["visible_mask_sha256"],
                     int(row["targets"])) for row in csv.DictReader(stream)}
    if args.seed !=17:
        selected = read(args.output/"s6_component_trials/screen.json")
        assert selected["status"]=="CANDIDATE" and set(args.arms)=={selected["candidate"],selected["baseline"]}
    else:
        assert args.arms==["B0","BP","N0","N1","N2","C"]
    original_observation = frozen_train.observation

    def with_mask_identity(model,sample,settings,epoch,backward=False,stopped=lambda:False):
        result = original_observation(model,sample,settings,epoch,backward,stopped)
        for kind, metrics in result["tasks"].items():
            keep,target = task_masks(sample,model.config,kind,17,epoch)
            assert sum(int(mask.sum()) for mask in target)==metrics["targets"]
            metrics.update(target_hash=mask_sha(target),visible_hash=mask_sha(keep))
        return result

    frozen_train.observation = with_mask_identity
    try:
        for arm in args.arms:
            path = root/(arm if args.seed==17 else f"{arm}_seed{args.seed}")
            path.mkdir(exist_ok=False)
            start = time.perf_counter()
            model = FRPretraining(cfg,options(arm),seed=args.seed)
            if args.seed==17 and parents[arm]:
                parent = root/parents[arm]
                assert read(parent/"status.json")["status"]=="COMPLETE"
                other_binding = read(parent/"identity.json")
                old = other_binding["initial_parameters"]
            else:
                old = {}
            names = {key:digest([list(tensor.shape),str(tensor.dtype),
                                sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()])
                     for key,tensor in model.state_dict().items()}
            common = {key:names[key] for key in old.keys() & names.keys()
                      if not (arm=="N0" and key.startswith("encoder.spectral."))}
            assert all(old[key]==value for key,value in common.items())
            binding = dict(experiment_id=arm,experiment_family="V3-FR",parent_id=parents[arm],
                changed_factor={"B0":"reference eager","BP":"physical Q/K positioning","N0":"ordered frontend and fusion",
                "N1":"two residual blocks","N2":"FiLM modulation","C":"D1 terminal","PD":"D1 terminal"}[arm],
                design_version="1.0",design_sha256=frozen["design_sha256"],plan_sha256=frozen["plan_sha256"],
                git_commit=frozen["git_commit"],dirty_diff_hash=frozen["dirty_diff_hash"],source_snapshot_hash=digest(files),
                historical_snapshot_hash=frozen["source_snapshot_hash"],candidate_files=files,
                resolved_config_hash=digest(asdict(options(arm))),environment_lock_hash=hash_file(args.output/"environment.json"),
                initial_parameter_hash=parameter_sha(model),initial_parameters=names,common_parameter_hash=digest(common),
                dataset_manifest_hash=frozen["dataset_manifest_hash"],train_members_hash=frozen["train_member_hash"],
                val_members_hash=frozen["val_member_hash"],normalization_hash=digest(frozen["data"]),
                metadata_catalog_hash=resolver.catalog_sha,geometry_version="frozen-area-metric-plus-q2-v1",
                mask_protocol_hash=frozen["mask_protocol_hash"],schedule_hash=digest(order),
                init_seed=args.seed,mask_seed=17,data_order_seed=17,precision="float32",executor="dual_eager",
                optimizer="AdamW",scheduler="historical 6990 prefix",effective_batch=32,options=asdict(options(arm)),
                trainable_modules=[name for name,p in model.named_parameters() if p.requires_grad])
            write(path/"identity.json",binding)
            model = model.to(device)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                    lr=settings["lr"],weight_decay=settings["weight_decay"])
            parallel = ParallelBatch(model,train,resolver,settings,settings["devices"],
                                     secondary_data=(other["TRAIN"],other_resolver))
            state = dict(cursor=0,epoch=0,spent_updates=0,observations_seen=0)
            checkpoints, validations, peak = {}, [], dict.fromkeys(settings["devices"],0)
            training_masks = {}
            views, pixels = dict(A=0,B1=0),dict(A=0,B1=0)
            save_seconds = update_seconds = 0.
            status = "RUNNING"
            try:
                with (path/"updates.jsonl").open("x",encoding="utf-8",buffering=1) as stream:
                    for cursor in range(len(plan)+1):
                        if cursor%25==0 or cursor in (42,116,233):
                            before = time.perf_counter()
                            ckpt = path/f"update_{cursor:03d}.pt"
                            frozen_train.save(ckpt,model,opt,state,binding)
                            checkpoints[cursor] = hash_file(ckpt)
                            save_seconds += time.perf_counter()-before
                        if cursor in (0,42,116,233):
                            val_node = evaluate(model,val,resolver,settings,path,arm,args.seed,cursor,
                                                set(panel["fixed_val"]) if cursor in (0,233) else set())
                            with (path/f"val_{cursor:03d}/metrics_by_observation.csv").open(encoding="utf-8-sig") as f:
                                actual = {(r["observation_id"],r["task"]):(r["target_mask_hash"],r["visible_mask_hash"],
                                          int(r["target_count"])) for r in csv.DictReader(f)}
                            assert actual==val_masks
                            validations.append(val_node)
                            print("S6" if args.seed==17 else "S7",arm,args.seed,cursor,
                                  {k:round(v["mse"],6) for k,v in val_node["groups"].items() if k.endswith("/all")},flush=True)
                        if cursor==len(plan):
                            break
                        model.train()
                        for dev in peak:
                            torch.cuda.reset_peak_memory_stats(dev)
                        before = time.perf_counter()
                        result = frozen_train.update(model,train,plan.batch(cursor),{},resolver,opt,state,settings,6990,
                                                    run=path,backward_batch=parallel.backward,
                                                    stopped=lambda:(args.output/"STOP_REQUESTED").exists())
                        result["seconds"] = time.perf_counter()-before
                        update_seconds += result["seconds"]
                        assert [row["id"] for row in result["members"]]==order[cursor]
                        for row in result["members"]:
                            for task,v in row["tasks"].items():
                                key = (row["id"],task)
                                assert key not in training_masks
                                training_masks[key] = (v["target_hash"],v["visible_hash"],v["targets"])
                                views[task] += 1
                                pixels[task] += v["targets"]
                        for dev in peak:
                            peak[dev] = max(peak[dev],torch.cuda.max_memory_allocated(dev))
                        stream.write(json.dumps(result,allow_nan=False)+"\n")
                        if (cursor+1)%16==0:
                            write(path/"progress.json",dict(cursor=cursor+1,views=views,wall_seconds=time.perf_counter()-start))
                            print("S6",arm,args.seed,cursor+1,"updates",flush=True)
                assert views=={"A":7442,"B1":7442} and state["spent_updates"]==233
                mask_digest = digest(sorted((list(key),value) for key,value in training_masks.items()))
                base_dir = args.output/"s6_component_trials/B0"
                if path!=base_dir:
                    assert mask_digest==read(base_dir/"status.json")["training_masks_hash"]
                status = "COMPLETE"
                write(path/"status.json",dict(status=status,**state,arm=arm,seed=args.seed,checkpoint_hashes=checkpoints,
                      checkpoint_hash=checkpoints[233],training_masks_hash=mask_digest,actual_task_views=views,
                      target_pixel_counts=pixels,validation=validations,peak_allocated_bytes=peak,
                      parameters=sum(p.numel() for p in model.parameters()),
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      wall_seconds=time.perf_counter()-start,update_seconds=update_seconds,save_seconds=save_seconds,
                      replayed_updates=0))
            except Exception as error:
                write(path/"status.json",dict(status="STOPPED" if (args.output/"STOP_REQUESTED").exists() else "BLOCKED",
                      **state,arm=arm,seed=args.seed,actual_task_views=views,checkpoint_hashes=checkpoints,error=repr(error),
                      wall_seconds=time.perf_counter()-start))
                raise
            finally:
                parallel.close()
    finally:
        frozen_train.observation = original_observation
