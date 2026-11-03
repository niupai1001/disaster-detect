"""Native whole-band prediction, balanced32 observations and resumable adaptive budget."""

from .execution_gate import require_training_authorization
if __name__ == "__main__": require_training_authorization()
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

if __name__=="__main__" and torch.cuda.is_available():torch.cuda.init()

from .data import ROOT,RELEASE,NativeSet,BalancedSampler,MaskSampler
from .model import make_model
from .check_precision import tile_values
from src.ssl.tvlp_writer import TvlpRun


def apply_flip(value,horizontal,vertical):
    value=torch.where(horizontal[:,None,None],value.flip(-1),value)
    return torch.where(vertical[:,None,None],value.flip(-2),value)


def evaluation(model,data,statistics,precision,engineering=False):
    model.eval();bands={};scene=[]
    with torch.no_grad():
        for sensor in ["S2","L8"]:
            count=len(data.records[sensor]);limit=min(count,2) if engineering else count
            targets=[0] if engineering else range(len(data.metadata[sensor]))
            for target in targets:
                info=data.metadata[sensor][target];name=info["name"]
                sums=np.zeros(4,np.float64)
                for start in range(0,limit,2):
                    ids=torch.arange(start,min(start+2,limit),device=data.device)
                    c=len(data.metadata[sensor]);hidden=torch.zeros(len(ids),c,device=data.device,dtype=torch.bool);hidden[:,target]=True
                    inputs=data.bands(sensor,ids,statistics,hidden);query=data.query(sensor,target,ids)
                    manager=torch.autocast("cuda",dtype=torch.bfloat16) if precision!="float32" else contextlib.nullcontext()
                    with manager:prediction,coverage=model.decoder(model.encode(inputs),query)
                    valid=data.valid[sensor][name][ids]&coverage;truth=data.value[sensor][name][ids]
                    difference=(prediction.float()-truth).double()
                    for j,index in enumerate(ids.tolist()):
                        values=difference[j][valid[j]]
                        record=dict(id=data.records[sensor][index]["observation_id"],group=data.records[sensor][index]["group"],acquisition=data.records[sensor][index]["acquisition_id"],sensor=sensor,band=name,count=len(values),sum_absolute=float(values.abs().sum()),sum_squared=float(values.square().sum()),sum_error=float(values.sum()))
                        sums+=np.array([record["count"],record["sum_absolute"],record["sum_squared"],record["sum_error"]]);scene.append(record)
                n=max(sums[0],1);rmse=float(np.sqrt(sums[2]/n))
                bands[name]=dict(sensor=sensor,count=int(sums[0]),mae=float(sums[1]/n),rmse=rmse,bias=float(sums[3]/n),normalized_rmse=rmse/statistics[name]["std"])
    score=float(np.mean([np.mean([r["normalized_rmse"] for r in bands.values() if r["sensor"]==sensor]) for sensor in ["S2","L8"]]))
    model.train()
    return dict(score=score,bands=bands,scenes=scene,selection="sensor-balanced mean per-band RMSE/TRAIN_std",native_target_grid=True)


def train(args):
    cfg=json.loads(Path(os.environ.get("OBSERVATION_CONFIG",Path(__file__).parent/"config.json")).read_text());settings=cfg["training"]
    contract=json.loads((RELEASE/"release.json").read_text())
    if contract["status"]!="ACCEPTED_LOCAL_NATIVE_FOR_MASKED":raise ValueError("native release not accepted")
    statistics=json.loads((RELEASE/"statistics.json").read_text())["bands"]
    masking=json.loads((RELEASE/"masking.json").read_text())
    precision=settings["precision"]
    if precision!="float32":
        check=json.loads((ROOT/"observation_model"/settings["precision_acceptance"]).read_text())
        if check["status"]!="passed":raise ValueError("BF16 actual comparison not passed")
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    last=out/"last.pt"
    if last.exists() and not args.resume:raise FileExistsError("existing run requires explicit resume")
    torch.manual_seed(0);random.seed(0);np.random.seed(0)
    coordinate_mode=getattr(args,"coordinate_mode","legacy_local")
    model=make_model(args.variant,coordinate_mode=coordinate_mode).cuda();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=settings["lr"],weight_decay=settings["weight_decay"])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=.5,patience=settings["lr_patience"]-1,threshold=settings["relative_improvement"],threshold_mode="rel",min_lr=settings["min_lr"])
    data=NativeSet("train");val=NativeSet("val")
    sampler=BalancedSampler({s:len(r) for s,r in data.records.items()})
    masks=MaskSampler(masking);target_rng=np.random.default_rng(0);augment=torch.Generator(device="cuda").manual_seed(0)
    step=0;best=float("inf");anchor=float("inf");stale=0;exposure=0;stop_reason="update_limit";history=[]
    if args.resume:
        state=torch.load(args.resume,map_location="cuda",weights_only=False)
        model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"])
        sampler.load_state_dict(state["sampler"]);masks.generator.bit_generator.state=state["mask_rng"];target_rng.bit_generator.state=state["target_rng"]
        augment.set_state(state["augmentation_rng"].cpu());torch.set_rng_state(state["torch_rng"].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]]);random.setstate(state["python_rng"]);np.random.set_state(state["numpy_rng"])
        step,best,anchor,stale,exposure,history=[state[k] for k in ["step","best","anchor","stale","exposure","history"]]
    else:torch.save(model.state_dict(),out/"initial.pt")
    maximum=100 if args.engineering else cfg["budgets"]["masked"]["max"]
    if not args.engineering:maximum=cfg.get("repair_max_updates",{}).get("masked_"+args.variant,maximum)
    minimum=0 if args.engineering else cfg["budgets"]["masked"]["min"]
    interval=50 if args.engineering else settings["validation_interval"]
    resolved=dict(config=cfg,arguments=vars(args),physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),variant=args.variant,release=str(RELEASE),release_hashes=contract["hashes"],statistics=statistics,engineering=args.engineering,max_updates=maximum,min_updates=minimum,parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),cnn_capacity=getattr(model,"capacity_match",None),observations_per_update=32,loss="equal sensor/target weighting; unbiased uniform16of400 tile estimator of full valid pixel Huber",precision=precision,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},members={s:[r["observation_id"] for r in data.records[s]] for s in data.records})
    (out/"resolved_config.json").write_text(json.dumps(resolved,indent=2))
    tv=TvlpRun(args.tv_root,project="conditional-observation",name=args.tag,run_id=args.tag,resume=bool(args.resume),seed=0,config=resolved,metric_definitions={"train/loss":{"direction":"min"},"val/score":{"direction":"min"},"train/lr":{"direction":"none"}})
    t0=time.perf_counter();start_exposure=exposure;torch.cuda.reset_peak_memory_stats()
    def save(path):
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),sampler=sampler.state_dict(),mask_rng=masks.generator.bit_generator.state,target_rng=target_rng.bit_generator.state,augmentation_rng=augment.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),step=step,best=best,anchor=anchor,stale=stale,exposure=exposure,history=history,statistics=statistics,resolved_config=resolved,stop_reason=stop_reason)
        temporary=path.with_suffix(".tmp");torch.save(state,temporary);temporary.replace(path)
    try:
        while step<maximum:
            model.train();optimizer.zero_grad(set_to_none=True)
            if step<settings["warmup_updates"]:
                for group in optimizer.param_groups:group["lr"]=settings["lr"]*(step+1)/settings["warmup_updates"]
            total=0.;target_log={};pixel_count=0;by_sensor={}
            for sensor in ["S2","L8"]:
                indices=torch.tensor(sampler.take(sensor,16),device="cuda")
                target=int(target_rng.integers(len(data.metadata[sensor])));target_log[sensor]=data.metadata[sensor][target]["name"]
                hidden=masks.sample(sensor,target,16,"cuda")
                tiles=torch.randperm(400,device="cuda",generator=augment)[:settings["train_target_tiles"]]
                horizontal=torch.rand(16,device="cuda",generator=augment)<.5;vertical=torch.rand(16,device="cuda",generator=augment)<.5
                sensor_total=0.
                for start in range(0,16,settings["microbatch"]):
                    stop=min(start+settings["microbatch"],16);ids=indices[start:stop]
                    hh,vv=horizontal[start:stop],vertical[start:stop]
                    inputs=data.bands(sensor,ids,statistics,hidden[start:stop]);query=data.query(sensor,target,ids)
                    for band in inputs:
                        band.value=apply_flip(band.value,hh,vv);band.valid=apply_flip(band.valid,hh,vv)
                        band.phase_xy_m=torch.stack([torch.where(hh,band.native_gsd_m-band.phase_xy_m[:,0],band.phase_xy_m[:,0]),torch.where(vv,band.native_gsd_m-band.phase_xy_m[:,1],band.phase_xy_m[:,1])],1)
                    query.phase_xy_m=inputs[target].phase_xy_m
                    name=query.band;truth=apply_flip(data.value[sensor][name][ids],hh,vv);truth_valid=apply_flip(data.valid[sensor][name][ids],hh,vv)
                    selected_truth=tile_values(truth,tiles,query.grid_gsd_m);selected_valid=tile_values(truth_valid,tiles,query.grid_gsd_m)
                    manager=torch.autocast("cuda",dtype=torch.bfloat16) if precision!="float32" else contextlib.nullcontext()
                    with manager:prediction,coverage=model.decoder(model.encode(inputs),query,tiles)
                    valid=selected_valid&coverage
                    per_pixel=F.huber_loss(prediction.float(),selected_truth,reduction="none",delta=cfg["huber_delta"])
                    denominator=truth_valid.sum((1,2)).clamp_min(1)
                    per_observation=(per_pixel*valid).sum((1,2))*(400/len(tiles))/denominator
                    loss=per_observation.sum()/32
                    if not torch.isfinite(loss):raise FloatingPointError("nonfinite native training loss")
                    loss.backward();total+=float(loss.detach());sensor_total+=float(loss.detach());pixel_count+=int(valid.sum())
                by_sensor[sensor+"/loss"]=sensor_total*2;by_sensor[sensor+"/target_index"]=target
            grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],settings["clip"],error_if_nonfinite=True)
            optimizer.step();step+=1;exposure+=32
            elapsed=time.perf_counter()-t0
            tv.log("train",dict(loss=total,lr=optimizer.param_groups[0]["lr"],exposure=exposure,target_valid_pixels=pixel_count,gradient_norm=float(grad),locations_per_second=(exposure-start_exposure)/max(elapsed,1e-9),gpu_allocated_mib=torch.cuda.memory_allocated()/1024**2,gpu_peak_mib=torch.cuda.max_memory_allocated()/1024**2,**by_sensor),global_step=step,wall_time_sec=elapsed);tv.heartbeat(step)
            if step%10==0 or step==1:print(json.dumps(dict(tag=args.tag,step=step,loss=total,lr=optimizer.param_groups[0]["lr"],seconds=elapsed,targets=target_log)),flush=True)
            if step%interval==0:
                result=evaluation(model,val,statistics,precision,args.engineering);score=result["score"]
                if not np.isfinite(score):raise FloatingPointError("nonfinite validation score")
                improved=score<best
                if score<anchor*(1-settings["relative_improvement"]):anchor=score;stale=0
                else:stale+=1
                best=min(best,score);history.append(dict(step=step,score=score,lr=optimizer.param_groups[0]["lr"],exposure=exposure))
                if step>=settings["warmup_updates"]:scheduler.step(score)
                (out/f"validation_{step}.json").write_text(json.dumps(result,indent=2))
                tv.log("val",dict(score=score,**{k+"/rmse":v["rmse"] for k,v in result["bands"].items()}),global_step=step)
                save(last)
                if improved:save(out/"best_unlabeled.pt")
                if not args.engineering and step>=minimum and stale>=settings["stop_patience"] and optimizer.param_groups[0]["lr"]<=settings["lr"]*settings["stop_lr_fraction"]:
                    stop_reason="validation_plateau";break
            if step%100==0:save(last)
            if step%settings["milestone_interval"]==0:torch.save(dict(model=model.state_dict(),step=step,statistics=statistics),out/f"step{step}.pt")
            if args.stop_after and step>=args.stop_after:
                stop_reason="engineering_pause";save(last);tv.finish("cancelled","engineering pause for disk resume");return
        save(last);tv.finish()
        result=dict(status="engineering_complete" if args.engineering else "training_complete_pending_evaluation",step=step,best_score=best,last_score=history[-1]["score"] if history else None,stop_reason=stop_reason,exposure=exposure,sampler_min_exposure={s:min(v) for s,v in sampler.exposure.items()},seconds=time.perf_counter()-t0,peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2)
        (out/"training_result.json").write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except BaseException as exc:
        tv.finish("failed",type(exc).__name__+": "+str(exc));raise


def main():
    p=argparse.ArgumentParser();p.add_argument("--variant",choices=["transformer","cnn","random"],default="transformer");p.add_argument("--tag",required=True);p.add_argument("--out",required=True);p.add_argument("--tv-root",required=True);p.add_argument("--resume");p.add_argument("--engineering",action="store_true");p.add_argument("--stop-after",type=int,default=0)
    p.add_argument("--coordinate-mode",choices=["legacy_local","explicit_global"],default="legacy_local")
    train(p.parse_args())


if __name__=="__main__":main()
