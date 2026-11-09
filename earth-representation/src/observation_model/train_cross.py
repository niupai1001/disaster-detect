"""Matched masked continuation versus real bidirectional cross30 supervision."""

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

if __name__=="__main__":torch.cuda.init()

from .data import ROOT,RELEASE,NativeSet,BalancedSampler,MaskSampler
from .paired import NativePairs,CROSS_RELEASE
from .model import make_model,TargetQuery,checkpoint_coordinate_mode
from .train_masked import apply_flip
from .check_precision import tile_values
from src.ssl.tvlp_writer import TvlpRun


def flipped_inputs(data,sensor,ids,statistics,hidden,hh,vv):
    bands=data.bands(sensor,ids,statistics,hidden)
    for band in bands:
        band.value=apply_flip(band.value,hh,vv);band.valid=apply_flip(band.valid,hh,vv)
        phase=band.phase_xy_m
        band.phase_xy_m=torch.stack([torch.where(hh,band.native_gsd_m-phase[:,0],phase[:,0]),torch.where(vv,band.native_gsd_m-phase[:,1],phase[:,1])],1)
    return bands


def cross_query(data,target,index,phase):
    meta=data.metadata[target][index]
    return TargetQuery(target,meta["name"],meta["wavelength_nm"],meta["bandwidth_nm"],meta["native_gsd_m"],30,phase)


def evaluate_cross(model,pairs,statistics,precision):
    model.eval();metrics={};scenes=[]
    with torch.no_grad():
        for source,target in [("S2","L8"),("L8","S2")]:
            aggregate={b["name"]:np.zeros(4) for b in pairs.data.metadata[target]}
            for i,row in enumerate(pairs.records):
                ids=pairs.indices[source][i:i+1];inputs=pairs.data.bands(source,ids,statistics)
                manager=torch.autocast("cuda",dtype=torch.bfloat16) if precision!="float32" else contextlib.nullcontext()
                with manager:
                    representation=model.encode(inputs)
                    for target_index,band in enumerate(pairs.data.metadata[target]):
                        query=cross_query(pairs.data,target,target_index,pairs.phase[i:i+1])
                        prediction,coverage=model.decoder(representation,query)
                        valid=pairs.valid[target][i:i+1,target_index]&coverage
                        difference=(prediction.float()-pairs.target[target][i:i+1,target_index]).double()[valid]
                        totals=np.array([len(difference),float(difference.abs().sum()),float(difference.square().sum()),float(difference.sum())])
                        aggregate[band["name"]]+=totals
                        scenes.append(dict(site_id=row["site_id"],group=row["group"],direction=source+"_to_"+target,band=band["name"],count=int(totals[0]),sum_absolute=totals[1],sum_squared=totals[2],sum_error=totals[3]))
            direction={}
            for name,totals in aggregate.items():
                n=max(totals[0],1);rmse=float(np.sqrt(totals[2]/n));direction[name]=dict(count=int(totals[0]),mae=float(totals[1]/n),rmse=rmse,bias=float(totals[3]/n),normalized_rmse=rmse/statistics[name]["std"])
            metrics[source+"_to_"+target]=direction
    score=float(np.mean([np.mean([b["normalized_rmse"] for b in direction.values()]) for direction in metrics.values()]))
    model.train();return dict(score=score,directions=metrics,scenes=scenes,pairs=len(pairs.records),boundary="exploratory on qualified real native targets; no HLS proxy truth, no Landsat10m truth")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=["masked_control","cross30"],required=True);parser.add_argument("--initial",required=True);parser.add_argument("--out",required=True);parser.add_argument("--tag",required=True);parser.add_argument("--tv-root",required=True);parser.add_argument("--resume");parser.add_argument("--engineering",action="store_true");parser.add_argument("--stop-after",type=int,default=0)
    parser.add_argument("--observation-mode",choices=["none","fixed"]);parser.add_argument("--gate");args=parser.parse_args()
    config=json.loads(Path(os.environ.get("OBSERVATION_CONFIG",Path(__file__).parent/"config.json")).read_text());settings=config["training"]
    if not config["cross_execution"]["required_by_latest_user"]:raise ValueError("latest cross-source execution policy missing")
    acceptance=json.loads((CROSS_RELEASE/"release.json").read_text())
    if acceptance["status"]!="ACCEPTED_EXPLORATORY_REAL_CROSS30":raise ValueError("qualified real native pairs missing")
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);last=out/"last.pt"
    if last.exists() and not args.resume:raise FileExistsError("existing cross run requires resume")
    torch.manual_seed(0);np.random.seed(0);random.seed(0)
    initial=torch.load(args.initial,map_location="cpu",weights_only=False)
    initial_hash=hashlib.sha256(Path(args.initial).read_bytes()).hexdigest()
    if args.observation_mode is not None:
        if args.mode!="cross30":raise ValueError("observation comparison continues the same cross30 objective in both arms")
        if not args.engineering:
            gate=json.loads(Path(args.gate).read_text()) if args.gate else {}
            if gate.get("status")!="PASS_P3_BIDIRECTIONAL" or gate.get("initial_sha256")!=initial_hash:
                raise ValueError("formal observation comparison requires a passed bidirectional gate and its fixed common checkpoint")
    model=make_model(coordinate_mode=checkpoint_coordinate_mode(initial)).cuda();model.load_state_dict(initial["model"])
    statistics=initial["statistics"];precision=settings["precision"]
    if statistics!=json.loads((RELEASE/"statistics.json").read_text())["bands"]:raise ValueError("initial checkpoint uses another data release's statistics")
    data=NativeSet("train");pairs=NativePairs(data);validation=NativePairs(NativeSet("val"))
    pools={s:[i for i in range(len(data.records[s])) if i not in set(pairs.indices[s].tolist())] for s in ["S2","L8"]}
    sampler=BalancedSampler({s:len(v) for s,v in pools.items()});pair_sampler=BalancedSampler({"pairs":len(pairs.records)})
    masks=MaskSampler(json.loads((RELEASE/"masking.json").read_text()));rng=np.random.default_rng(0);augment=torch.Generator(device="cuda").manual_seed(0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings["lr"],weight_decay=settings["weight_decay"])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=.5,patience=settings["lr_patience"]-1,threshold=settings["relative_improvement"],threshold_mode="rel",min_lr=settings["min_lr"])
    step=0;best=float("inf");anchor=float("inf");stale=0;history=[];stop_reason="update_limit"
    if args.resume:
        state=torch.load(args.resume,map_location="cuda",weights_only=False);model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"])
        if state["resolved_config"]["mode"]!=args.mode or state["resolved_config"]["arguments"].get("observation_mode")!=args.observation_mode:raise ValueError("resume cannot change objective or observation arm")
        sampler.load_state_dict(state["sampler"]);pair_sampler.load_state_dict(state["pair_sampler"]);rng.bit_generator.state=state["target_rng"];masks.generator.bit_generator.state=state["mask_rng"]
        augment.set_state(state["augmentation_rng"].cpu());torch.set_rng_state(state["torch_rng"].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]]);np.random.set_state(state["numpy_rng"]);random.setstate(state["python_rng"])
        step,best,anchor,stale,history=[state[k] for k in ["step","best","anchor","stale","history"]]
    else:torch.save(model.state_dict(),out/"initial.pt")
    budget="observation" if args.observation_mode is not None else "cross30"
    maximum=4 if args.engineering else config["budgets"][budget]["max"];minimum=0 if args.engineering else config["budgets"][budget]["min"];interval=2 if args.engineering else settings["validation_interval"]
    if not args.engineering and args.observation_mode is None:maximum=config.get("repair_max_updates",{}).get("cross_"+args.mode,maximum)
    resolved=dict(arguments=vars(args),config=config,mode=args.mode,initial_sha256=initial_hash,initial_step=initial["step"],initial_kind="weights only; new optimizer and stage schedule",pair_release=acceptance,statistics=statistics,observations_per_update=32,pair_slots_per_update=8,distinct_train_pairs=len(pairs.records),encoder_views_per_update=48,control="same observations/masks/views; masked loss only; cross prediction logged without gradient",cross_loss="masked native reconstruction + equal-weight true bidirectional cross30 Huber",selection="mean direction-balanced per-band RMSE/TRAIN_std",source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")})
    if args.observation_mode is not None:
        resolved["observation"]=dict(mode=args.observation_mode,weight=1,source="full native S2",estimate_grid="10m nominal grid, first center5m inside1200m window",operator="fixed7x7 Gaussian sigma10m plus actual physical area sampling to real Landsat30m grid",not_calibrated_OLI_PSF=True,encoder_views_reused=True,selection="same direct cross30 validation score in both arms")
        resolved["control"]="same masked+cross30 objectives, observations, masks and augmentation schedule; only fixed arm adds observation loss"
    (out/"resolved_config.json").write_text(json.dumps(resolved,indent=2))
    tv=TvlpRun(args.tv_root,project="conditional-observation",name=args.tag,run_id=args.tag,resume=bool(args.resume),seed=0,config=resolved,metric_definitions={"train/loss":{"direction":"min"},"val/score":{"direction":"min"}})
    start=time.perf_counter();start_step=step;torch.cuda.reset_peak_memory_stats()
    def save(path):
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),sampler=sampler.state_dict(),pair_sampler=pair_sampler.state_dict(),target_rng=rng.bit_generator.state,mask_rng=masks.generator.bit_generator.state,augmentation_rng=augment.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),step=step,best=best,anchor=anchor,stale=stale,history=history,stop_reason=stop_reason,statistics=statistics,resolved_config=resolved,exposure=32*step)
        temporary=path.with_suffix(".tmp");torch.save(state,temporary);temporary.replace(path)
    try:
        while step<maximum:
            model.train();optimizer.zero_grad(set_to_none=True)
            if step<settings["warmup_updates"]:
                for group in optimizer.param_groups:group["lr"]=settings["lr"]*(step+1)/settings["warmup_updates"]
            pair_ids=torch.tensor(pair_sampler.take("pairs",8),device="cuda")
            ph=torch.rand(8,device="cuda",generator=augment)<.5;pv=torch.rand(8,device="cuda",generator=augment)<.5
            indices={};flips={};masked_total=0.;cross_total=0.;observation_total=0.;direction_losses={};masked_pixels=0;cross_pixels=0;observation_pixels=0
            for sensor in ["S2","L8"]:
                single=torch.tensor([pools[sensor][i] for i in sampler.take(sensor,8)],device="cuda")
                indices[sensor]=torch.cat([single,pairs.indices[sensor][pair_ids]])
                hh=torch.cat([torch.rand(8,device="cuda",generator=augment)<.5,ph]);vv=torch.cat([torch.rand(8,device="cuda",generator=augment)<.5,pv]);flips[sensor]=(hh,vv)
                target=int(rng.integers(len(data.metadata[sensor])));hidden=masks.sample(sensor,target,16,"cuda");tiles=torch.randperm(400,device="cuda",generator=augment)[:settings["train_target_tiles"]]
                for begin in range(0,16,settings["microbatch"]):
                    end=min(begin+settings["microbatch"],16);ids=indices[sensor][begin:end];h,v=hh[begin:end],vv[begin:end]
                    inputs=flipped_inputs(data,sensor,ids,statistics,hidden[begin:end],h,v);query=data.query(sensor,target,ids);query.phase_xy_m=inputs[target].phase_xy_m
                    truth=apply_flip(data.value[sensor][query.band][ids],h,v);valid=apply_flip(data.valid[sensor][query.band][ids],h,v)
                    manager=torch.autocast("cuda",dtype=torch.bfloat16) if precision!="float32" else contextlib.nullcontext()
                    with manager:prediction,coverage=model.decoder(model.encode(inputs),query,tiles)
                    keep=tile_values(valid,tiles,query.grid_gsd_m)&coverage
                    masked_pixels+=int(keep.sum())
                    error=F.huber_loss(prediction.float(),tile_values(truth,tiles,query.grid_gsd_m),reduction="none",delta=.02)
                    loss=((error*keep).sum((1,2))*(400/len(tiles))/valid.sum((1,2)).clamp_min(1)).sum()/32
                    loss.backward();masked_total+=float(loss.detach())
            for source,target in [("S2","L8"),("L8","S2")]:
                target_index=int(rng.integers(len(data.metadata[target])));direction_loss=0.
                for begin in range(0,8,settings["microbatch"]):
                    end=min(begin+settings["microbatch"],8);selected=pair_ids[begin:end];h,v=ph[begin:end],pv[begin:end]
                    ids=pairs.indices[source][selected]
                    inputs=flipped_inputs(data,source,ids,statistics,None,h,v)
                    phase=pairs.phase[selected];phase=torch.stack([torch.where(h,30-phase[:,0],phase[:,0]),torch.where(v,30-phase[:,1],phase[:,1])],1)
                    query=cross_query(data,target,target_index,phase)
                    truth=apply_flip(pairs.target[target][selected,target_index],h,v);valid=apply_flip(pairs.valid[target][selected,target_index],h,v)
                    gradient=contextlib.nullcontext() if args.mode=="cross30" else torch.no_grad()
                    manager=torch.autocast("cuda",dtype=torch.bfloat16) if precision!="float32" else contextlib.nullcontext()
                    with gradient:
                        with manager:
                            representation=model.encode(inputs);prediction,coverage=model.decoder(representation,query)
                        error=F.huber_loss(prediction.float(),truth,reduction="none",delta=.02)
                        loss=((error*valid*coverage).sum((1,2))/valid.sum((1,2)).clamp_min(1)).sum()/16
                        if not torch.isfinite(loss):raise FloatingPointError("nonfinite real cross-source loss")
                        cross_loss_value=float(loss.detach());cross_pixels+=int((valid&coverage).sum())
                        if args.observation_mode is not None and source=="S2":
                            from .observation_constraint import observe_fine_prediction
                            observe_grad=contextlib.nullcontext() if args.observation_mode=="fixed" else torch.no_grad()
                            observe_amp=torch.autocast("cuda",dtype=torch.bfloat16) if precision!="float32" else contextlib.nullcontext()
                            with observe_grad:
                                with observe_amp:_,_,observed,operator_coverage=observe_fine_prediction(model,representation,query)
                                keep=valid&(operator_coverage>=.8)
                                observation_pixels+=int(keep.sum())
                                observation_loss=((F.huber_loss(observed,truth,reduction="none",delta=.02)*keep).sum((1,2))/keep.sum((1,2)).clamp_min(1)).sum()/8
                                if not torch.isfinite(observation_loss):raise FloatingPointError("nonfinite observation consistency loss")
                                observation_total+=float(observation_loss.detach())
                                if args.observation_mode=="fixed":loss=loss+observation_loss
                        if args.mode=="cross30":loss.backward()
                    direction_loss+=cross_loss_value
                cross_total+=direction_loss;direction_losses[source+"_to_"+target]=direction_loss*2
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings["clip"],error_if_nonfinite=True);optimizer.step();step+=1
            total=masked_total+(cross_total if args.mode=="cross30" else 0)+(observation_total if args.observation_mode=="fixed" else 0)
            tv.log("train",dict(loss=total,masked_loss=masked_total,cross_loss=cross_total,lr=optimizer.param_groups[0]["lr"],observation_exposure=32*step,encoder_view_exposure=48*step,pair_slot_exposure=8*step,distinct_train_pair_locations=len(pairs.records),unique_pairs_this_update=len(pair_ids.unique()),unique_observations_this_update=sum(len(ids.unique()) for ids in indices.values()),gradient_norm=float(norm),gpu_peak_mib=torch.cuda.max_memory_allocated()/1024**2,observations_per_second=32*(step-start_step)/max(time.perf_counter()-start,1e-9),**direction_losses),global_step=step);tv.heartbeat(step)
            tv.log("train",dict(masked_valid_pixels=masked_pixels,cross_diagnostic_valid_pixels=cross_pixels,cross_target_exposure=16*step if args.mode=="cross30" else 0,gpu_allocated_mib=torch.cuda.memory_allocated()/1024**2),global_step=step)
            if args.observation_mode is not None:tv.log("train",dict(observation_loss=observation_total,observation_valid_pixels=observation_pixels,observation_target_exposure=8*step if args.observation_mode=="fixed" else 0,observation_diagnostic_exposure=8*step),global_step=step)
            if step%10==0 or step==1:print(json.dumps(dict(tag=args.tag,step=step,loss=total,cross_loss=cross_total,directions=direction_losses,seconds=time.perf_counter()-start)),flush=True)
            if step%interval==0:
                if args.observation_mode is None:result=evaluate_cross(model,validation,statistics,precision)
                else:
                    from .observation_constraint import evaluate_observation
                    result=evaluate_observation(model,validation,statistics,precision)
                score=result["score"]
                if not np.isfinite(score):raise FloatingPointError("nonfinite cross validation")
                improved=score<best;best=min(best,score)
                if score<anchor*(1-settings["relative_improvement"]):anchor=score;stale=0
                else:stale+=1
                if step>=settings["warmup_updates"]:scheduler.step(score)
                history.append(dict(step=step,score=score,lr=optimizer.param_groups[0]["lr"]))
                (out/f"validation_{step}.json").write_text(json.dumps(result,indent=2));tv.log("val",dict(score=score,**{direction+"/normalized_rmse":float(np.mean([v["normalized_rmse"] for v in values.values()])) for direction,values in result["directions"].items()}),global_step=step)
                if args.observation_mode is not None:tv.log("val",dict(observed_fine30_normalized_rmse=result["observed_fine30"]["score"]),global_step=step)
                save(last)
                if improved:save(out/"best_unlabeled.pt")
                if not args.engineering and step>=minimum and stale>=settings["stop_patience"] and optimizer.param_groups[0]["lr"]<=settings["lr"]*settings["stop_lr_fraction"]:stop_reason="validation_plateau";break
            if step%100==0:save(last)
            if step%1000==0:torch.save(dict(model=model.state_dict(),step=step,statistics=statistics),out/f"step{step}.pt")
            if args.stop_after and step>=args.stop_after:
                stop_reason="engineering_pause";save(last);tv.finish("cancelled","cross engineering pause for disk resume");return
        save(last);tv.finish();result=dict(status="engineering_complete" if args.engineering else "training_complete_pending_evaluation",mode=args.mode,observation_mode=args.observation_mode,budget_stage=budget,step=step,stop_reason=stop_reason,best_score=best,last_score=history[-1]["score"],observation_exposure=32*step,pair_slot_exposure=8*step,pair_exposure=pair_sampler.exposure,seconds=time.perf_counter()-start,peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2)
        (out/"training_result.json").write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except BaseException as exc:tv.finish("failed",type(exc).__name__+": "+str(exc));raise


if __name__=="__main__":main()
