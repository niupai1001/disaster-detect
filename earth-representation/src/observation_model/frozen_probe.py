"""One linear readout family, encoder fixed before labels; bounded auxiliary diagnostics."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import ROOT, BalancedSampler
from .frozen_features import auxiliary_data
from src.ssl.tvlp_writer import TvlpRun

sys.path.insert(0,str(ROOT.parent/"landslide-seg/src"))
from landslide_seg.training.segformer_metrics import pixel_average_precision,evaluate_tile_v2,summarize_tiles_v2,summarize_by_event_v2

CLASSES=[10,20,30,40,50,60,70,80,90,95,100]


def load_features(path, indices):
    return torch.from_numpy(np.array(path[indices],dtype=np.float32)).cuda()


def map_score(probability,truth):
    per_class=[pixel_average_precision(probability[:,i],truth[:,i]) for i in range(truth.shape[1])]
    present=[r for r in per_class if r is not None]
    return float(np.mean(present)) if present else None,per_class


@torch.no_grad()
def score(head,features,ids,mean,std,labels,valid,rows,task,dataset,batch=8):
    probabilities=[];truths=[];confusions={}
    classes=11 if dataset=="worldcover" else 2
    for start in range(0,len(ids),batch):
        indices=ids[start:start+batch];x=load_features(features,indices)
        if task=="classification":
            x=x.mean((2,3));probabilities.append(head((x-mean)/std).sigmoid().cpu().numpy());truths.append(labels[indices].cpu().numpy())
        else:
            logits=head((x-mean)/std);prob=F.interpolate(logits,size=labels.shape[-2:],mode="bilinear",align_corners=False).softmax(1)
            predicted=prob.argmax(1);target=labels[indices].cuda();mask=valid[indices].cuda()
            probabilities.append(prob.cpu().numpy())
            for j,index in enumerate(indices):
                group=rows[index]["event_id"] if dataset=="landslide" else "all"
                values=target[j][mask[j]]*classes+predicted[j][mask[j]]
                counts=torch.bincount(values,minlength=classes*classes).reshape(classes,classes).cpu().numpy()
                confusions[group]=confusions.get(group,np.zeros_like(counts))+counts
    probability=np.concatenate(probabilities)
    if task=="classification":
        truth=np.concatenate(truths);value,per_class=map_score(probability,truth)
        return value,dict(mean_average_precision=value,per_class_ap=per_class,positive_counts=truth.sum(0).tolist()),probability
    groups={}
    for group,matrix in confusions.items():
        union=matrix.sum(0)+matrix.sum(1)-np.diag(matrix)
        iou=np.divide(np.diag(matrix),union,out=np.zeros(classes,dtype=float),where=union>0)
        value=float(iou[1]) if dataset=="landslide" else float(iou[union>0].mean())
        groups[group]=dict(score=value,confusion=matrix.tolist(),per_class_iou=[float(v) if n else None for v,n in zip(iou,union)])
    value=float(np.mean([r["score"] for r in groups.values()]))
    return value,dict(score=value,groups=groups),probability


def cases(tv,source,rows,ids,probability,labels,valid,task,selected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _,image,_,_,_,_=auxiliary_data(source["dataset"])
    count=min(3,len(ids));fig,axes=plt.subplots(count,5 if task=="segmentation" else 2,figsize=(14,3*count),squeeze=False,constrained_layout=True)
    for j,index in enumerate(ids[:count]):
        rgb=np.clip(image[index,[2,1,0]].transpose(1,2,0)*3,0,1)
        axes[j,0].imshow(rgb);axes[j,0].set_title(rows[index]["sample_id"]+" / visible RGB")
        if task=="classification":
            x=np.arange(len(selected));axes[j,1].bar(x-.2,labels[index].numpy(),width=.4,label="Map presence")
            axes[j,1].bar(x+.2,probability[j],width=.4,label="Predicted probability");axes[j,1].set_xticks(x,[str(CLASSES[k]) for k in selected]);axes[j,1].legend()
        else:
            target=labels[index].numpy();prediction=probability[j].argmax(0);v=valid[index].numpy()
            picture=axes[j,1].imshow(np.ma.masked_where(~v,target),vmin=0,vmax=10 if source["dataset"]=="worldcover" else 1,interpolation="nearest");axes[j,1].set_title("Measured label / map label")
            axes[j,2].imshow(np.ma.masked_where(~v,prediction),vmin=0,vmax=10 if source["dataset"]=="worldcover" else 1,interpolation="nearest");axes[j,2].set_title("Frozen encoder + linear head")
            axes[j,3].imshow((target!=prediction)&v,cmap="magma",vmin=0,vmax=1);axes[j,3].set_title("Label disagreement")
            axes[j,4].imshow(v,cmap="gray",vmin=0,vmax=1);axes[j,4].set_title("Valid evaluation pixels")
        for ax in axes[j]:ax.tick_params(labelsize=7)
    if task=="segmentation":
        codes=CLASSES if source["dataset"]=="worldcover" else [0,1]
        bar=fig.colorbar(picture,ax=axes[:,1:3].ravel().tolist(),ticks=range(len(codes)),shrink=.75)
        bar.set_ticklabels([str(c) for c in codes]);bar.set_label("Map code" if source["dataset"]=="worldcover" else "0 background /1 landslide")
    folder=tv.dir/"artifacts";folder.mkdir(exist_ok=True);fig.savefig(folder/"fixed_cases.png",dpi=140);plt.close(fig)
    tv._events.write(json.dumps(dict(type="artifact",timestamp=datetime.now(timezone.utc).isoformat(),session_id=tv.session_id,
                       path="artifacts/fixed_cases.png",artifact_type="image",tag="fixed-real-cases",phase="val" if source["dataset"]=="landslide" else "test",role="comparison",
                       metadata=dict(generated_after_training=True,ids=[rows[i]["sample_id"] for i in ids[:count]],selection="first3 members, never selected by prediction quality")))+"\n")
    tv._events.flush()


def run(folder,out,task,maximum):
    source=json.loads((folder/"metadata.json").read_text());rows=json.loads((folder/"members.json").read_text());dataset=source["dataset"]
    if task=="classification" and dataset!="worldcover":raise ValueError("classification probe is for existing WorldCover map agreement")
    out.mkdir(exist_ok=True,parents=True)
    if (out/"result.json").exists():return json.loads((out/"result.json").read_text())
    features=np.load(folder/"features.npy",mmap_mode="r")
    with np.load(folder/"targets.npz") as z:raw=z["label"].copy();mask=z["valid"].copy()
    target=np.full(raw.shape,255,dtype=np.int64)
    for i,label in enumerate(CLASSES if dataset=="worldcover" else [0,1]):target[raw==label]=i
    mask &= target!=255
    indices={split:[i for i,r in enumerate(rows) if r["split"]==split] for split in ["train","val","test"]}
    train=indices["train"];validation=indices["val"]
    valid=torch.from_numpy(mask);labels=torch.from_numpy(target)
    selected=None
    if task=="classification":
        fraction=torch.stack([((labels==i)&valid).sum((1,2))/valid.sum((1,2)).clamp_min(1) for i in range(11)],1)
        labels=fraction>=.01;positives=labels[train].sum(0);selected=((positives>=3)&(positives<=len(train)-3)).nonzero().flatten().tolist()
        if len(selected)<2:
            result=dict(status="blocked_insufficient_TRAIN_label_variation",selected_classes=[CLASSES[k] for k in selected],source=str(folder),encoder_fixed=True)
            (out/"result.json").write_text(json.dumps(result,indent=2));return result
        labels=labels[:,selected].float();classes=len(selected)
        pooled=torch.cat([load_features(features,list(range(i,min(i+8,len(rows))))).mean((2,3)) for i in range(0,len(rows),8)])
        mean=pooled[train].mean(0);std=pooled[train].std(0,unbiased=False).clamp_min(1e-6)
    else:
        classes=11 if dataset=="worldcover" else 2;sums=torch.zeros(features.shape[1],device="cuda");squares=torch.zeros_like(sums);count=0
        for i in range(0,len(train),8):
            x=load_features(features,train[i:i+8]);sums+=x.sum((0,2,3));squares+=x.square().sum((0,2,3));count+=x.shape[0]*x.shape[2]*x.shape[3]
        mean=(sums/count)[None,:,None,None];std=(squares/count-(sums/count).square()).clamp_min(1e-12).sqrt()[None,:,None,None]
        counts=torch.bincount(labels[train][valid[train]],minlength=classes).float().cuda();weight=torch.where(counts>0,counts.clamp_min(1).rsqrt(),0);weight/=weight[weight>0].mean()
    torch.manual_seed(1234)
    head=(nn.Linear(features.shape[1],classes) if task=="classification" else nn.Conv2d(features.shape[1],classes,1)).cuda()
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=.5,patience=4,min_lr=1e-6)
    sampler=BalancedSampler({"train":len(train)},seed=1234);step=0;best=-1.;history=[];last=out/"last.pt"
    if last.exists():
        state=torch.load(last,map_location="cuda",weights_only=False);head.load_state_dict(state["head"]);optimizer.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"]);sampler.load_state_dict(state["sampler"])
        step=state["step"];best=state["best"];history=state["history"];torch.set_rng_state(state["torch_rng"].cpu());torch.cuda.set_rng_state_all([r.cpu() for r in state["cuda_rng"]])
    else:torch.save(head.state_dict(),out/"initial_head.pt")
    tag=folder.parent.name+"_"+folder.name+"_"+task
    tv=TvlpRun(Path(os.environ.get("OBSERVATION_TV_ROOT",ROOT/"observation_model/outputs/tvlp")),project="conditional-observation",name="Frozen linear probe / "+tag,run_id=os.environ.get("OBSERVATION_RUN_PREFIX","obs_")+"probe_"+tag,resume=last.exists(),seed=1234,
               metric_definitions={"train/loss":{"direction":"min"},"val/score":{"direction":"max"}},
               config=dict(feature_metadata=source,task=task,max_updates=maximum,batch=32,head="single linear layer /1x1 convolution; input dimensions and parameter counts disclosed",encoder_optimized=False,head_selected_on="VAL only",selected_classes=selected))
    start=time.perf_counter();start_step=step;torch.cuda.reset_peak_memory_stats()
    def save(path):
        state=dict(head=head.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),sampler=sampler.state_dict(),step=step,best=best,history=history,
                   mean=mean,std=std,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),feature_metadata=source,task=task,head_seed=1234)
        temporary=path.with_suffix(".tmp");torch.save(state,temporary);temporary.replace(path)
    try:
        while step<maximum:
            head.train();ids=[train[i] for i in sampler.take("train",32)];optimizer.zero_grad(set_to_none=True)
            if task=="classification":loss=F.binary_cross_entropy_with_logits(head((pooled[ids]-mean)/std),labels[ids].cuda())
            else:
                logits=head((load_features(features,ids)-mean)/std);logits=F.interpolate(logits,size=labels.shape[-2:],mode="bilinear",align_corners=False)
                loss=F.cross_entropy(logits,labels[ids].cuda().masked_fill(~valid[ids].cuda(),255),weight=weight,ignore_index=255)
            if not torch.isfinite(loss):raise FloatingPointError("nonfinite frozen probe loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),3.,error_if_nonfinite=True);optimizer.step();step+=1
            elapsed=time.perf_counter()-start
            tv.log("train",dict(loss=float(loss.detach()),lr=optimizer.param_groups[0]["lr"],observation_exposure=step*32,
                                 observations_per_second=(step-start_step)*32/max(elapsed,1e-9),gpu_allocated_mib=torch.cuda.memory_allocated()/1024**2,gpu_peak_mib=torch.cuda.max_memory_allocated()/1024**2),global_step=step);tv.heartbeat(step)
            if step%100==0 or step==maximum:
                head.eval();value,details,_=score(head,features,validation,mean,std,labels,valid,rows,task,dataset)
                if value is None or not np.isfinite(value):raise ValueError("no finite validation probe score")
                improved=value>best;best=max(best,value);scheduler.step(value);history.append(dict(step=step,score=value))
                tv.log("val",dict(score=value),global_step=step);save(last)
                if improved:save(out/"best.pt")
                print(json.dumps(dict(tag=tag,step=step,score=value,best=best)),flush=True)
        final_ids=validation if dataset=="landslide" else indices["test"]
        head.eval();last_value,last_details,last_probability=score(head,features,final_ids,mean,std,labels,valid,rows,task,dataset)
        chosen=torch.load(out/"best.pt",map_location="cuda",weights_only=False);head.load_state_dict(chosen["head"])
        best_value,best_details,probability=score(head,features,final_ids,mean,std,labels,valid,rows,task,dataset)
        cases(tv,source,rows,final_ids,probability,labels,valid,task,selected)
        result=dict(status="completed_auxiliary_only",task=task,dataset=dataset,head_seed=1234,step=step,best_step=chosen["step"],validation_best=best,
                    evaluation_split="val" if dataset=="landslide" else "test",best_score=best_value,last_score=last_value,best_details=best_details,last_details=last_details,
                    head_parameters=sum(p.numel() for p in head.parameters()),feature_metadata=source,selected_class_codes=[CLASSES[k] for k in selected] if selected else None,
                    seconds=time.perf_counter()-start,peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2,observation_exposure=step*32,
                    gpu_name=torch.cuda.get_device_name(),cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES","all; usingcuda:0"),
                    boundary="fixed encoder, linear readout only; WorldCover is map agreement; slides auxiliary and DEV reused for head selection; input dimension/head parameter differences disclosed, not strict equal-capacity encoders")
        if dataset=="landslide":
            tile_rows=[evaluate_tile_v2(probability[j,1],target[i]==1,mask[i],sample_id=rows[i]["sample_id"],event_id=rows[i]["event_id"]) for j,i in enumerate(final_ids)]
            result["best_project_metrics"]=dict(metric_version="project_segformer_v2",per_event=summarize_by_event_v2(tile_rows),overall=summarize_tiles_v2(tile_rows))
            (out/"best_tiles.json").write_text(json.dumps(tile_rows))
        np.savez_compressed(out/"predictions.npz",best=probability,last=last_probability,ids=np.array([rows[i]["sample_id"] for i in final_ids]))
        (out/"result.json").write_text(json.dumps(result,indent=2));tv.finish();return result
    except BaseException as exc:tv.finish("failed",str(exc));raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--features",type=Path,required=True);parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--task",choices=["classification","segmentation","both"],default="both");parser.add_argument("--max-updates",type=int,default=2000);args=parser.parse_args()
    for task in ["classification","segmentation"] if args.task=="both" else [args.task]:
        result=run(args.features,args.out/task,task,args.max_updates);print(json.dumps({k:result.get(k) for k in ["status","dataset","task","best_score","last_score"]}),flush=True)


if __name__=="__main__":main()
