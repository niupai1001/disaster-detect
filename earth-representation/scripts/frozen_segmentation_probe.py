"""Auxiliary landslide transfer: frozen encoder, identical lightweight decoder.

This does not replace independent land-cover classification/segmentation.
The fixed project TRAIN/DEV members and cached valid/target masks are retained.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "landslide-seg/src"))
from scripts.cross_sensor_eval import build
from src.data.triples import S2_NAMES, CENTRE, GSD
from src.ssl.tvlp_writer import TvlpRun
from landslide_seg.training.segformer_metrics import evaluate_tile_v2, summarize_tiles_v2, summarize_by_event_v2


class Decoder(torch.nn.Module):
    def __init__(self, channels, classes=1, output_size=128):
        super().__init__()
        # Construct the shared trunk first so initialization is identical for DOFA too.
        self.classes=classes
        self.output_size=output_size
        self.fuse = torch.nn.Sequential(torch.nn.Conv2d(128,32,3,padding=1),torch.nn.ReLU(),torch.nn.Conv2d(32,classes,1))
        self.projections = torch.nn.ModuleList(torch.nn.Conv2d(c,32,1) for c in channels)

    def forward(self, features):
        projected=[F.interpolate(p(f),size=(self.output_size//4,self.output_size//4),mode="bilinear",align_corners=False) for p,f in zip(self.projections,features)]
        logits=F.interpolate(self.fuse(torch.cat(projected,1)),size=(self.output_size,self.output_size),mode="bilinear",align_corners=False)
        return logits[:,0] if self.classes==1 else logits


def data():
    manifest=ROOT.parent/"landslide-seg/manifests/v4r1/pool_100.jsonl"
    rows=[json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    images=[];targets=[];valid=[]
    for row in rows:
        with np.load(row["cache"]) as d:
            # Source cache axes are x,y; transpose image, target and validity together.
            v=d["valid"].astype(bool).T & (d["mask"].T !=255)
            images.append(np.where(v[None],d["x10"].transpose(0,2,1).astype(np.float32)/10000,0))
            targets.append(d["mask"].T==1);valid.append(v)
    return rows,torch.from_numpy(np.stack(images)).cuda(),torch.from_numpy(np.stack(targets)).cuda(),torch.from_numpy(np.stack(valid)).cuda(),hashlib.sha256(manifest.read_bytes()).hexdigest()


def encoder(checkpoint):
    if checkpoint=="dofa":
        path=ROOT.parent/"landslide-seg/scripts/native_official_probe.py"
        spec=importlib.util.spec_from_file_location("earth_native_dofa",path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        model,info=module.build_dofa(torch.device("cuda"))
        def forward(x):
            return model(F.interpolate(x,size=(224,224),mode="bilinear",align_corners=False),[CENTRE[k]/1000 for k in S2_NAMES])
        return model,forward,info
    torch.manual_seed(0)
    model=build().cuda().eval()
    if checkpoint!="random":
        state=torch.load(checkpoint,map_location="cuda",weights_only=False)
        model.load_state_dict(state["teacher"])
    model.requires_grad_(False)
    c=torch.tensor([CENTRE[k] for k in S2_NAMES],device="cuda")
    g=torch.tensor([GSD[k] for k in S2_NAMES],device="cuda")
    def forward(x):
        levels=model.backbone.encoder(x,c,g)
        return [levels[k] for k in ["s4","s8","s16","s32"]]
    return model,forward,dict(checkpoint=checkpoint,checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint!="random" else None,params=sum(p.numel() for p in model.parameters()))


@torch.no_grad()
def predict(head,features,ids,batch=32):
    return torch.cat([head([f[chunk] for f in features]).sigmoid() for chunk in ids.split(batch)])


def event_iou(prob,target,valid,events):
    pred=prob>=.5
    values=[]
    for name in sorted(set(events)):
        subset=torch.tensor([e==name for e in events],device=prob.device)
        p=pred[subset];y=target[subset];v=valid[subset]
        tp=(p&y&v).sum();union=((p|y)&v).sum()
        values.append(float(tp/union.clamp_min(1)))
    return sum(values)/len(values)


def run(checkpoint,tag,rows,images,target,valid,manifest_sha,max_epochs):
    out=ROOT/"outputs/verification/probes"/tag
    if (out/"result.json").exists():
        return json.loads((out/"result.json").read_text())
    out.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    model,forward,info=encoder(checkpoint)
    features=None
    with torch.no_grad():
        for start in range(0,len(images),16):
            batch=forward(images[start:start+16])
            if features is None:
                features=[torch.empty((len(images),*x.shape[1:]),device="cuda",dtype=torch.float32) for x in batch]
            for f,x in zip(features,batch):f[start:start+len(x)]=x.float()
    train=torch.tensor([i for i,r in enumerate(rows) if r["split"]=="train"],device="cuda")
    val=torch.tensor([i for i,r in enumerate(rows) if r["split"]=="val"],device="cuda")
    events=[rows[i]["event_id"] for i in val.tolist()]
    stats=[]
    for f in features:
        mean=f[train].mean((0,2,3),keepdim=True);std=f[train].std((0,2,3),keepdim=True).clamp_min(1e-6)
        f.sub_(mean).div_(std);stats.append(dict(mean=mean.cpu(),std=std.cpu()))
    del model
    torch.manual_seed(1234)
    head=Decoder([f.shape[1] for f in features]).cuda()
    opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=.5,patience=2,min_lr=1e-5)
    generator=torch.Generator(device="cuda").manual_seed(1234)
    best=-1.;stale=0;step=0;start_epoch=0;score=0.
    last=out/"last.pt"
    if last.exists():
        state=torch.load(last,map_location="cuda",weights_only=False)
        head.load_state_dict(state["head"]);opt.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"])
        best,stale,step,start_epoch=state["best"],state["stale"],state["step"],state["epoch"]
        generator.set_state(state["generator"].cpu())
        score=state.get("score",best)
    else:
        torch.save(head.state_dict(),out/"initial_head.pt")
    tv=TvlpRun(ROOT/"outputs/verification/tvlp",project="earth-representation",name=tag,
        run_id="probe_"+tag,resume=last.exists(),seed=1234,
        metric_definitions={"train/loss":{"direction":"min"},"val/event_mean_iou":{"direction":"max"}},
        config=dict(checkpoint=checkpoint,head_seed=1234,max_epochs=max_epochs,train=2223,dev=669,manifest_sha256=manifest_sha),
        model=dict(name="Frozen encoder + multiscale 32-channel decoder"))
    epoch=start_epoch-1
    try:
        for epoch in range(start_epoch,max_epochs):
            order=train[torch.randperm(len(train),generator=generator,device="cuda")]
            head.train();loss_sum=0.
            for ids in order.split(32):
                logits=head([f[ids] for f in features]);y=target[ids].float();v=valid[ids]
                probability=logits.sigmoid()
                bce=F.binary_cross_entropy_with_logits(logits,y,reduction="none")[v].mean()
                dice=1-(2*(probability*y*v).sum()+1e-6)/((probability*v).sum()+(y*v).sum()+1e-6)
                loss=.5*bce+.5*dice
                if not torch.isfinite(loss):raise FloatingPointError("nonfinite frozen probe loss")
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
                loss_sum+=float(loss.detach());step+=1
            head.eval()
            prob=predict(head,features,val)
            score=event_iou(prob,target[val],valid[val],events)
            scheduler.step(score);improved=score>best
            stale=0 if improved else stale+1
            if improved:best=score
            state=dict(head=head.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),generator=generator.get_state(),
                best=best,score=score,stale=stale,step=step,epoch=epoch+1,train_feature_stats=stats)
            tmp=out/"checkpoint.tmp";torch.save(state,tmp);tmp.replace(last)
            if improved:torch.save(state,out/"best.pt")
            tv.log("train",{"loss":loss_sum/len(order.split(32)),"lr":opt.param_groups[0]["lr"]},global_step=step)
            tv.log("val",{"event_mean_iou":score},global_step=step);tv.heartbeat(step)
            print(json.dumps(dict(tag=tag,epoch=epoch+1,step=step,iou=score,best=best)),flush=True)
            if epoch+1>=10 and stale>=5:break
        last_prob=predict(head,features,val).cpu().numpy()
        chosen=torch.load(out/"best.pt",map_location="cuda",weights_only=False);head.load_state_dict(chosen["head"])
        best_prob=predict(head,features,val).cpu().numpy()
        gt=target[val].cpu().numpy();vm=valid[val].cpu().numpy()
        tile_rows=[evaluate_tile_v2(p,y,v,sample_id=rows[i]["sample_id"],event_id=rows[i]["event_id"]) for i,p,y,v in zip(val.tolist(),best_prob,gt,vm)]
        result=dict(tag=tag,status="completed_auxiliary_only",encoder=info,checkpoint=checkpoint,
            best_event_mean_iou=best,last_event_mean_iou=score,best_epoch=chosen["epoch"],last_epoch=epoch+1,
            step=step,head_parameters=sum(p.numel() for p in head.parameters()),metric_version="project_segformer_v2",
            per_event=summarize_by_event_v2(tile_rows),overall=summarize_tiles_v2(tile_rows),
            seconds=time.perf_counter()-t0,peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2,
            train=2223,dev=669,manifest_sha256=manifest_sha,orientation="image,target,valid transposed together",input="physical reflectance; no stats_core25 z-score",feature_normalization="TRAIN-only per-channel mean/std",
            boundary="DEV reused for head selection; historical encoders may have seen DEV imagery during old pretraining; not independent land-cover evaluation")
        (out/"result.json").write_text(json.dumps(result,indent=2))
        np.savez_compressed(out/"predictions.npz",best=best_prob,last=last_prob,ids=np.array([rows[i]["sample_id"] for i in val.tolist()]))
        tv.finish()
        del features,head,opt
        torch.cuda.empty_cache()
        return result
    except BaseException as exc:
        tv.finish("failed",str(exc));raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--tag",required=True);ap.add_argument("--checkpoint",required=True);ap.add_argument("--max-epochs",type=int,default=40)
    args=ap.parse_args();rows,x,y,v,sha=data();run(args.checkpoint,args.tag,rows,x,y,v,sha,args.max_epochs)


if __name__=="__main__":main()
