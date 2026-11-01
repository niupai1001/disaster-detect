"""Fine-tune the verified baseline and best eligible CNN after frozen probes."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.frozen_segmentation_probe import ROOT,Decoder,data,encoder,event_iou
from src.data.triples import S2_NAMES,CENTRE,GSD
from src.ssl.tvlp_writer import TvlpRun
from landslide_seg.training.segformer_metrics import evaluate_tile_v2,summarize_tiles_v2,summarize_by_event_v2


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--probe-tag",required=True);args=ap.parse_args()
    probe=ROOT/"outputs/verification/probes"/args.probe_tag
    result=json.loads((probe/"result.json").read_text())
    if result["checkpoint"] in ("dofa","random") or args.probe_tag.startswith("legacy_"):
        raise ValueError("only verified CNN candidates can enter this controlled fine-tuning pair")
    out=ROOT/"outputs/verification/finetune"/args.probe_tag;out.mkdir(parents=True,exist_ok=True)
    if (out/"result.json").exists():return
    rows,x,y,v,sha=data();train=torch.tensor([i for i,r in enumerate(rows) if r["split"]=="train"],device="cuda")
    val=torch.tensor([i for i,r in enumerate(rows) if r["split"]=="val"],device="cuda");events=[rows[i]["event_id"] for i in val.tolist()]
    model,_,info=encoder(result["checkpoint"]);enc=model.backbone.encoder;enc.requires_grad_(True)
    state=torch.load(probe/"best.pt",map_location="cuda",weights_only=False)
    torch.manual_seed(1234);head=Decoder([64,128,256,512]).cuda();head.load_state_dict(state["head"])
    stats=[(a["mean"],a["std"]) for a in state["train_feature_stats"]]
    del model
    c=torch.tensor([CENTRE[k] for k in S2_NAMES],device="cuda");g=torch.tensor([GSD[k] for k in S2_NAMES],device="cuda")
    def forward(image):
        levels=enc(image,c,g)
        return head([(levels[k]-m)/s for k,(m,s) in zip(["s4","s8","s16","s32"],stats)])
    @torch.no_grad()
    def predict():
        enc.eval();head.eval()
        return torch.cat([forward(x[ids]).sigmoid() for ids in val.split(32)])
    opt=torch.optim.AdamW([dict(params=enc.parameters(),lr=1e-5),dict(params=head.parameters(),lr=1e-4)],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=.5,patience=5,min_lr=[1e-7,1e-6])
    rng=torch.Generator(device="cuda").manual_seed(1234)
    best=-1.;stale=0;step=0;start=0;score=0.;epoch=-1
    if (out/"last.pt").exists():
        state=torch.load(out/"last.pt",map_location="cuda",weights_only=False)
        enc.load_state_dict(state["encoder"]);head.load_state_dict(state["head"]);opt.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"])
        rng.set_state(state["rng"].cpu());best,stale,step,start,score=[state[k] for k in ["best","stale","step","epoch","score"]]
    tv=TvlpRun(ROOT/"outputs/verification/tvlp",project="earth-representation",name="finetune_"+args.probe_tag,
        run_id="finetune_"+args.probe_tag,resume=(out/"last.pt").exists(),seed=1234,
        metric_definitions={"train/loss":{"direction":"min"},"val/event_mean_iou":{"direction":"max"}},
        config=dict(encoder_lr=1e-5,head_lr=1e-4,max_epochs=100,min_epochs=20,patience=10,batch_size=32,manifest_sha256=sha,probe=str(probe)))
    t0=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    try:
        for epoch in range(start,100):
            enc.train();head.train();total=0.
            order=train[torch.randperm(len(train),generator=rng,device="cuda")]
            for ids in order.split(32):
                image=x[ids];target=y[ids];valid=v[ids]
                horizontal=torch.rand((len(ids),1,1),device="cuda",generator=rng)<.5
                vertical=torch.rand((len(ids),1,1),device="cuda",generator=rng)<.5
                image=torch.where(horizontal[:,None],image.flip(-1),image);target=torch.where(horizontal,target.flip(-1),target);valid=torch.where(horizontal,valid.flip(-1),valid)
                image=torch.where(vertical[:,None],image.flip(-2),image);target=torch.where(vertical,target.flip(-2),target);valid=torch.where(vertical,valid.flip(-2),valid)
                logits=forward(image);probability=logits.sigmoid();target=target.float()
                bce=F.binary_cross_entropy_with_logits(logits,target,reduction="none")[valid].mean()
                dice=1-(2*(probability*target*valid).sum()+1e-6)/((probability*valid).sum()+(target*valid).sum()+1e-6)
                loss=.5*bce+.5*dice
                if not torch.isfinite(loss):raise FloatingPointError("nonfinite fine-tuning loss")
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(head.parameters()),1.);opt.step();step+=1;total+=float(loss.detach())
            prob=predict();score=event_iou(prob,y[val],v[val],events);scheduler.step(score)
            improved=score>best;stale=0 if improved else stale+1
            if improved:best=score
            state=dict(encoder=enc.state_dict(),head=head.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=rng.get_state(),best=best,stale=stale,step=step,epoch=epoch+1,score=score,train_feature_stats=state["train_feature_stats"])
            tmp=out/"checkpoint.tmp";torch.save(state,tmp);tmp.replace(out/"last.pt")
            if improved:torch.save(state,out/"best.pt")
            tv.log("train",{"loss":total/len(order.split(32)),"lr":opt.param_groups[0]["lr"]},global_step=step);tv.log("val",{"event_mean_iou":score},global_step=step);tv.heartbeat(step)
            print(json.dumps(dict(tag=args.probe_tag,epoch=epoch+1,step=step,iou=score,best=best)),flush=True)
            if epoch+1>=20 and stale>=10:break
        last=prob.cpu().numpy();chosen=torch.load(out/"best.pt",map_location="cuda",weights_only=False);enc.load_state_dict(chosen["encoder"]);head.load_state_dict(chosen["head"]);best_prob=predict().cpu().numpy()
        gt=y[val].cpu().numpy();valid=v[val].cpu().numpy()
        tiles=[evaluate_tile_v2(p,t,m,sample_id=rows[i]["sample_id"],event_id=rows[i]["event_id"]) for i,p,t,m in zip(val.tolist(),best_prob,gt,valid)]
        report=dict(status="completed_auxiliary_finetuning",tag=args.probe_tag,best_event_mean_iou=best,last_event_mean_iou=score,best_epoch=chosen["epoch"],last_epoch=epoch+1,step=step,pretraining_checkpoint=result["checkpoint"],overall=summarize_tiles_v2(tiles),per_event=summarize_by_event_v2(tiles),seconds=time.perf_counter()-t0,peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2,manifest_sha256=sha,seed=1234,input="physical reflectance, cache valid mask; fixed TRAIN-only feature standardization from frozen probe",boundary="DEV selected the candidate and epochs; no independent landcover claim")
        (out/"result.json").write_text(json.dumps(report,indent=2));np.savez_compressed(out/"predictions.npz",best=best_prob,last=last,ids=np.array([rows[i]["sample_id"] for i in val.tolist()]))
        tv.finish()
    except BaseException as exc:tv.finish("failed",str(exc));raise


if __name__=="__main__":main()
