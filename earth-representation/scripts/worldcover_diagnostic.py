"""Existing WorldCover agreement, not independent ground-truth validation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if __name__=="__main__" and torch.cuda.is_available():torch.cuda.init()

from scripts.frozen_segmentation_probe import ROOT,Decoder,encoder
from src.ssl.tvlp_writer import TvlpRun
from landslide_seg.training.segformer_metrics import pixel_average_precision

CLASSES=[10,20,30,40,50,60,70,80,90,95,100]
NAMES=["树木覆盖","灌丛","草地","农田","建成区","裸地或稀疏植被","雪冰","永久水体","草本湿地","红树林","苔藓地衣"]


def map_score(prob,target):
    values=[pixel_average_precision(prob[:,i],target[:,i]) for i in range(prob.shape[1])]
    present=[x for x in values if x is not None]
    return float(np.mean(present)) if present else None,values


def confusion(prob,target,valid):
    p=prob.argmax(1);y=target
    return torch.bincount((y[valid]*len(CLASSES)+p[valid]).long(),minlength=len(CLASSES)**2).reshape(len(CLASSES),len(CLASSES))


def miou(count):
    intersection=count.diag();union=count.sum(0)+count.sum(1)-intersection
    values=intersection/union.clamp_min(1)
    return float(values[union>0].mean()),[float(v) if u>0 else None for v,u in zip(values,union)]


def run(tag,path,images,targets,valid,rows):
    out=ROOT/"outputs/verification/worldcover"/tag
    if (out/"result.json").exists():return json.loads((out/"result.json").read_text())
    out.mkdir(parents=True,exist_ok=True)
    indices={s:torch.tensor([i for i,r in enumerate(rows) if r["split"]==s],device="cuda") for s in ["train","val","test"]}
    model,forward,info=encoder(path);parts=None
    with torch.no_grad():
        for start in range(0,len(images),8):
            result=forward(images[start:start+8])
            if parts is None:parts=[torch.empty((len(images),*f.shape[1:]),device="cuda") for f in result]
            for part,value in zip(parts,result):part[start:start+len(value)]=value
    del model
    train,val,test=[indices[k] for k in ["train","val","test"]]
    pooled=parts[-1].mean((2,3));mean=pooled[train].mean(0);std=pooled[train].std(0).clamp_min(1e-6);pooled=(pooled-mean)/std
    labels=torch.stack([((targets==i)&valid).sum((1,2))/valid.sum((1,2)).clamp_min(1)>=.01 for i in range(len(CLASSES))],1)
    positive=labels[train].sum(0);active=(positive>=3)&(positive<=len(train)-3)
    selected=active.nonzero().flatten();labels=labels[:,active]
    if len(selected)<2:raise ValueError("not enough nonconstant land-cover labels in TRAIN")
    torch.manual_seed(1234);classifier=torch.nn.Linear(pooled.shape[1],len(selected)).cuda()
    opt=torch.optim.AdamW(classifier.parameters(),lr=.01,weight_decay=.0001)
    best=-1.;stale=0;cls_step=0
    tv=TvlpRun(ROOT/"outputs/verification/tvlp",project="earth-representation",name="WorldCover弱标签｜"+tag,run_id="worldcover_"+tag,seed=1234,
        metric_definitions={"val/classification_map":{"direction":"max"},"val/segmentation_miou":{"direction":"max"},"train/classification_loss":{"direction":"min"},"train/segmentation_loss":{"direction":"min"}},
        config=dict(checkpoint=path,source="WorldCover product agreement only",image_grid_m=20,split="15/5/5 geographic groups; no known pretraining-neighborhood overlap",classification="multi-label >=1% valid area; TRAIN nonconstant classes only"))
    try:
        for step in range(1,1001):
            loss=F.binary_cross_entropy_with_logits(classifier(pooled[train]),labels[train].float())
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if step%20:continue
            with torch.no_grad():prob=classifier(pooled[val]).sigmoid().cpu().numpy()
            score,_=map_score(prob,labels[val].cpu().numpy())
            if score is None:raise ValueError("no positive validation classes")
            improved=score>best;stale=0 if improved else stale+20
            if improved:best=score;torch.save(dict(head=classifier.state_dict(),mean=mean.cpu(),std=std.cpu(),classes=selected.cpu(),step=step),out/"classification_best.pt")
            tv.log("train",{"classification_loss":float(loss.detach())},global_step=step);tv.log("val",{"classification_map":score},global_step=step);tv.heartbeat(step)
            cls_step=step
            if step>=100 and stale>=100:break
        classification_best=best
        state=torch.load(out/"classification_best.pt",map_location="cuda",weights_only=False);classifier.load_state_dict(state["head"])
        with torch.no_grad():test_prob=classifier(pooled[test]).sigmoid().cpu().numpy()
        test_labels=labels[test].cpu().numpy();test_map,per_class_ap=map_score(test_prob,test_labels)
        prevalence=test_labels.mean(0).tolist()
        del classifier,pooled,opt
        # The segmentation head uses the same four-level consumer as the landslide probe.
        feature_stats=[]
        for part in parts:
            m=part[train].mean((0,2,3),keepdim=True);s=part[train].std((0,2,3),keepdim=True).clamp_min(1e-6);part.sub_(m).div_(s)
            feature_stats.append(dict(mean=m.cpu(),std=s.cpu()))
        torch.manual_seed(1234);head=Decoder([p.shape[1] for p in parts],classes=len(CLASSES),output_size=256).cuda()
        opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
        counts=torch.bincount(targets[train][valid[train]],minlength=len(CLASSES)).float()
        weights=torch.where(counts>0,counts.clamp_min(1).rsqrt(),0);weights=weights/weights[weights>0].mean()
        rng=torch.Generator(device="cuda").manual_seed(1234);best=-1.;stale=0;seg_step=0
        @torch.no_grad()
        def evaluate(ids):
            total=torch.zeros(len(CLASSES),len(CLASSES),device="cuda",dtype=torch.long)
            for batch in ids.split(8):total+=confusion(head([p[batch] for p in parts]),targets[batch],valid[batch])
            return total
        for epoch in range(40):
            order=train[torch.randperm(len(train),generator=rng,device="cuda")];head.train();total=0.
            for batch in order.split(8):
                logits=head([p[batch] for p in parts]);loss=F.cross_entropy(logits,targets[batch],weight=weights,ignore_index=255)
                opt.zero_grad(set_to_none=True);loss.backward();opt.step();seg_step+=1;total+=float(loss.detach())
            head.eval();score,_=miou(evaluate(val));improved=score>best;stale=0 if improved else stale+1
            if improved:best=score;torch.save(dict(head=head.state_dict(),epoch=epoch+1,feature_stats=feature_stats),out/"segmentation_best.pt")
            tv.log("train",{"segmentation_loss":total/len(order.split(8))},global_step=cls_step+seg_step);tv.log("val",{"segmentation_miou":score},global_step=cls_step+seg_step);tv.heartbeat(cls_step+seg_step)
            if epoch+1>=10 and stale>=8:break
        chosen=torch.load(out/"segmentation_best.pt",map_location="cuda",weights_only=False);head.load_state_dict(chosen["head"])
        test_confusion=evaluate(test);test_miou,per_class_iou=miou(test_confusion)
        report=dict(status="supplemental_model_map_agreement_only",tag=tag,checkpoint=path,encoder=info,
            classifier=dict(classes=[CLASSES[i] for i in selected.tolist()],names=[NAMES[i] for i in selected.tolist()],best_step=state["step"],last_step=cls_step,validation_map=classification_best,test_map=test_map,per_class_ap=per_class_ap,test_prevalence=prevalence,test_positive_counts=test_labels.sum(0).tolist()),
            segmentation=dict(classes=CLASSES,names=NAMES,validation_miou=best,test_miou=test_miou,per_class_iou=per_class_iou,confusion=test_confusion.cpu().tolist(),best_epoch=chosen["epoch"],last_epoch=epoch+1),
            boundary="both tasks derive from the same WorldCover map source; not independent tasks, field truth, or native10m detail validation; 60/20/20 scene split by geographic groups")
        (out/"result.json").write_text(json.dumps(report,ensure_ascii=False,indent=2));tv.finish();print(json.dumps(dict(tag=tag,test_map=test_map,test_miou=test_miou)),flush=True)
        del parts,head,opt;torch.cuda.empty_cache();return report
    except BaseException as exc:tv.finish("failed",str(exc));raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--include-routes",action="store_true");args=ap.parse_args()
    data=ROOT/"outputs/verification/worldcover_data"
    manifest=json.loads((data/"manifest.json").read_text())
    with np.load(data/"arrays.npz") as z:
        images=torch.from_numpy(z["image"]).cuda();raw=torch.from_numpy(z["label"].astype("int64")).cuda();valid=torch.from_numpy(z["valid"]).cuda()
    target=torch.full_like(raw,255)
    for i,label in enumerate(CLASSES):target[raw==label]=i
    valid &= target!=255;target[~valid]=255
    jobs=[("random","random")]+[(tag,str(ROOT/"outputs/verification/runs"/(tag+".best_unlabeled.pt"))) for tag in ["verified_s2_standard","verified_s2_generic","verified_s2_physical","corrective_scale_uniform","corrective_scale_random","corrective_scale_selective"]]
    jobs += [("legacy_off3k",str(ROOT/"outputs/dino_official/off3k.pt"))]+[("legacy_e1_"+str(i),str(ROOT/f"outputs/e1_official/E1-{i}.pt")) for i in [0,1,2]]+[("legacy_e2_"+str(i),str(ROOT/f"outputs/e2_official/E2-{i}.pt")) for i in [0,3,4]]
    if args.include_routes:jobs += [("sensor_"+r,str(ROOT/"outputs/verification/runs"/("sensor_"+r+".best_unlabeled.pt"))) for r in ["single","mixed","paired"]]
    jobs += [("DOFA","dofa")]
    results=[]
    for tag,path in jobs:results.append(run(tag,path,images,target,valid,manifest["rows"]))
    (ROOT/"outputs/verification/worldcover_comparison.json").write_text(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
