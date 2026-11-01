"""Unseen development locations: 18 queries, 290 TRAIN +18 DEV gallery."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if __name__=="__main__" and torch.cuda.is_available():torch.cuda.init()

from scripts.frozen_segmentation_probe import encoder,ROOT
from src.data.triples import S2_NAMES,LANDSAT_NAMES,CENTRE,GSD


def statistics(query,gallery,truth,nearby):
    similarity=query@gallery.T
    true=similarity[torch.arange(len(query),device=query.device),truth]
    greater=(similarity>true[:,None]).sum(1);equal=(similarity==true[:,None]).sum(1)
    rank=greater+.5*(equal-1)+1
    # Expected hit probability breaks exact ties without favoring gallery order.
    p1=((1-greater)/equal).clamp(0,1);p10=((10-greater)/equal).clamp(0,1)
    order=similarity.argsort(dim=1,descending=True,stable=True)
    near=torch.as_tensor(nearby,device=query.device)
    return dict(recall_at_1=float(p1.mean()),recall_at_10=float(p10.mean()),median_rank=float(rank.median()),
        near640m_recall_at_1=float(near.gather(1,order[:,:1]).any(1).float().mean()),
        near640m_recall_at_10=float(near.gather(1,order[:,:10]).any(1).float().mean()),
        query_n=len(query),gallery_n=len(gallery),per_query_r1=p1.cpu().tolist(),per_query_r10=p10.cpu().tolist())


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--include-routes",action="store_true");args=ap.parse_args()
    out=ROOT/"outputs/verification"
    mt=json.loads((out/"sensor_release/manifest.json").read_text());me=json.loads((out/"sensor_evaluation/manifest.json").read_text())
    with np.load(out/"sensor_release/arrays.npz") as z:train={k:z[k] for k in ["s2","landsat"]}
    with np.load(out/"sensor_evaluation/arrays.npz") as z:test={k:z[k] for k in ["s2","landsat"]}
    arrays={k:np.concatenate([train[k],test[k]]) for k in train}
    ids=mt["train_ids"]+me["ids"];ntrain=len(mt["train_ids"])
    records={r["sample_id"]:r for r in [json.loads(l) for l in (ROOT.parent/"landslide-seg/manifests/v4r1/pool_100.jsonl").read_text().splitlines()]}
    from pyproj import Transformer
    xy=[]
    for sid in ids:
        r=records[sid];b=r["bbox"];lon,lat=Transformer.from_crs(r["crs"],"EPSG:4326",always_xy=True).transform((b[0]+b[2])/2,(b[1]+b[3])/2);xy.append([lon,lat])
    coords=np.radians(xy);q=coords[ntrain:]
    dlat=q[:,None,1]-coords[None,:,1];dlon=q[:,None,0]-coords[None,:,0]
    a=np.sin(dlat/2)**2+np.cos(q[:,None,1])*np.cos(coords[None,:,1])*np.sin(dlon/2)**2
    nearby=6371000*2*np.arcsin(np.sqrt(np.clip(a,0,1)))<=640
    jobs=[("random","random")]+[(tag,str(out/"runs"/(tag+".best_unlabeled.pt"))) for tag in ["verified_s2_standard","verified_s2_generic","verified_s2_physical","corrective_scale_uniform","corrective_scale_random","corrective_scale_selective"]]
    if args.include_routes:jobs += [("sensor_"+r,str(out/"runs"/("sensor_"+r+".best_unlabeled.pt"))) for r in ["single","mixed","paired"]]
    jobs += [("DOFA","dofa")]
    results=[]
    for tag,path in jobs:
        model,_,info=encoder(path);features={}
        with torch.no_grad():
            for key,names in [("s2",S2_NAMES),("landsat",LANDSAT_NAMES)]:
                centre=torch.tensor([CENTRE[k] for k in names],device="cuda")
                gsd=torch.tensor([GSD[k] for k in names],device="cuda")
                rows=[]
                for start in range(0,len(ids),32):
                    x=torch.from_numpy(arrays[key][start:start+32]).cuda()
                    if path=="dofa":z=model(F.interpolate(x,size=(224,224),mode="bilinear",align_corners=False),(centre/1000).tolist())[-1].mean((2,3))
                    else:z=model.backbone.encoder(x,centre,gsd)["global"]
                    rows.append(F.normalize(z,dim=1))
                features[key]=torch.cat(rows)
        truth=torch.arange(ntrain,len(ids),device="cuda")
        row=dict(tag=tag,checkpoint=path,S2_to_Landsat=statistics(features["s2"][ntrain:],features["landsat"],truth,nearby),Landsat_to_S2=statistics(features["landsat"][ntrain:],features["s2"],truth,nearby))
        # Scene-cluster resampling describes this single model, not seed uncertainty.
        groups=sorted({r["landsat_scene"] for r in me["rows"]});rng=np.random.default_rng(1234)
        boot=[]
        for _ in range(1000):
            sampled=rng.choice(groups,len(groups),replace=True)
            indices=[i for group in sampled for i,r in enumerate(me["rows"]) if r["landsat_scene"]==group]
            boot.append(float(np.mean(np.asarray(row["S2_to_Landsat"]["per_query_r1"])[indices])))
        row["scene_cluster_R1_interval95"]=np.quantile(boot,[.025,.975]).tolist();row["scene_groups"]=len(groups)
        results.append(row);print(json.dumps(dict(tag=tag,r1=row["S2_to_Landsat"]["recall_at_1"],r10=row["S2_to_Landsat"]["recall_at_10"])),flush=True)
        del model,features;torch.cuda.empty_cache()
    (out/"geographic_retrieval.json").write_text(json.dumps(dict(status="geographic_DEV_diagnostic_not_blind",query_ids=me["ids"],gallery_ids=ids,query_event_counts={ev:sum(r["event_id"]==ev for r in me["rows"]) for ev in sorted({r["event_id"] for r in me["rows"]})},definition="expected hit probability under exact ties; rank is one-based midrank; near640m is a separate tolerance metric",chance=dict(R1=1/len(ids),R10=10/len(ids)),results=results),indent=2))


if __name__=="__main__":main()

