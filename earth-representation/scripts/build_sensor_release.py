"""Freeze near-time, stable, co-registered two-sensor training locations."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.io import netcdf_file
from src.data import triples
from src.data.landslide_patches import _read_patch

def common(a,v):
    a=a[:,:126,:126];v=v[:126,:126]
    weight=v.reshape(42,3,42,3).mean((1,3))
    result=(a*v).reshape(a.shape[0],42,3,42,3).mean((2,4))/np.maximum(weight,1e-8)
    return result,weight>=.999

def corr(a,b,v):
    if v.sum()<30 or a[v].std()<.01 or b[v].std()<.01:return -1.
    return float(np.corrcoef(a[v],b[v])[0,1])

def admit(path,meta):
    sid=Path(path).stem;p=triples.ROOT/"sen12_landsat"/(sid+".nc")
    if not p.exists():return None,"missing_landsat"
    with netcdf_file(str(p),"r",mmap=False) as n:
        gap=int(n.date_gap_days);scene=n.scene_id.decode();date=n.sensing_date.decode()
        if n.s2_date.decode()!=meta["date"] or int(n.s2_time_index)!=meta["post_index"]:
            return None,"metadata_mismatch"
    if gap>2:return None,"gap_over_2_days"
    a,va=_read_patch(Path(path),time_index=meta["post_index"])
    ls=triples._compose(p,triples.LANDSAT_VARS,triples.LANDSAT_NAMES,1)
    b=ls["image"];vb=ls["valid_mask"];valid=va&vb
    if valid.mean()<.8:return None,"joint_QA_under_80_percent"
    aa,ma=common(a[[2,7]],valid);bb,mb=common(b[[3,4]],valid)
    nd0=(aa[1]-aa[0])/np.maximum(aa[1]+aa[0],1e-6)
    nd1=(bb[1]-bb[0])/np.maximum(bb[1]+bb[0],1e-6);v=ma&mb
    zero=corr(nd0,nd1,v);delta=float(np.mean(abs(nd0[v]-nd1[v]))) if v.any() else 1.
    shifted=[]
    for axis in [0,1]:
        for offset in [-1,1]:
            shifted_v=np.roll(v,offset,axis)&v
            if axis==0:shifted_v[0 if offset>0 else -1,:]=False
            else:shifted_v[:,0 if offset>0 else -1]=False
            shifted.append(corr(nd0,np.roll(nd1,offset,axis),shifted_v))
    if zero<.6 or delta>.2 or zero<max(shifted)-.05:return None,"unstable_or_misaligned"
    row=dict(sample_id=sid,event_id=meta["event_id"],landsat_scene=scene,landsat_date=date,gap_days=gap,
        s2_date=meta["date"],valid_fraction=float(valid.mean()),ndvi_correlation=zero,ndvi_abs_difference=delta)
    return (np.where(valid[None],a,0),np.where(valid[None],b,0),valid,row),None


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);ap.add_argument("--limit",type=int,default=0);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    source=json.loads(Path("outputs/verification/runs/verified_s2_standard.provenance.json").read_text())
    records={r["sample_id"]:r for r in [json.loads(l) for l in Path("../landslide-seg/manifests/v4r1/pool_100.jsonl").read_text().splitlines()]}
    rows=[];s2s=[];lss=[];valids=[];excluded=Counter()
    for path in source["members"]:
        accepted,reason=admit(path,records[Path(path).stem])
        if accepted is None:excluded[reason]+=1;continue
        a,b,v,r=accepted;rows.append(r);s2s.append(a);lss.append(b);valids.append(v)
        if args.limit and len(rows)>=args.limit:break
    if len(rows)<4:raise ValueError(f"only {len(rows)} strict pairs; exclusions={dict(excluded)}")
    validation=[];validation_valid=[];evaluation=[];evaluation_excluded=Counter()
    for path in source["validation_members"]:
        a,v=_read_patch(Path(path));validation.append(a);validation_valid.append(v)
        accepted,reason=admit(path,records[Path(path).stem])
        if accepted is None:evaluation_excluded[reason]+=1
        else:evaluation.append(accepted)
    arrays=dict(s2=np.stack(s2s),landsat=np.stack(lss),valid=np.stack(valids),validation=np.stack(validation),validation_valid=np.stack(validation_valid))
    if evaluation:
        arrays.update(evaluation_s2=np.stack([r[0] for r in evaluation]),evaluation_landsat=np.stack([r[1] for r in evaluation]),evaluation_valid=np.stack([r[2] for r in evaluation]))
    np.savez_compressed(out/"arrays.npz",**arrays)
    manifest=dict(schema_version=1,status="engineering" if args.limit else "accepted",train_ids=[r["sample_id"] for r in rows],
        validation_ids=[Path(p).stem for p in source["validation_members"]],rows=rows,excluded=dict(excluded),
        evaluation_ids=[r[3]["sample_id"] for r in evaluation],evaluation_rows=[r[3] for r in evaluation],evaluation_excluded=dict(evaluation_excluded),
        arrays_sha256=hashlib.sha256((out/"arrays.npz").read_bytes()).hexdigest(),
        criteria=dict(max_gap_days=2,joint_valid_min=.8,common_support_m=30,ndvi_corr_min=.6,ndvi_abs_difference_max=.2,registration_tolerance="zero-shift correlation no more than .05 below best ±1 pixel"),
        source_membership="verified_s2_standard.provenance.json",notes="stable textured subset; not representative of all land covers; validation uses frozen S2 development members; paired evaluation uses geographic DEV only")
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(train=len(rows),validation=len(validation),evaluation=len(evaluation),excluded=dict(excluded),evaluation_excluded=dict(evaluation_excluded))))
if __name__=="__main__":main()
