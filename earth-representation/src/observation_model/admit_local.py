"""Freeze local-only native observations, scene groups and qualified cross targets."""
import collections
import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from rasterio.transform import Affine
from rasterio.warp import reproject
from rasterio.enums import Resampling
from scipy.ndimage import shift
from scipy.optimize import minimize

from .native_data import S2,LANDSAT,optical_from_four_view

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/"observation_model/outputs/phase0"


def rebuild_downloaded_windows():
    """Only already completed files: this function has no network reader."""
    sources=OUT/"recovered_sources";cache=OUT/"native_recovered";cache.mkdir(exist_ok=True)
    candidates={r["sample_id"]:r for r in map(json.loads,(OUT/"pair_candidates.jsonl").read_text().splitlines()) if r["sensor"]=="landsat"}
    failures=[]
    for site in sources.iterdir() if sources.exists() else []:
        row=candidates[site.name];b=row["bbox"];cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2;bbox=[cx-600,cy-600,cx+600,cy+600]
        for sensor in ["S2","L8"]:
            folder=site/sensor
            if not (folder/"product.json").exists():continue
            keys=[key for key,_,_ in (S2 if sensor=="S2" else LANDSAT)]+(["SCL"] if sensor=="S2" else ["qa_pixel","qa_radsat"])
            if any(not (folder/(k+".tif")).exists() for k in keys):
                failures.append(dict(site=site.name,sensor=sensor,reason="incomplete_local_window; downloads paused; no repair fetch"));continue
            try:
                paths={key:dict(file=str((folder/(key+".tif")).relative_to(sources))) for key in keys}
                fake=dict(site_id=site.name,crs=row["crs"],files={"s2_l2a" if sensor=="S2" else "landsat_l2":paths})
                arrays,meta=optical_from_four_view(sources,fake,sensor,bbox)
                from pyproj import Transformer
                t=Transformer.from_crs(meta["crs"],4326,always_xy=True);ll=t.transform(bbox[0],bbox[1]);ur=t.transform(bbox[2],bbox[3])
                path=cache/(site.name+"_"+sensor+".npz");np.savez_compressed(path,**arrays)
                meta.update(source_family="official_native_recovered_existing_6309",role_constraint="candidate",region=row["region"],geographic_bbox=[*ll,*ur],cache=str(path),cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                path.with_suffix(".json").write_text(json.dumps(meta,indent=2))
            except Exception as exc:failures.append(dict(site=site.name,sensor=sensor,reason=str(exc)))
    (OUT/"local_rebuild_exclusions.json").write_text(json.dumps(failures,indent=2))


def acquisition_key(row):
    name=row["product_id"]
    if row["sensor"]=="S2":
        match=re.search(r"(S2[ABC]).*?(\d{8}T\d{6}).*?T(\d{2}[A-Z]{3})",name)
        if match:return "_".join(match.groups())
    else:
        match=re.search(r"(LC\d{2})_L2\w{2}_(\d{6})_(\d{8})",name)
        if match:return "_".join(match.groups())
    raise ValueError(f"cannot establish original acquisition identity: {name}")


def common_band(meta,arrays,name,destination):
    band=next(b for b in meta["bands"] if b["name"]==name)
    target=destination["bands"][0];shape=(40,40)
    value=np.zeros(shape,np.float32);coverage=np.zeros(shape,np.float32)
    arguments=dict(src_transform=Affine(*band["transform"]),src_crs=band["crs"],dst_transform=Affine(*target["transform"]),dst_crs=target["crs"],resampling=Resampling.average)
    reproject(arrays[name],value,**arguments)
    reproject(arrays[name+"_valid"].astype(np.float32),coverage,**arguments)
    return value/coverage.clip(1e-6),coverage>=.8


def pair_quality(a,b):
    with np.load(a["cache"]) as z:s2={k:z[k] for k in z.files}
    with np.load(b["cache"]) as z:ls={k:z[k] for k in z.files}
    red,rv=common_band(a,s2,"S2:B04",b);nir,nv=common_band(a,s2,"S2:B8A",b)
    valid=rv&nv&ls["L8:SR_B4_valid"]&ls["L8:SR_B5_valid"]
    x=(nir-red)/(nir+red+1e-6);y=(ls["L8:SR_B5"]-ls["L8:SR_B4"])/(ls["L8:SR_B5"]+ls["L8:SR_B4"]+1e-6)
    corr=float(np.corrcoef(x[valid],y[valid])[0,1]) if valid.sum()>10 and min(x[valid].std(),y[valid].std())>1e-6 else None
    difference=float(np.abs(x[valid]-y[valid]).mean()) if valid.any() else None
    gray=ls["L8:SR_B4"]
    texture=min(float(red[valid].std()),float(gray[valid].std())) if valid.any() else 0.
    inner=valid.copy();inner[:2]=False;inner[-2:]=False;inner[:,:2]=False;inner[:,-2:]=False
    def objective(delta):
        shifted=shift(red,delta,order=1,mode="constant",cval=0,prefilter=False)
        u=shifted[inner];v=gray[inner]
        u=u-u.mean();v=v-v.mean()
        return -float((u*v).sum()/max(np.sqrt((u*u).sum()*(v*v).sum()),1e-10))
    residual=None;registration_score=None
    if inner.sum()>64 and texture>.001:
        trial=minimize(objective,np.zeros(2),method="Powell",bounds=[(-1.5,1.5)]*2,options=dict(xtol=.02,ftol=1e-4,maxiter=30))
        residual=float(np.linalg.norm(trial.x)*30);registration_score=-float(trial.fun)
    gap=abs((dt.date.fromisoformat(a["date"][:10])-dt.date.fromisoformat(b["date"][:10])).days)
    broad=gap<=2 and valid.mean()>=.8
    strict=broad and residual is not None and residual<=15 and registration_score>=.6 and corr is not None and corr>=.6 and difference<=.2
    return dict(gap_days=gap,joint_valid_fraction=float(valid.mean()),ndvi_correlation=corr,ndvi_absolute_difference=difference,registration_residual_m=residual,registration_correlation=registration_score,broad_eligible=bool(broad),strict_eligible=bool(strict),target_grid=dict(transform=b["bands"][0]["transform"],crs=b["crs"],shape=[40,40],gsd_m=30))


def main():
    existing=OUT/"release_local_v1/release.json"
    if existing.exists() and json.loads(existing.read_text()).get("status")=="ACCEPTED_LOCAL_NATIVE_FOR_MASKED":
        print("Existing accepted native release is frozen; no files changed",flush=True);return
    config=json.loads((ROOT/"observation_model/config.json").read_text())
    if config["download_enabled"]:raise ValueError("this local-only release expects the user's download pause")
    rebuild_downloaded_windows()
    observations=[];exclusions=[]
    for folder in ["native_local","native_recovered"]:
        for path in sorted((OUT/folder).glob("*.json")):
            r=json.loads(path.read_text());r["observation_id"]=r["source_site"]+"_"+r["sensor"]
            if min(b["valid_fraction"] for b in r["bands"])<.5:
                exclusions.append(dict(id=r["observation_id"],reason="per-band valid coverage below50%"));continue
            observations.append(r)
    observations=list({(r["product_id"],tuple(r["bbox"]),r["sensor"]):r for r in observations}.values())
    sites=sorted({r["source_site"] for r in observations});parent={s:s for s in sites}
    def find(s):
        while parent[s]!=s:parent[s]=parent[parent[s]];s=parent[s]
        return s
    def merge(a,b):
        a,b=find(a),find(b);parent[max(a,b)]=min(a,b)
    seen={};cells={}
    for r in observations:
        sid=r["source_site"];r["acquisition_id"]=acquisition_key(r);key=(r["sensor"],r["acquisition_id"])
        if key in seen:merge(sid,seen[key])
        seen[key]=sid
        box=r["geographic_bbox"];lon=(box[0]+box[2])/2;lat=(box[1]+box[3])/2
        cell=(math.floor(lon),math.floor(lat))
        if cell in cells:merge(sid,cells[cell])
        cells[cell]=sid
    # Also merge nearby sites across arbitrary1degree cell boundaries.
    centers={r["source_site"]:((r["geographic_bbox"][0]+r["geographic_bbox"][2])/2,(r["geographic_bbox"][1]+r["geographic_bbox"][3])/2) for r in observations}
    for i,a in enumerate(sites):
        for b in sites[i+1:]:
            x,y=centers[a];xx,yy=centers[b]
            if np.hypot((x-xx)*np.cos(np.deg2rad((y+yy)/2)),y-yy)*111000<10000:merge(a,b)
    groups=sorted({find(s) for s in sites});rng=np.random.default_rng(0);rng.shuffle(groups)
    assigned={g:("train" if i<int(.7*len(groups)) else "val" if i<int(.85*len(groups)) else "test") for i,g in enumerate(groups)}
    constrained={find(r["source_site"]) for r in observations if r["role_constraint"]=="heldout"}
    for g in constrained:
        if assigned[g]=="train":assigned[g]="val"
    for r in observations:r.update(group=find(r["source_site"]),split=assigned[find(r["source_site"])] )
    by_site=collections.defaultdict(dict)
    for r in observations:by_site[r["source_site"]][r["sensor"]]=r
    pairs=[]
    for i,(sid,sensors) in enumerate(by_site.items()):
        if "S2" not in sensors or "L8" not in sensors:continue
        a,b=sensors["S2"],sensors["L8"]
        q=pair_quality(a,b)
        pairs.append(dict(site_id=sid,split=a["split"],group=a["group"],s2_id=a["observation_id"],landsat_id=b["observation_id"],s2_scene=a["product_id"],landsat_scene=b["product_id"],**q))
        if i%30==0:print("pair quality",i,"of",len(by_site),flush=True)
    release=OUT/"release_local_v1";release.mkdir(exist_ok=True)
    for name,rows in [("single_masked.jsonl",observations),("cross30.jsonl",pairs)]:
        (release/name).write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in rows))
    (release/"exclusions.json").write_text(json.dumps(exclusions,indent=2))
    summary=dict(status="frozen_local_release_pending_final_checks",download_status="USER_PAUSED",observations=dict(collections.Counter(r["split"]+"/"+r["sensor"] for r in observations)),groups=dict(collections.Counter(assigned.values())),strict_pairs=dict(collections.Counter(r["split"] for r in pairs if r["strict_eligible"])),strict_pair_regions={split:len({r["group"] for r in pairs if r["split"]==split and r["strict_eligible"]}) for split in ["train","val","test"]},broad_pairs=dict(collections.Counter(r["split"] for r in pairs if r["broad_eligible"])),hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in release.glob("*.jsonl")},boundary="local-only native data; primary6309 legacy caches excluded as quantitative truth; no recovery of nonconforming data after user pause")
    (release/"release.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=="__main__":main()
