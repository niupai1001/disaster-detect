"""Inventory every cached location/time without treating interpolation as native data."""
import collections
import datetime as dt
import json
import re
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.io import netcdf_file

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parent
OUTPUT=ROOT/"observation_model/outputs/phase0"
BAD=(0,1,3,8,9,10,11,255)


def text(value):
    return value.decode() if isinstance(value,bytes) else str(value)


def dates_from_time(dataset):
    units=text(dataset.attrs["units"])
    match=re.fullmatch(r"days since (\d{4}-\d{2}-\d{2})(?:.*)?",units)
    if not match:raise ValueError(f"unsupported actual time units: {units}")
    origin=dt.date.fromisoformat(match[1])
    return [origin+dt.timedelta(days=float(x)) for x in dataset[:]]


def companion(path):
    with netcdf_file(str(path),"r",mmap=False) as nc:
        attrs={k:text(v) for k,v in nc._attributes.items()}
        qa=np.array(nc.variables["QA_PIXEL" if "QA_PIXEL" in nc.variables else "Fmask"][:])
        if "QA_PIXEL" in nc.variables:
            valid=(qa.astype(np.uint16)&63)==0
            if "QA_RADSAT" in nc.variables:valid &= np.array(nc.variables["QA_RADSAT"][:])==0
        else:valid=(qa!=255)&((qa.astype(np.uint8)&30)==0)
        first=next(v for k,v in nc.variables.items() if k.startswith("SR_B") or re.fullmatch(r"B\d\w?",k))
        return attrs,valid,str(first.data.dtype)


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/"observation_model/config.json").read_text())
    geometry={}
    for name in ["s2_10band_v4_p20.jsonl","v4r1/pool_100.jsonl"]:
        p=WORKSPACE/"landslide-seg/manifests"/name
        for line in p.read_text(encoding="utf-8").splitlines():
            row=json.loads(line);geometry[row["sample_id"]]=row
    legacy={k:{p.stem:p for p in (ROOT/"data"/folder).glob("*.nc")} for k,folder in [("landsat","sen12_landsat"),("hls","sen12_hls")]}
    ids=sorted(set(legacy["landsat"])|set(legacy["hls"]))
    regions=collections.Counter();scenes={k:set() for k in legacy};gaps=collections.Counter();roles=collections.Counter();errors=[]
    t0=time.time();observations=0;pair_n=0
    state=OUTPUT/"status.json"
    def status(done,phase):
        state.write_text(json.dumps(dict(phase=phase,locations_done=done,locations_total=len(ids),s2_observations=observations,pair_candidates=pair_n,seconds=time.time()-t0),indent=2))
    status(0,"inventory_running")
    with (OUTPUT/"locations.jsonl").open("w",encoding="utf-8") as locations,(OUTPUT/"single_observations.jsonl").open("w",encoding="utf-8") as singles,(OUTPUT/"pair_candidates.jsonl").open("w",encoding="utf-8") as pairs:
        for index,sid in enumerate(ids):
            try:
                region=sid.rsplit("_s2_",1)[0];regions[region]+=1
                source=WORKSPACE/"landslide-seg/data/sen12"/(sid+".nc")
                geo=geometry.get(sid,{})
                with h5py.File(source,"r") as f:
                    dates=dates_from_time(f["time"]);scl=f["SCL"][:]
                    quality=~np.isin(scl,BAD)
                    fraction=quality.mean((1,2))
                    center=[float(np.asarray(f.attrs[k]).reshape(-1)[0]) for k in ["center_lon","center_lat"]]
                role="protected" if region in config["protected_regions"] else "candidate"
                roles[role]+=1
                row=dict(sample_id=sid,region=region,role=role,source=str(source),bbox=geo.get("bbox"),crs=geo.get("crs"),source_center_attributes=center,center_attribute_warning="legacy names center_lon/lat actually contain projected coordinates; use verified CRS/bbox",dates=[d.isoformat() for d in dates],s2_valid_fractions=fraction.tolist(),native_admitted=False,native_exclusion="multitemporal derived cache; original product and native band grids require recovery")
                locations.write(json.dumps(row)+"\n")
                for frame,date in enumerate(dates):
                    singles.write(json.dumps(dict(sample_id=sid,sensor="S2",date=date.isoformat(),time_index=frame,role=role,valid_fraction=float(fraction[frame]),source=str(source),original_product_id=None,native_admitted=False))+"\n");observations+=1
                for sensor,paths in legacy.items():
                    if sid not in paths:continue
                    attrs,valid,dtype=companion(paths[sid]);scene=attrs["scene_id"];scenes[sensor].add(scene)
                    acquired=dt.date.fromisoformat(attrs["sensing_date"][:10]);frame=min(range(len(dates)),key=lambda j:abs((dates[j]-acquired).days));gap=abs((dates[frame]-acquired).days)
                    joint=float((quality[frame].T & valid).mean())
                    candidate=dict(sample_id=sid,region=region,role=role,sensor=sensor,source=str(paths[sid]),source_product=scene,date=acquired.isoformat(),s2_date=dates[frame].isoformat(),s2_time_index=frame,gap_days=gap,joint_cached_QA_fraction=joint,dtype=dtype,native_gsd_m=attrs.get("native_gsd_m"),stored_grid_m=attrs.get("resampled_to_m"),schema=attrs.get("schema_version"),native_admitted=False,reason="legacy int16 resampled cache; QA/time candidate only, not reflectance truth",bbox=geo.get("bbox"),crs=geo.get("crs"))
                    pairs.write(json.dumps(candidate)+"\n");pair_n+=1
                    if sensor=="landsat":gaps["within_2d" if gap<=2 else "over_2d"]+=1;gaps["within_2d_and_cached_joint_QA80" if gap<=2 and joint>=.8 else "not_joint_time_QA_candidate"]+=1
            except Exception as exc:errors.append(dict(sample_id=sid,error=repr(exc)))
            if (index+1)%100==0:status(index+1,"inventory_running");print(f"inventory {index+1}/{len(ids)}",flush=True)
    summary=dict(locations=len(ids),regions=dict(regions),roles=dict(roles),s2_actual_time_entries=observations,unique_source_product_ids={k:len(v) for k,v in scenes.items()},landsat_nearest_real_time=dict(gaps),errors=errors,native_admitted=0,note="Candidate count is not native or calibrated admission. No interpolated cache promoted to native truth.")
    (OUTPUT/"inventory_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    status(len(ids),"inventory_complete_pending_native_admission")
    print(json.dumps({k:v for k,v in summary.items() if k!="errors"},ensure_ascii=False),flush=True)
    print("errors",len(errors),errors[:2],flush=True)


if __name__=="__main__":main()
