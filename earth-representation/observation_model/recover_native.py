"""Recover missing true native windows for eligible existing near-time positions."""
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

import numpy as np
from pyproj import Transformer
import rasterio

from .metered_http import DownloadBudget
from .native_data import S2,LANDSAT,native_window,optical_from_four_view

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/"observation_model/outputs/phase0"
PC="https://planetarycomputer.microsoft.com/api/stac/v1"


def clean_item(item):
    return dict(id=item["id"],properties=item["properties"],assets={k:{p:(v.split("?")[0] if p=="href" else v) for p,v in a.items()} for k,a in item["assets"].items()})


class Recovery:
    def __init__(self):
        self.budget=DownloadBudget(OUT/"download_budget.json",10737418240);self.tokens={}
        self.catalogues={};self.root=OUT/"recovered_sources";self.root.mkdir(exist_ok=True)

    def sign(self,url):
        parts=urlsplit(url);account=parts.hostname.split(".")[0];container=parts.path.split("/")[1];key=(account,container)
        if key not in self.tokens:self.tokens[key]=self.budget.json_request("GET",f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}")["token"]
        return url+"?"+self.tokens[key]

    def feature(self,row,sensor):
        if sensor=="L8":return self.budget.json_request("GET",PC+"/collections/landsat-c2-l2/items/"+row["source_product"])
        b=row["bbox"];transform=Transformer.from_crs(row["crs"],4326,always_xy=True)
        left,bottom=transform.transform(b[0],b[1]);right,top=transform.transform(b[2],b[3])
        result=self.budget.json_request("POST",PC+"/search",json=dict(collections=["sentinel-2-l2a"],bbox=[left,bottom,right,top],datetime=row["s2_date"]+"T00:00:00Z/"+row["s2_date"]+"T23:59:59Z",limit=20))
        features=result["features"]
        if not features:raise ValueError("no official S2 L2A product at actual existing observation date")
        return min(features,key=lambda f:f["properties"].get("eo:cloud_cover",100))

    def window(self,feature,key,bbox,gsd,path,bbox_crs):
        if path.exists():return
        url=self.sign(feature["assets"][key]["href"])
        def opener(name,mode="r"):
            if str(name)!="band.tif":raise FileNotFoundError(name)
            return self.budget.open(url)
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
            value,valid,transform,crs=native_window("band.tif",bbox,gsd,opener,bbox_crs=bbox_crs)
        path.parent.mkdir(parents=True,exist_ok=True)
        nodata=None if key in ["qa_pixel","qa_radsat"] else 0
        with rasterio.open(path,"w",driver="GTiff",height=value.shape[0],width=value.shape[1],count=1,dtype=value.dtype,crs=crs,transform=transform,compress="deflate",nodata=nodata) as dst:
            dst.write(value,1)

    def sensor(self,row,sensor,bbox):
        site=row["sample_id"];folder=self.root/site/sensor;folder.mkdir(parents=True,exist_ok=True)
        metadata=folder/"product.json"
        feature=json.loads(metadata.read_text()) if metadata.exists() else self.feature(row,sensor)
        if sensor=="S2" and float(feature["properties"]["s2:processing_baseline"])>=4:
            raise ValueError("PB>=04 exact BOA metadata recovery pending; leave out of quantitative release")
        if not metadata.exists():metadata.write_text(json.dumps(clean_item(feature),indent=2))
        bands=S2 if sensor=="S2" else LANDSAT
        keys=[(key,g) for key,_,g in bands]+([("SCL",20)] if sensor=="S2" else [("qa_pixel",30),("qa_radsat",30)])
        paths={}
        for key,gsd in keys:
            path=folder/(key+".tif");self.window(feature,key,bbox,gsd,path,row["crs"])
            paths[key]=dict(file=str(path.relative_to(self.root)))
        fake=dict(site_id=site,crs=row["crs"],files={"s2_l2a" if sensor=="S2" else "landsat_l2":paths})
        arrays,meta=optical_from_four_view(self.root,fake,sensor,bbox)
        if min(b["valid_fraction"] for b in meta["bands"])<.5:raise ValueError("official native QA coverage below50%")
        cache=OUT/"native_recovered";cache.mkdir(exist_ok=True);path=cache/(site+"_"+sensor+".npz")
        np.savez_compressed(path,**arrays)
        transform=Transformer.from_crs(meta["crs"],4326,always_xy=True);ll=transform.transform(bbox[0],bbox[1]);ur=transform.transform(bbox[2],bbox[3])
        meta.update(role_constraint="candidate",geographic_bbox=[*ll,*ur],source_family="official_native_recovered_existing_6309",region=row["region"],cache=str(path),cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_manifest="pair_candidates.jsonl")
        path.with_suffix(".json").write_text(json.dumps(meta,indent=2));return meta


def main():
    config=json.loads((ROOT/"observation_model/config.json").read_text())
    if not config.get("download_enabled",False):
        print("DOWNLOADS_PAUSED_BY_USER: local inventory/admission only",flush=True)
        return
    recovery=Recovery()
    rows=[json.loads(s) for s in (OUT/"pair_candidates.jsonl").read_text().splitlines()]
    rows=[r for r in rows if r["sensor"]=="landsat" and r["role"]=="candidate" and r["gap_days"]<=2 and r.get("bbox")]
    # Round-robin regions: prioritize coverage rather than exhausting one scene.
    rows=sorted(rows,key=lambda r:(sum(x["region"]==r["region"] and x["sample_id"]<r["sample_id"] for x in rows),r["region"]))
    errors=[];done=0
    for row in rows:
        b=row["bbox"];cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2;bbox=[cx-600,cy-600,cx+600,cy+600]
        for sensor in ["S2","L8"]:
            existing=OUT/"native_recovered"/(row["sample_id"]+"_"+sensor+".json")
            if existing.exists():continue
            try:recovery.sensor(row,sensor,bbox)
            except Exception as exc:
                errors.append(dict(sample_id=row["sample_id"],sensor=sensor,error=str(exc).split("?")[0]))
                if "DOWNLOAD_BUDGET_EXHAUSTED" in str(exc):break
        done+=1
        state=dict(status="recovering_native_windows",locations_done=done,locations_total=len(rows),received_bytes=recovery.budget.received,cap_bytes=recovery.budget.cap,errors=len(errors))
        (OUT/"native_recovery_status.json").write_text(json.dumps(state,indent=2));(OUT/"native_recovery_errors.json").write_text(json.dumps(errors,indent=2))
        print(json.dumps(state),flush=True)
        if recovery.budget.cap-recovery.budget.received<262144:break
    state["status"]="native_recovery_complete_pending_admission" if done==len(rows) else "budget_exhausted_pending_admission"
    (OUT/"native_recovery_status.json").write_text(json.dumps(state,indent=2))


if __name__=="__main__":main()
