"""Existing official map labels, explicitly supplemental weak supervision."""
import json,hashlib
from pathlib import Path
import numpy as np
import rasterio
import yaml

def overlap(a,b,margin=0):return a[0]<b[2]+margin and b[0]<a[2]+margin and a[1]<b[3]+margin and b[1]<a[3]+margin

def main():
    base=Path("F:/firepaper")
    source=json.loads((base/"data/china_forest_100/manifest.json").read_text())
    regions=yaml.safe_load(Path("configs/regions.yaml").read_text())["regions"]
    accepted=[];excluded=[]
    for row in source["samples"]:
        if any(overlap(row["bounds_wgs84"],r["bbox"],.15) for r in regions.values()):
            excluded.append(row["sample_id"]);continue
        accepted.append(row)
    groups={r["region_key"]:r["region_key"] for r in accepted}
    def find(k):
        while groups[k]!=k:k=groups[k]
        return k
    for i,a in enumerate(accepted):
        for b in accepted[i+1:]:
            if overlap(a["bounds_wgs84"],b["bounds_wgs84"],.1):
                x,y=find(a["region_key"]),find(b["region_key"]);groups[max(x,y)]=min(x,y)
    unique=sorted({find(r["region_key"]) for r in accepted})
    rng=np.random.default_rng(1234);order=rng.permutation(unique).tolist();n=len(order)
    assigned={k:("train" if i<int(.6*n) else "val" if i<int(.8*n) else "test") for i,k in enumerate(order)}
    bands=["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]
    images=[];labels=[];valid=[];rows=[]
    for row in accepted:
        with rasterio.open(base/row["paths"]["pre_fire"]) as f:
            image=f.read([row["band_order"].index(k)+1 for k in bands]).astype("float32")
        with rasterio.open(base/row["paths"]["landcover"]) as f:label=f.read(1)
        with rasterio.open(base/row["paths"]["valid_mask"]) as f:v=f.read(1).astype(bool)
        v &= np.isfinite(image).all(0)&(label!=0)
        images.append(np.where(v[None],image,0));labels.append(label);valid.append(v)
        rows.append(dict(sample_id=row["sample_id"],region_key=row["region_key"],group=find(row["region_key"]),split=assigned[find(row["region_key"])],bounds_wgs84=row["bounds_wgs84"],date=row["acquisition_datetime"]))
    out=Path("outputs/verification/worldcover_data");out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/"arrays.npz",image=np.stack(images),label=np.stack(labels),valid=np.stack(valid))
    manifest=dict(status="supplemental_weak_labels_only",label_source="existing WorldCover map; no forest-mask or synthetic fire labels used",native_input_grid_m=20,original_map_grid_m=10,evaluation_grid_m=20,source_radiometry="existing float32 surface reflectance; no second scaling",rows=rows,excluded_pretraining_neighborhood=excluded,split_groups={k:[g for g,s in assigned.items() if s==k] for k in ["train","val","test"]},split_counts={k:sum(r["split"]==k for r in rows) for k in ["train","val","test"]},arrays_sha256=hashlib.sha256((out/"arrays.npz").read_bytes()).hexdigest(),boundary="forest-dominated local inventory; model-derived map agreement, not field truth or proof of native10m boundary preservation")
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(samples=len(rows),splits=manifest["split_counts"],excluded=excluded)))
if __name__=="__main__":main()
