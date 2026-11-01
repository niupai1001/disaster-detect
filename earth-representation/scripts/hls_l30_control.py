"""Official L30 provenance and matched Landsat C2 comparison on a common grid."""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from scipy.io import netcdf_file

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import _common as C
from src.data.adapters import landsat_c2,_base
from src.data.adapters.hls_v2 import _token,_header
from src.data.qa_masks import landsat_qa_masks,hls_fmask_masks


def main():
    out=ROOT/"outputs/verification"
    reference=np.load(out/"same_acquisition_arrays.npz")
    grid=Affine(*reference["transform"][:6]);height,width=reference["valid"].shape
    from rasterio.transform import array_bounds
    bounds=array_bounds(height,width,grid);crs="EPSG:32736"
    bbox=transform_bounds(crs,"EPSG:4326",*bounds)
    p=next((ROOT/"data/companions_v2_selected_time/sen12_landsat").glob("chimanimani*.nc"))
    with netcdf_file(str(p),"r",mmap=False) as n:source_id=n.scene_id.decode();date=n.sensing_date.decode()[:10]
    source_fields=source_id.split("_");identity=[source_fields[i] for i in [0,2,3]]
    token=_token()
    candidates,_=C.search("HLSL30_2.0",bbox,date,date,limit=100)
    wanted_tile=json.loads((out/"same_acquisition_control.json").read_text())["hls_product"].split(".")[2]
    matched=None
    for item in candidates:
        if item["id"].split(".")[2] != wanted_tile:continue
        response=C.requests.get(item["assets"]["metadata"]["href"],headers={"Authorization":"Bearer "+token},timeout=60)
        response.raise_for_status();root=ET.fromstring(response.text)
        attrs={}
        for el in root.iter():
            if el.tag.split("}")[-1]=="AdditionalAttribute":
                names=[x.text for x in el.iter() if x.tag.split("}")[-1]=="Name"]
                vals=[x.text for x in el.iter() if x.tag.split("}")[-1]=="Value"]
                if names and vals:attrs[names[0]]=vals[0]
        original=attrs.get("LANDSAT_PRODUCT_ID","")
        if all(x in original for x in identity):matched=(item,original);break
    if matched is None:raise ValueError("no L30 whose original Landsat product matches platform/path-row/date")
    item,original=matched
    scenes=landsat_c2.scenes(bbox,date,date,limit=100)
    scene=next(s for s in scenes if all(x in s.scene_id for x in identity))
    s={k:_base.read_grid(scene.assets[k],grid,crs,(height,width),resampling=Resampling.nearest if k.startswith("qa_") else Resampling.average) for k in ["red","nir08","qa_pixel","qa_radsat"]}
    h={k:_base.read_grid(item["assets"][k]["href"],grid,crs,(height,width),header=str(_header(token)),resampling=Resampling.nearest if k=="Fmask" else Resampling.average) for k in ["B04","B05","Fmask"]}
    r=s["red"]*.0000275-.2;n=s["nir08"]*.0000275-.2
    a=(n-r)/np.maximum(n+r,1e-6)
    hr=h["B04"]*.0001;hn=h["B05"]*.0001;b=(hn-hr)/np.maximum(hn+hr,1e-6)
    valid=landsat_qa_masks(s["qa_pixel"],s["qa_radsat"])["valid"]&hls_fmask_masks(h["Fmask"])["valid"]
    valid &= np.isfinite(a)&np.isfinite(b)&(s["red"]!=0)&(h["B04"]!=-9999)&(h["B05"]!=-9999)
    debug=dict(hls_l30_product=item["id"],hls_source_product=original,landsat_l2_product=scene.scene_id,
        landsat_valid_fraction=float(landsat_qa_masks(s["qa_pixel"],s["qa_radsat"])["valid"].mean()),
        hls_valid_fraction=float(hls_fmask_masks(h["Fmask"])["valid"].mean()),joint_fraction=float(valid.mean()),
        landsat_qa_unique=np.unique(s["qa_pixel"]).tolist(),hls_qa_unique=np.unique(h["Fmask"]).tolist())
    (out/"hls_l30_debug.json").write_text(json.dumps(debug,indent=2))
    np.savez_compressed(out/"hls_l30_debug_arrays.npz",landsat_ndvi=a,l30_ndvi=b,valid=valid,transform=np.array(tuple(grid)))
    windows=[]
    for y in range(0,height-42,42):
        for x in range(0,width-42,42):
            v=valid[y:y+42,x:x+42];aa=a[y:y+42,x:x+42];bb=b[y:y+42,x:x+42]
            if v.mean()>=.8 and aa[v].std()>.01 and bb[v].std()>.01:windows.append((aa,bb,v,y,x))
    windows=windows[:20]
    if len(windows)<20:raise ValueError(f"only {len(windows)} QA-valid windows")
    rows=[]
    for i,(aa,bb,v,y,x) in enumerate(windows):
        def cor(z,m):return float(np.corrcoef(aa[m],z[m])[0,1])
        m=v&np.roll(v,1,1);m[:,:1]=False
        wrong=windows[(i+7)%len(windows)]
        rows.append(dict(window=[y,x,42,42],valid_fraction=float(v.mean()),same=cor(bb,v),shift1=cor(np.roll(bb,1,1),m),wrong=cor(wrong[1],v&wrong[2])))
    medians={k:float(np.median([r[k] for r in rows])) for k in ["same","shift1","wrong"]}
    result=dict(status="accepted" if medians["same"]>max(medians["shift1"],medians["wrong"]) else "failed",hls_l30_product=item["id"],hls_source_product=original,landsat_l2_product=scene.scene_id,matched_fields=identity,independent_acquisitions=1,common_support_m=30,medians=medians,rows=rows)
    (out/"hls_l30_control.json").write_text(json.dumps(result,indent=2))
    np.savez_compressed(out/"hls_l30_control_arrays.npz",landsat_ndvi=a,l30_ndvi=b,valid=valid,transform=np.array(tuple(grid)))
    print(json.dumps(result))


if __name__=="__main__":main()
