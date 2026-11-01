"""Raw S2/Landsat versus HLS S30/L30 on the same 20 common-band windows."""
import json
import sys
from pathlib import Path

import numpy as np
from affine import Affine
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import _common as C
from src.data.adapters import sentinel2,landsat_c2,_base
from src.data.adapters.hls_v2 import _token,_header
from scripts.cross_sensor_eval import build,retrieval_stats
from scripts.frozen_segmentation_probe import encoder


def prepare():
    out=ROOT/"outputs/verification"
    path=out/"hls_feature_inputs.npz"
    if path.exists():return
    a=np.load(out/"same_acquisition_arrays.npz");b=np.load(out/"hls_l30_control_arrays.npz")
    grid=Affine(*a["transform"][:6]);common=a["valid"]&b["valid"]
    height,width=common.shape;windows=[]
    for y in range(0,height-42,42):
        for x in range(0,width-42,42):
            v=common[y:y+42,x:x+42]
            if v.mean()>=.8 and a["s2_ndvi"][y:y+42,x:x+42][v].std()>.01:windows.append([y,x,42,42])
    windows=windows[:20]
    if len(windows)<20:raise ValueError("insufficient four-product common windows")
    y0=min(w[0] for w in windows);x0=min(w[1] for w in windows)
    y1=max(w[0]+42 for w in windows);x1=max(w[1]+42 for w in windows)
    target=grid*Affine.translation(x0,y0);shape=(y1-y0,x1-x0);crs="EPSG:32736"
    bbox=transform_bounds(crs,"EPSG:4326",*array_bounds(*shape,target))
    ms=json.loads((out/"same_acquisition_control.json").read_text());ml=json.loads((out/"hls_l30_control.json").read_text())
    sd=ms["hls_source_product"].split("_")[2][:8];sd=sd[:4]+"-"+sd[4:6]+"-"+sd[6:]
    ld=ml["landsat_l2_product"].split("_")[3];ld=ld[:4]+"-"+ld[4:6]+"-"+ld[6:]
    s=next(s for s in sentinel2.scenes(bbox,sd,sd) if (s.extra.get("product_uri") or s.scene_id)==ms["s2_l2a_product"])
    l=next(s for s in landsat_c2.scenes(bbox,ld,ld) if s.scene_id==ml["landsat_l2_product"])
    hitems={k:C.get_json(C.CMR_STAC+"/collections/"+collection+"/items/"+item) for k,collection,item in [("s30","HLSS30_2.0",ms["hls_product"]),("l30","HLSL30_2.0",ml["hls_l30_product"])]}
    header=str(_header(_token()))
    keys={"s2":["B02","B03","B04","B8A","B11","B12"],"landsat":["blue","green","red","nir08","swir16","swir22"],"s30":["B02","B03","B04","B8A","B11","B12"],"l30":["B02","B03","B04","B05","B06","B07"]}
    stacks={}
    for key in keys:
        assets=s.assets if key=="s2" else l.assets if key=="landsat" else {k:v["href"] for k,v in hitems[key]["assets"].items()}
        arrays=np.stack([_base.read_grid(assets[k],target,crs,shape,header=header if key in ("s30","l30") else "",resampling=Resampling.average) for k in keys[key]]).astype("float32")
        if key=="s2":arrays=(arrays+s.extra["boa_offset"])/s.extra["quantification"]
        elif key=="landsat":arrays=arrays*.0000275-.2
        else:arrays*=.0001
        stacks[key]=np.stack([np.where(common[y:y+42,x:x+42][None],arrays[:,y-y0:y-y0+42,x-x0:x-x0+42],0) for y,x,_,_ in windows])
    np.savez_compressed(path,**stacks)
    (out/"hls_feature_input_manifest.json").write_text(json.dumps(dict(windows=windows,common_grid_m=30,products=dict(s2=ms["s2_l2a_product"],landsat=ml["landsat_l2_product"],s30=ms["hls_product"],l30=ml["hls_l30_product"]),bands=keys,independent_acquisitions=2,notes="20 windows from one geographic area; coordinates shared, virtual products not independent observations"),indent=2))


def main():
    prepare();out=ROOT/"outputs/verification";arrays=np.load(out/"hls_feature_inputs.npz")
    s2waves=[492.4,559.8,664.6,864.7,1613.7,2202.4]
    lswaves=[481.59,561.25,654.59,864.67,1608.86,2200.73]
    results=[]
    for name in ["spectral_statistics","verified_CNN","DOFA"]:
        fs={}
        if name=="verified_CNN":
            model=build().cuda().eval();model.load_state_dict(torch.load(out/"runs/verified_s2_standard.best_unlabeled.pt",map_location="cuda",weights_only=False)["teacher"])
        elif name=="DOFA":model,_,_=encoder("dofa")
        with torch.no_grad():
            for key in ["s2","landsat","s30","l30"]:
                x=torch.from_numpy(arrays[key]).cuda();waves=s2waves if key in ("s2","s30") else lswaves
                if name=="spectral_statistics":
                    z=torch.cat([x.mean((2,3)),x.std((2,3))],1)
                elif name=="verified_CNN":
                    gsd=[10,10,10,20,20,20] if key=="s2" else [30]*6
                    z=model.backbone.encoder(x,torch.tensor(waves,device="cuda"),torch.tensor(gsd,device="cuda"))["global"]
                else:z=model(F.interpolate(x,size=(224,224),mode="bilinear",align_corners=False),[v/1000 for v in waves])[-1].mean((2,3))
                fs[key]=F.normalize(z,dim=1)
        results.append(dict(model=name,raw=retrieval_stats(fs["s2"],fs["landsat"]),harmonized=retrieval_stats(fs["s30"],fs["l30"])))
        if name!="spectral_statistics":del model
    (out/"hls_feature_reference.json").write_text(json.dumps(dict(status="small_engineering_reference_only",results=results),indent=2));print(json.dumps(results))


if __name__=="__main__":main()
