"""S2 L2A / HLS S30 same-acquisition control, official product IDs required."""
import json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling
from scipy.io import netcdf_file
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parent))
import _common as C
from src.data.adapters import sentinel2,_base
from src.data.adapters.hls_v2 import _token,_header,utm_epsg_from_tile
from src.data.qa_masks import s2_scl_masks,hls_fmask_masks

def main():
 out=Path("outputs/verification")
 source=next(Path("data/companions_v2_selected_time/sen12_hls").glob("chimanimani*.nc"))
 with netcdf_file(str(source),"r",mmap=False) as n: sid=n.scene_id.decode()
 item=C.get_json(C.CMR_STAC+"/collections/HLSS30_2.0/items/"+sid)
 token=_token();response=C.requests.get(item["assets"]["metadata"]["href"],headers={"Authorization":"Bearer "+token},timeout=60);response.raise_for_status()
 root=ET.fromstring(response.text);attrs={}
 for el in root.iter():
  if el.tag.split("}")[-1]=="AdditionalAttribute":
   names=[x.text for x in el.iter() if x.tag.split("}")[-1]=="Name"]
   vals=[x.text for x in el.iter() if x.tag.split("}")[-1]=="Value"]
   if names and vals:attrs[names[0]]=vals[0]
 uri=attrs["PRODUCT_URI"];parts=uri.split("_");acquisition=parts[2];platform=parts[0];orbit=next(x for x in parts if x.startswith("R") and len(x)==4);tile=next(x for x in parts if x.startswith("T") and len(x)==6)
 date=acquisition[:4]+"-"+acquisition[4:6]+"-"+acquisition[6:8]
 records=[json.loads(s) for s in Path("../landslide-seg/manifests/v4r1/pool_100.jsonl").read_text().splitlines() if s.strip()]
 records=[r for r in records if r["event_id"]=="chimanimani"]
 bbox=[min(r["bbox"][0] for r in records),min(r["bbox"][1] for r in records),max(r["bbox"][2] for r in records),max(r["bbox"][3] for r in records)]
 crs=records[0]["crs"];wgs=transform_bounds(crs,"EPSG:4326",*bbox)
 scenes=sentinel2.scenes(wgs,date,date)
 matches=[s for s in scenes if all(x in (s.extra.get("product_uri") or s.scene_id) for x in [platform,acquisition,orbit,tile])]
 if not matches:raise ValueError("no L2A matching HLS source platform/acquisition/orbit/tile")
 scene=matches[0]
 left=np.floor(bbox[0]/30)*30;top=np.ceil(bbox[3]/30)*30
 width=int(np.ceil((bbox[2]-left)/30));height=int(np.ceil((top-bbox[1])/30));grid=from_origin(left,top,30,30)
 s={k:_base.read_grid(scene.assets[k],grid,crs,(height,width),resampling=Resampling.nearest if k=="SCL" else Resampling.average) for k in ["B04","B8A","SCL"]}
 h={k:_base.read_grid(item["assets"][k]["href"],grid,crs,(height,width),header=str(_header(token)),resampling=Resampling.nearest if k=="Fmask" else Resampling.average) for k in ["B04","B8A","Fmask"]}
 sr={k:(s[k].astype("float32")+scene.extra["boa_offset"])/scene.extra["quantification"] for k in ["B04","B8A"]}
 hr={k:h[k].astype("float32")*.0001 for k in ["B04","B8A"]}
 valid=s2_scl_masks(s["SCL"])["valid"] & hls_fmask_masks(h["Fmask"])["valid"]
 a=(sr["B8A"]-sr["B04"])/np.maximum(sr["B8A"]+sr["B04"],1e-6);b=(hr["B8A"]-hr["B04"])/np.maximum(hr["B8A"]+hr["B04"],1e-6)
 valid &= np.isfinite(a)&np.isfinite(b)&(h["B04"]!=-9999)&(h["B8A"]!=-9999)
 windows=[]
 for y in range(0,height-42,42):
  for x in range(0,width-42,42):
   v=valid[y:y+42,x:x+42]
   if v.mean()>=.8:
    aa=a[y:y+42,x:x+42];bb=b[y:y+42,x:x+42]
    if aa[v].std()>.01 and bb[v].std()>.01:windows.append((aa,bb,v,y,x))
 windows=windows[:20]
 if len(windows)<20:raise ValueError(f"only {len(windows)} quality controls")
 rows=[]
 for i,(aa,bb,v,y,x) in enumerate(windows):
  def cor(z,m):return float(np.corrcoef(aa[m],z[m])[0,1])
  m=v&np.roll(v,1,1);m[:,:1]=False
  wrong=windows[(i+7)%len(windows)];rows.append(dict(window=[y,x,42,42],valid_fraction=float(v.mean()),same=cor(bb,v),shift1=cor(np.roll(bb,1,1),m),transpose=cor(bb.T,v&v.T),wrong=cor(wrong[1],v&wrong[2])))
 positive=np.median([r["same"] for r in rows]);wrong=np.median([r["wrong"] for r in rows]);shift=np.median([r["shift1"] for r in rows])
 result=dict(status="accepted" if positive>wrong and positive>shift else "failed",hls_product=sid,hls_source_product=uri,s2_l2a_product=scene.extra.get("product_uri") or scene.scene_id,matched_fields=[platform,acquisition,orbit,tile],independent_acquisitions=1,common_grid_m=30,rows=rows,medians=dict(same=float(positive),shift1=float(shift),wrong=float(wrong)))
 (out/"same_acquisition_control.json").write_text(json.dumps(result,indent=2))
 np.savez_compressed(out/"same_acquisition_arrays.npz",s2_ndvi=a,hls_ndvi=b,valid=valid,transform=np.array(tuple(grid)))
 fig,axs=plt.subplots(4,3,figsize=(9,10))
 for row,(aa,bb,v,_,_) in zip(axs,windows[:4]):
  for ax,im,title in zip(row,[aa,bb,abs(aa-bb)],["S2 L2A NDVI","HLS S30 NDVI","Absolute difference"]):ax.imshow(np.where(v,im,np.nan),vmin=0,vmax=1);ax.set_title(title);ax.axis("off")
 fig.tight_layout();fig.savefig(out/"same_acquisition_control.png",dpi=130)
 print(json.dumps(result))
if __name__=="__main__":main()
