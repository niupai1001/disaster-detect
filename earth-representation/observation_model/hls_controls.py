"""Reuse real same-acquisition source/HLS controls without any network requests."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def correlation(a,b,valid):
    x=a[valid];y=b[valid]
    if len(x)<10 or min(x.std(),y.std())<1e-6:return None
    return float(np.corrcoef(x,y)[0,1])


def main():
    prior=ROOT/"outputs/verification";results=[]
    for kind,json_name,array_name,keys in [("S2_S30","same_acquisition_control.json","same_acquisition_arrays.npz",("s2_ndvi","hls_ndvi")),("Landsat_L30","hls_l30_control.json","hls_l30_control_arrays.npz",("landsat_ndvi","l30_ndvi"))]:
        metadata=json.loads((prior/json_name).read_text())
        with np.load(prior/array_name) as z:a=z[keys[0]];b=z[keys[1]];valid=z["valid"]
        boxes=[r["window"] for r in metadata["rows"]];rows=[]
        for i,(row,col,h,w) in enumerate(boxes):
            other_row,other_col,_,_=boxes[(i+1)%len(boxes)]
            aa=a[row:row+h,col:col+w];bb=b[row:row+h,col:col+w];vv=valid[row:row+h,col:col+w]
            wrong=b[other_row:other_row+h,other_col:other_col+w];wrong_valid=valid[other_row:other_row+h,other_col:other_col+w]
            rows.append(dict(window_y_x_h_w=[row,col,h,w],same=correlation(aa,bb,vv),wrong=correlation(aa,wrong,vv&wrong_valid)))
        same=float(np.median([r["same"] for r in rows if r["same"] is not None]));wrong=float(np.median([r["wrong"] for r in rows if r["wrong"] is not None]))
        if not same>wrong:raise ValueError("same-acquisition control no longer distinguishes wrong locations")
        products={k:v for k,v in metadata.items() if "product" in k and isinstance(v,str)}
        results.append(dict(kind=kind,products=products,source_json=str(prior/json_name),source_array_sha256=hashlib.sha256((prior/array_name).read_bytes()).hexdigest(),windows=rows,median_same=same,median_wrong=wrong,independent_acquisitions=1,grid_m=30))
    path=ROOT/"observation_model/outputs/phase0/hls_same_acquisition_controls.json"
    path.write_text(json.dumps(dict(status="recomputed_local_same_acquisition_controls_pass",results=results,boundary="20 windows per source acquisition; two source acquisitions total, not40 independent scenes; HLS is a derived product, not independent spectral truth"),indent=2))
    print(json.dumps([{k:r[k] for k in ["kind","median_same","median_wrong"]} for r in results]))


if __name__=="__main__":main()
