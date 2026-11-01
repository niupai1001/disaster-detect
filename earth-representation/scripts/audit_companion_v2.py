import json
from pathlib import Path
import numpy as np
from scipy.io import netcdf_file
from src.data.qa_masks import landsat_qa_masks,hls_fmask_masks

def main():
    root=Path("data/companions_v2_selected_time")
    rows=[]
    for product in ("landsat","hls"):
        for p in sorted((root/f"sen12_{product}").glob("*.nc")):
            with netcdf_file(str(p),"r",mmap=False) as n:
                keys=[k for k in n.variables if k.startswith("SR_B") or (product=="hls" and k.startswith("B"))]
                a=np.stack([np.array(n.variables[k][:]) for k in keys])
                qa=np.array(n.variables["QA_PIXEL" if product=="landsat" else "Fmask"][:])
                valid=landsat_qa_masks(qa,np.array(n.variables["QA_RADSAT"][:]))["valid"] if product=="landsat" else hls_fmask_masks(qa)["valid"]
                valid &= np.isfinite(a).all(0)
                rows.append(dict(sample=p.stem,product=product,scene=n.scene_id.decode(),schema=int(n.schema_version),dtype=str(a.dtype),valid_fraction=float(valid.mean()),quantiles=np.quantile(a[:,valid],[.01,.5,.99]).tolist() if valid.any() else [],gap_days=int(n.date_gap_days)))
    out=Path("outputs/verification/companion_audit.json")
    out.write_text(json.dumps(rows,indent=2))
    print(json.dumps({"files":len(rows),"products":{p:sum(x["product"]==p for x in rows) for p in ("landsat","hls")}}))

if __name__=="__main__":main()
