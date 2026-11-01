"""Physical checks, scene isolation and TRAIN statistics for the local-only release."""
import collections
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import rasterio

from .data import ROOT,RELEASE,rows,fit_statistics,masking_contract,BalancedSampler
from .native_data import native_window,aerosol_valid
from src.data.qa_masks import hls_fmask_masks


def hls_controls():
    result=[]
    for family in ["S30","L30"]:
        folders=sorted((ROOT/"data/raw/hls").glob("HLS."+family+".*"))[:2]
        if len(folders)<2:raise ValueError("local HLS2+2 samples missing; do not download under user pause")
        for folder in folders:
            xml=ET.parse(next(folder.glob("*.xml"))).getroot()
            attributes={}
            for element in xml.iter():
                if element.tag.split("}")[-1]!="AdditionalAttribute":continue
                name=next((x.text for x in element if x.tag.split("}")[-1]=="Name"),None)
                value=next((x.text for x in element.iter() if x.tag.split("}")[-1]=="Value"),None)
                if name:attributes[name]=value
            scale=float(attributes["REF_SCALE_FACTOR"]);offset=float(attributes["ADD_OFFSET"])
            with rasterio.open(folder/"B04_aoi.tif") as ds:
                dn=ds.read(1);resolution=ds.res;transform=ds.transform;crs=str(ds.crs);nodata=ds.nodata
            with rasterio.open(folder/"Fmask_aoi.tif") as ds:
                qa=ds.read(1)
                assert ds.transform==transform and str(ds.crs)==crs
            assert resolution==(30.,30.)
            valid=hls_fmask_masks(qa)["valid"]&np.isfinite(dn)
            if nodata is not None:valid &= dn!=nodata
            sr=dn[valid].astype(np.float32)*scale+offset
            if not len(sr):raise ValueError("local HLS sample has no valid red reflectance")
            result.append(dict(product=folder.name,collection="HLSS30_2.0" if family=="S30" else "HLSL30_2.0",doi=attributes.get("IDENTIFIER_PRODUCT_DOI"),scale=scale,offset=offset,source=str(folder),grid_m=30,crs=crs,valid_pixels=len(sr),reflectance_quantiles=np.quantile(sr,[.01,.5,.99]).tolist()))
    return result


def main():
    config=json.loads((ROOT/"observation_model/config.json").read_text());assert not config["download_enabled"]
    np.testing.assert_allclose(np.array([10000,20000,30000,40000],np.float32)*.0000275-.2,[.075,.35,.625,.9],atol=1e-7)
    np.testing.assert_array_equal(hls_fmask_masks(np.array([0,1,2,4,255],np.uint8))["valid"],[True,True,False,False,False])
    np.testing.assert_array_equal(aerosol_valid(np.array([0,1,2,66,130,192,194,224,226],np.uint16)),[True,False,True,True,True,False,True,False,False])
    examples=[]
    all_rows=rows()
    for sensor in ["S2","L8"]:
        for row in [r for r in all_rows if r["sensor"]==sensor][:2]:
            with np.load(row["cache"]) as z:
                band=row["bands"][0];dn,valid,transform,crs=native_window(band["source"],row["bbox"],band["native_gsd_m"],bbox_crs=row["crs"])
                reference=dn.astype(np.float32)*band["scale"]+band["offset"]
                mask=z[band["name"]+"_valid"]
                np.testing.assert_allclose(z[band["name"]][mask],reference[mask],atol=1e-7)
                assert tuple(transform)[:6]==tuple(band["transform"]) and str(crs)==str(band["crs"])
                examples.append(dict(id=row["observation_id"],sensor=sensor,product=row["product_id"],max_error=float(np.max(np.abs(z[band["name"]][mask]-reference[mask]))),native_grid_m=band["native_gsd_m"]))
    memberships={s:{r["group"] for r in all_rows if r["split"]==s} for s in ["train","val","test"]}
    acquisitions={s:{r["acquisition_id"] for r in all_rows if r["split"]==s} for s in memberships}
    for a,b in [("train","val"),("train","test"),("val","test")]:
        assert not memberships[a]&memberships[b] and not acquisitions[a]&acquisitions[b]
    lengths={s:sum(r["split"]=="train" and r["sensor"]==s for r in all_rows) for s in ["S2","L8"]}
    sampler=BalancedSampler(lengths)
    for s,n in lengths.items():
        ids=sampler.take(s,n+7);assert len(set(ids))==n and min(sampler.exposure[s])>=1
    stats=fit_statistics();masking_contract()
    summary=json.loads((RELEASE/"release.json").read_text())
    pairs=[json.loads(s) for s in (RELEASE/"cross30.jsonl").read_text().splitlines()]
    strict={s:[r for r in pairs if r["split"]==s and r["strict_eligible"]] for s in memberships}
    cross_gate=all(len({r["group"] for r in strict[s]})>=minimum for s,minimum in [("train",4),("val",2),("test",2)]) and all(len({(r["s2_scene"],r["landsat_scene"]) for r in strict[s]})>=3 for s in ["val","test"])
    checks=dict(status="local_physical_and_membership_checks_pass",source_examples=examples,hls_local_product_controls=hls_controls(),mask_and_DN_known_values=True,splits_disjoint_by_group_and_acquisition=True,short_cycle_all_members_updated=True,statistics_train_only=True,cross_source_gate_passed=cross_gate,cross_source_status="CONDITIONAL_READY" if cross_gate else "BLOCKED_INSUFFICIENT_LOCAL_QUALIFIED_PAIR_COVERAGE_NO_DOWNLOAD",independent_landcover="missing; WorldCover product agreement is supplemental")
    if not cross_gate and config.get("cross_execution",{}).get("required_by_latest_user"):
        checks["cross_source_status"]="EXPLORATORY_REQUIRED_QUALITY_GATE_UNCHANGED_LIMITED_COVERAGE"
    (RELEASE/"checks.json").write_text(json.dumps(checks,indent=2))
    summary.update(status="ACCEPTED_LOCAL_NATIVE_FOR_MASKED",cross_source_gate_passed=cross_gate,hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in RELEASE.iterdir() if p.name!="release.json" and p.suffix in [".jsonl",".json"]})
    (RELEASE/"release.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(dict(status=summary["status"],cross_source_gate=cross_gate,source_examples=len(examples),hls_samples=4,statistics_bands=len(stats))))


if __name__=="__main__":main()
