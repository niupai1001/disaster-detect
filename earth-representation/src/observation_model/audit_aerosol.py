"""Read-only audit of a demonstrated missing L2 aerosol-quality condition."""
import collections
import json
from pathlib import Path

import numpy as np

from .data import ROOT,rows
from .native_data import native_window


def main():
    result=[]
    for row in rows():
        if row["sensor"]!="L8":continue
        path=Path(row["bands"][0]["source"]).parent/"qa_aerosol.tif"
        record=dict(id=row["observation_id"],split=row["split"],acquisition=row["acquisition_id"],source=str(path))
        if not path.exists():record["status"]="missing_local_aerosol_QA_no_download";result.append(record);continue
        aerosol,_,transform,crs=native_window(path,row["bbox"],30,bbox_crs=row["crs"])
        assert np.allclose(list(transform)[:6],row["bands"][0]["transform"],atol=1e-6,rtol=0)
        valid_retrieval=(aerosol&2)!=0;interpolated=(aerosol&32)!=0;high=(aerosol>>6)==3
        bad=((aerosol&1)!=0)|(high&(~valid_retrieval|interpolated))
        with np.load(row["cache"]) as z:base=np.logical_and.reduce([z[b["name"]+"_valid"] for b in row["bands"]])
        usable=float((base&~bad).mean())
        record.update(status="read",original_joint_valid_fraction=float(base.mean()),candidate_valid_fraction=usable,
                      high_aerosol_fraction=float(high.mean()),valid_retrieval_fraction=float(valid_retrieval.mean()),
                      interpolated_fraction=float(interpolated.mean()),candidate_pass80=usable>=.8)
        result.append(record)
    summary=dict(status="read_only_candidate_policy_audit_not_a_new_release",policy="exclude aerosol fill or high aerosol with invalid retrieval/interpolation; other existing QA unchanged",
                 source="https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands",
                 totals=dict(collections.Counter(r["split"] for r in result)),
                 candidate_pass80=dict(collections.Counter(r["split"] for r in result if r.get("candidate_pass80"))),
                 missing_QA=dict(collections.Counter(r["split"] for r in result if r["status"]!="read")),
                 boundary="all frozen members audited equally, no data or masks overwritten, no downloads; any new release changes the evaluation population and requires matched controls")
    folder=ROOT/"observation_model/outputs/diagnosis";folder.mkdir(exist_ok=True)
    (folder/"aerosol_members.json").write_text(json.dumps(result,indent=2));(folder/"aerosol_summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary))


if __name__=="__main__":main()
