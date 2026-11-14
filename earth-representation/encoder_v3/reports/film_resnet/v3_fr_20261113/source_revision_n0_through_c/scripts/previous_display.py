"""Recover the float values behind the previous B1-detail comparison figure."""
import argparse
import json
from pathlib import Path
from dataclasses import replace

import numpy as np
import torch

from run_v3_fr import bind_runtime,read,write,hash_file


@torch.no_grad()
def run(args):
    from encoder_v3.hierarchical_revised.tasks import task_masks
    from encoder_v3.patch_latent.model import Config,Pretraining
    from encoder_v3.shared_mae.data import load_data,sample_from_dataset

    torch.set_num_threads(2)
    config = Config(**read(args.release/"config.json")["model"])
    data,resolver = load_data()
    val = data["VAL"]
    prior_panel = read(args.prior/"summary/paired_comparisons.json")["B-DETAIL-OFF"]["B1/all"]["selected"]
    lookup = {r["id"]:i for i,r in enumerate(val.rows)}
    out = args.output/"s1_baseline"
    raw,rows,models = {},[],{}
    allowed = [np._core.multiarray._reconstruct,np.ndarray,np.dtype,np.dtypes.UInt32DType]
    for arm,stage in (("T-JOINT","p3_task_interference"),("B-DETAIL-OFF","p4_local_candidates")):
        checkpoint = args.prior/stage/arm/"last.pt"
        checkpoint_sha = hash_file(checkpoint)
        assert checkpoint_sha==read(checkpoint.parent/"status.json")["checkpoint_sha256"]
        model = Pretraining(config)
        with torch.serialization.safe_globals(allowed):
            state = torch.load(checkpoint,map_location="cpu",weights_only=True)["model"]
        model.load_state_dict(state,strict=True)
        models[arm] = model.eval(),checkpoint_sha
    for selection,id_ in prior_panel.items():
        sample = sample_from_dataset(val,lookup[id_],resolver,"cpu")
        keeps,targets = task_masks(sample,config,"B1",17,0)
        i = max(range(len(targets)),key=lambda k:int(targets[k].sum()))
        band = sample.channels[i]
        keep,target = keeps[i],targets[i]
        lo,hi = np.quantile(band.values[band.valid].numpy(),[.02,.98])
        truth = band.values[target].numpy()
        for arm,(model,checkpoint_sha) in models.items():
            encoded = model.encoder(sample.channels,keeps)
            main = model.decoder(encoded,[c.grid for c in sample.channels],targets)
            total = model.decoder.add_native_detail(main,encoded,sample.channels,keeps,targets) if arm=="T-JOINT" else main
            values = total[i].numpy()
            below,above = float(np.mean(values<lo)),float(np.mean(values>hi))
            var = float(values.var()/truth.var()) if truth.var()>1e-12 else None
            labels = []
            if max(below,above)>=.95:
                labels.append("DISPLAY_SATURATION")
            if abs(float(values.mean()-truth.mean()))>float(truth.std()):
                labels.append("RADIOMETRIC_BIAS")
            if var is not None and var<=.1:
                labels.append("LOW_SPATIAL_VARIATION")
            if len(labels)>1:
                labels.append("MIXED")
            full = torch.full_like(band.values,torch.nan)
            prefix = arm+"_"+id_+"_"
            for k,v in dict(truth=band.values,visible_values=torch.where(keep,band.values,0.),
                            visible_mask=keep,qa=band.valid,target_mask=target,
                            main_prediction=full.masked_scatter(target,main[i]),
                            detail_contribution=full.masked_scatter(target,total[i]-main[i]),
                            total_prediction=full.masked_scatter(target,total[i])).items():
                raw[prefix+k] = v.numpy()
            ref = next(j for j,m in enumerate(keeps) if bool(m.any()))
            raw[prefix+"visible_reference"] = torch.where(keeps[ref],sample.channels[ref].values,torch.nan).numpy()
            rows.append(dict(arm=arm,id=id_,selection=selection,band=band.name,task="B1",update=233,
                 checkpoint_sha256=checkpoint_sha,target_count=len(values),display_low=float(lo),display_high=float(hi),
                 clipped_below=below,clipped_above=above,variance_ratio=var,labels=labels or ["UNRESOLVED"],
                 pred_mean=float(values.mean()),true_mean=float(truth.mean()),pred_std=float(values.std()),
                 true_std=float(truth.std()),pred_quantiles=np.quantile(values,[.01,.05,.5,.95,.99]).tolist(),
                 pred_minimum=float(values.min()),pred_maximum=float(values.max()),
                 constant_zero_mse=float(np.mean(truth**2)),mse=float(np.mean((values-truth)**2)),
                 affine=band.grid.affine,crs=band.grid.crs,reference_band=sample.channels[ref].name))
    write(out/"previous_display_audit.json",dict(status="PASS",scope="Previous B1 detail comparison, CPU eager",rows=rows))
    np.savez_compressed(out/"previous_display_raw.npz",**raw)
    print("Previous display replayed",len(rows),"arm-observation pairs",flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("release","prior","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    args = parser.parse_args()
    bind_runtime(args.release)
    run(args)
