"""Replay best, median and worst D1 examples from the committed checkpoints."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from candidate_gate import config_device, options
from run_v3_fr import bind_runtime, hash_file, read, write
from screen import load_csv


@torch.inference_mode()
def run(args):
    from encoder_v3.film_resnet.model import FRPretraining
    from encoder_v3.hierarchical_revised.tasks import task_masks
    from encoder_v3.shared_mae.data import load_data, sample_from_dataset
    from run_v3_fr import mask_sha

    torch.set_num_threads(2)
    cfg, settings, device = config_device(args)
    datasets, resolver = load_data()
    val = datasets["VAL"]
    indices = {r["id"]: i for i, r in enumerate(val.rows)}
    source = args.campaign / "s6_component_trials"
    models = {}
    for arm in ("N2", "C"):
        path = source / arm / "update_233.pt"
        status = read(path.parent / "status.json")
        saved = read(path.parent / "identity.json")
        assert hash_file(path) == status["checkpoint_hash"]
        # This pickle was produced by this campaign. Never use this path for outside checkpoints.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint["identity"] == saved and checkpoint["runner"]["cursor"] == 233
        model = FRPretraining(cfg, options(arm), seed=saved["init_seed"])
        model.load_state_dict(checkpoint["model"], strict=True)
        models[arm] = model.to(device).eval()

    pairs = [r for r in load_csv(source / "paired_effects.csv") if (r["candidate"],r["baseline"]) == ("C","N2")]
    results = {arm: {(r["observation_id"],r["task"]):r for r in load_csv(source/arm/"val_233/metrics_by_observation.csv")}
               for arm in models}
    figures = args.campaign / "summary/figures"
    dest = source / "paired_examples"
    dest.mkdir(exist_ok=True)
    manifest = []
    for task in ("A", "B1"):
        ordered = sorted((r for r in pairs if r["task"] == task),key=lambda r:float(r["mse_delta"]))
        figure, axes = plt.subplots(3, 7, figsize=(14, 6.7), layout="constrained")
        for row_id, (case, row) in enumerate(zip(("Most improved","Median","Most degraded"),
                                      (ordered[0],ordered[len(ordered)//2],ordered[-1]),strict=True)):
            obs = row["observation"]
            sample = sample_from_dataset(val, indices[obs], resolver, device)
            keeps, targets = task_masks(sample, cfg, task, settings["validation_seed"], 0)
            i = max(range(len(targets)), key=lambda j:int(targets[j].sum()))
            band = sample.channels[i]
            mask, keep, qa, truth = [x.cpu().numpy() for x in (targets[i],keeps[i],band.valid,band.values)]
            start = sum(int(m.sum()) for m in targets[:i])
            count = int(mask.sum())
            outputs = {}
            limits = np.quantile(truth[qa], [.02,.98])
            for arm,model in models.items():
                output = model(sample, kind=task, seed=settings["validation_seed"])
                row_metric = results[arm][obs,task]
                assert mask_sha(output.targets)==row_metric["target_mask_hash"]
                torch.testing.assert_close(output.mse, torch.tensor(float(row_metric["mse"]),device=device),
                                           atol=1e-6,rtol=1e-5)
                for key in ("predictions","main_prediction","detail_contribution"):
                    image = np.full_like(truth, np.nan)
                    image[mask] = getattr(output,key)[start:start+count].cpu().numpy()
                    outputs[f"{arm}_{key}"] = image
            ref = next(j for j,m in enumerate(keeps) if bool(m.any()))
            reference = torch.where(keeps[ref],sample.channels[ref].values,torch.nan).cpu().numpy()
            filename = f"{task}_{row_id+1}.npz"
            np.savez_compressed(dest/filename,truth=truth,qa=qa,target_mask=mask,visible_mask=keep,
                visible_values=np.where(keep,truth,0),visible_reference=reference,display_scale=limits,
                affine=band.grid.affine,crs=band.grid.crs,**outputs)
            a,b = (outputs[f"{arm}_main_prediction"] for arm in models)
            left,right = (outputs[f"{arm}_predictions"] for arm in models)
            err = abs(right-truth)
            error_max = max(float(np.quantile(np.concatenate([abs(left-truth)[mask],err[mask]]),.98)),1e-8)
            shown = reference if task=="B1" else np.where(keep,truth,np.nan)
            shown_scale = np.nanquantile(reference,[.02,.98]) if task=="B1" else limits
            shown_title = f"Visible {sample.channels[ref].name}" if task=="B1" else "Visible target-band input"
            for column,(arr,title,scale,cmap) in enumerate([
                (np.where(qa,truth,np.nan),"Native truth",limits,"gray"),(shown,shown_title,shown_scale,"gray"),
                (mask,"Target mask",(0,1),"gray"),(a,"N2 main",limits,"gray"),(b,"C main",limits,"gray"),
                (right,"C total",limits,"gray"),(err,"C total error",(0,error_max),"magma")]):
                ax = axes[row_id,column]
                ax.imshow(arr,vmin=scale[0],vmax=scale[1],cmap=cmap,interpolation="none")
                ax.set_xticks([])
                ax.set_yticks([])
                if row_id==0:
                    ax.set_title(title,fontsize=9)
                if column==0:
                    ax.set_ylabel(f"{case}\n{row['sensor']} {band.name}\nMSE Δ {float(row['mse_delta']):+.4f}",fontsize=9)
            manifest.append(dict(task=task,selection=case,id=obs,sensor=row["sensor"],band=band.name,
                mse_delta=float(row["mse_delta"]),float_file=filename))
        figure.suptitle(f"V3-FR · {task} · D1 vs D0 · paired by observation at update 233")
        figure.savefig(figures/f"s6_D1_paired_{task}.png",dpi=155)
        plt.close(figure)
    write(dest/"selection.json",dict(rule="Rank total native MSE delta C minus N2 within task",
                                     examples=manifest))
    print("Replayed",len(manifest),"paired observation examples",flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    bind_runtime(args.release)
    run(args)
