"""Frozen A/B1 validation plus per-band main/detail and original prediction evidence."""
from collections import defaultdict
import math
import time

import numpy as np
import torch

from run_v3_fr import mask_sha, table, write


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def adjacent(x,mask):
    return tuple(np.diff(x,axis=axis)[np.take(mask,range(1,mask.shape[axis]),axis=axis)
                                    & np.take(mask,range(mask.shape[axis]-1),axis=axis)]
                 for axis in (0,1))


def band_metrics(raw, main, truth, target, valid, texture):
    raw,main,truth = (a.astype(np.float64,copy=False) for a in (raw,main,truth))
    a,b,c = raw[target], main[target],truth[target]
    diff = a-c
    main_err = b-c
    true_gradient,pred_gradient,main_gradient = (adjacent(v,target) for v in (truth,raw,main))
    high,high_main = texture
    t,p,m = (np.concatenate(x) for x in (true_gradient,pred_gradient,main_gradient))
    scale_low,scale_high = np.quantile(truth[valid],[.02,.98])
    mse = float(np.mean(diff**2))
    zero = float(np.mean(c**2))
    return dict(mse=mse,mae=float(abs(diff).mean()),historical_texture_gradient_mse=mean(high),
                main_texture_gradient_mse=mean(high_main),texture_axes=len(high),
                true_mean=float(c.mean()),pred_mean=float(a.mean()),true_std=float(c.std()),pred_std=float(a.std()),
                variance_ratio=float(a.var()/c.var()) if c.var()>1e-12 else None,
                correlation=float(np.corrcoef(a,c)[0,1]) if min(a.std(),c.std())>1e-6 else None,
                true_gradient_energy=float(np.mean(t**2)) if len(t) else None,
                pred_gradient_energy=float(np.mean(p**2)) if len(t) else None,
                signed_gradient_error=float(np.mean((p-t)**2)) if len(t) else None,
                main_signed_gradient_error=float(np.mean((m-t)**2)) if len(t) else None,
                zero_baseline_mse=zero,skill_vs_zero=1-mse/zero if zero>1e-6 else None,
                main_path_mse=float(np.mean(main_err**2)),detail_contribution_energy=float(np.mean((a-b)**2)),
                display_low=float(scale_low),display_high=float(scale_high),
                clipped_low_fraction=float(np.mean(a<scale_low)),clipped_high_fraction=float(np.mean(a>scale_high)))


@torch.no_grad()
def evaluate(model,val,resolver,settings,out,arm,seed,step,fixed_ids):
    from encoder_v3.shared_mae.data import sample_from_dataset
    from encoder_v3.hierarchical_revised.tasks import available_tasks,task_masks
    from probe_patch_latent import texture_mse

    start = time.perf_counter()
    model.eval()
    obs,rows,raw = [],[],{}
    for index,row in enumerate(val.rows):
        sample = sample_from_dataset(val,index,resolver,settings["device"])
        assert sample.identifier==row["id"] and sample.role=="VAL"
        for task in available_tasks(sample,model.config):
            keeps,targets = task_masks(sample,model.config,task,settings["validation_seed"],0)
            result = model(sample,kind=task,seed=settings["validation_seed"])
            assert torch.isfinite(result.predictions).all() and mask_sha(result.targets)==mask_sha(targets)
            pred_main = result.main_prediction
            total,main = result.predictions.cpu().numpy(),pred_main.cpu().numpy()
            base = dict(experiment_id=arm,checkpoint=f"update_{step:03d}",seed=seed,update=step,task=task,
                        observation_id=sample.identifier,source_group=row["release_isolation_group"],
                        acquisition_key=row["acquisition_key"],source_product=row["product"],sensor=row["sensor"])
            offset = 0
            member = []
            for i,(c,keep,target) in enumerate(zip(sample.channels,keeps,targets,strict=True)):
                mask = target.cpu().numpy()
                n = int(mask.sum())
                if not n:
                    continue
                truth,valid,visible = c.values.cpu().numpy(),c.valid.cpu().numpy(),keep.cpu().numpy()
                p,m = np.full_like(truth,np.nan),np.full_like(truth,np.nan)
                p[mask],m[mask] = total[offset:offset+n],main[offset:offset+n]
                main_native = torch.zeros_like(c.values).masked_scatter(target,pred_main[offset:offset+n])
                native = torch.zeros_like(c.values).masked_scatter(target,result.predictions[offset:offset+n])
                high,high_main = [],[]
                for axis in (0,1):
                    pairs = ((target[:-1]&target[1:]) if axis==0 else (target[:,:-1]&target[:,1:]))
                    if bool(pairs.any()):
                        true = c.values.diff(dim=axis)[pairs]
                        # Keep the frozen float32 quantile and tie handling.
                        strong = true.abs()>=torch.quantile(true.abs(),.75)
                        high.append(float((native.diff(dim=axis)[pairs]-true)[strong].square().mean()))
                        high_main.append(float((main_native.diff(dim=axis)[pairs]-true)[strong].square().mean()))
                offset += n
                metrics = band_metrics(p,m,truth,mask,valid,(high,high_main))
                assert math.isclose(metrics["mse"],float(result.channel_loss[i]),rel_tol=1e-5,abs_tol=1e-6)
                item = base | dict(band=c.name,native_gsd_x=c.grid.gsd_xy[0],native_gsd_y=c.grid.gsd_xy[1],
                                   qa_count=int(valid.sum()),visible_count=int(visible.sum()),target_count=n,
                                   target_mask_hash=mask_sha([target]),visible_mask_hash=mask_sha([keep]),
                                   same_grid_visible_bands=sum(bool(k.any()) for other,k in zip(sample.channels,keeps)
                                                               if other.grid==c.grid)) | metrics
                rows.append(item)
                member.append(item)
                if sample.identifier in fixed_ids:
                    prefix = f"{sample.identifier}_{task}_{c.name}"
                    raw.update({prefix+"_"+k:v for k,v in dict(truth=truth,qa=valid,visible_mask=visible,target_mask=mask,
                              visible_values=np.where(visible,truth,0.),main_prediction=m,detail_contribution=p-m,
                              total_prediction=p,affine=np.asarray(c.grid.affine),crs=np.asarray(c.grid.crs),
                              display_scale=np.asarray([metrics["display_low"],metrics["display_high"]])).items()})
            assert offset==len(total) and member
            if sample.identifier in fixed_ids:
                for c,keep in zip(sample.channels,keeps,strict=True):
                    if keep.any():
                        raw[f"{sample.identifier}_{task}_visible_reference_{c.name}"] = torch.where(
                            keep,c.values,torch.nan).cpu().numpy()
            hist = texture_mse(sample,result)
            axes = sum(r["texture_axes"] for r in member)
            for_texture = lambda field: (sum(r[field]*r["texture_axes"] for r in member if r["texture_axes"])/axes
                                         if axes else None)
            local = for_texture("historical_texture_gradient_mse")
            assert (local is None and hist is None) or math.isclose(local,hist,abs_tol=1e-6,rel_tol=1e-5), (
                sample.identifier,task,local,hist)
            obs.append(base | dict(mse=float(result.mse),texture_gradient_mse=hist,
                      main_path_mse=mean(r["main_path_mse"] for r in member),
                      main_texture_gradient_mse=for_texture("main_texture_gradient_mse"),
                      detail_contribution_energy=mean(r["detail_contribution_energy"] for r in member),
                      target_count=offset,target_mask_hash=mask_sha(targets),visible_mask_hash=mask_sha(keeps)))
    groups = defaultdict(list)
    for r in obs:
        for sensor in ("all",r["sensor"]):
            groups[(r["task"],sensor)].append(r)
    summary = {f"{task}/{sensor}":dict(count=len(group),**{k:mean(r[k] for r in group) for k in
               ("mse","texture_gradient_mse","main_path_mse","main_texture_gradient_mse","detail_contribution_energy")})
               for (task,sensor),group in groups.items()}
    node = out/f"val_{step:03d}"
    node.mkdir(exist_ok=False)
    table(node/"metrics_by_observation.csv",obs)
    table(node/"metrics_by_band.csv",rows)
    if raw:
        np.savez_compressed(node/"raw_predictions.npz",**raw)
    status = dict(step=step,groups=summary,seconds=time.perf_counter()-start,observation_task_rows=len(obs),
                  band_rows=len(rows))
    write(node/"summary.json",status)
    return status
