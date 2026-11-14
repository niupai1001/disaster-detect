"""S5 full-visible and independent-latent engineering fits on frozen TRAIN."""
from dataclasses import asdict
import json
import math
import time

import numpy as np
import torch

from candidate_gate import config_device, options
from run_v3_fr import hash_file, read, write


def fit(args):
    from encoder_v3.film_resnet.model import FRPretraining,TargetSpec
    from encoder_v3.hierarchical_revised.train import save
    from encoder_v3.latent_response.checkpoint import parameter_sha
    from encoder_v3.patch_latent.model import Encoded
    from encoder_v3.shared_mae.data import load_data,sample_from_dataset
    from encoder_v3.shared_mae.geometry import metric_geometry
    from torch.nn import functional as F

    assert all(read(args.output/"s4_decoder_execution"/name)["status"]=="PASS" for name in
               ("capacity.json","decoder_terminal_gate.json","one_step_equivalence.json","resume_gate.json"))
    cfg,settings,device = config_device(args)
    data,resolver = load_data()
    panel = read(args.output/"data_panel.json")
    members = {role:[sample_from_dataset(data["TRAIN"],item["index"],resolver,device) for item in panel[role]]
               for role in ("fit","read_only")}
    assert all(s.role=="TRAIN" for group in members.values() for s in group)
    out = args.output/"s5_capacity_fit"
    out.mkdir(exist_ok=False)
    steps = 400
    lr = .0003
    fit_ids = [s.identifier for s in members["fit"]]

    # Positive optimizer control uses only an independent test variable and an exact native mask.
    first = members["fit"][0].channels[0]
    index = first.valid.flatten().nonzero().flatten()
    assert len(index)>0
    target = first.values.flatten()[index].detach()
    control = torch.nn.Parameter(target.clone()+.05)
    control_opt = torch.optim.AdamW([control],lr=.01,weight_decay=0)
    before = (control-target).square().mean()
    before.backward()
    assert torch.count_nonzero(control.grad)==len(index)
    control_opt.step()
    after = (control-target).square().mean()
    assert after<before
    write(out/"positive_control.json",dict(status="PASS",initial=float(before.detach()),after=float(after.detach()),
                                         native_pixels=len(index),optimizer_updates=1))
    outcomes = {}
    for arm in ("AE-L","AE-F","OR-D0","OR-D1"):
        start = time.perf_counter()
        path = out/arm
        path.mkdir()
        choice = options("C" if arm in ("AE-F","OR-D1") else "B0")
        model = FRPretraining(cfg,choice,seed=17).to(device)
        model.legacy.requires_grad_(False)
        oracle = arm.startswith("OR-")
        latents = torch.nn.ParameterDict()
        fixed = {}
        if oracle:
            model.encoder.requires_grad_(False)
            if model.position is not None:
                model.position.requires_grad_(False)
            gen = torch.Generator(device=device).manual_seed(170012)
            for s in members["fit"]:
                with torch.no_grad():
                    geometry = metric_geometry(s.channels,cfg.grid_metres)
                    desc = model.encoder.descriptor([c.physical for c in s.channels])
                    # Only immutable native geometry determines the real feature shapes and support.
                    from types import SimpleNamespace
                    from encoder_v3.patch_latent.geometry import patch_map
                    support = torch.stack([patch_map(g,geometry,device).footprint
                                           for g in dict.fromkeys(geometry.grids)]).any(0).reshape(1,1,*geometry.shape)
                    supports = []
                    for width in cfg.widths:
                        key = f"{s.identifier}_{width}"
                        latents[key] = torch.nn.Parameter(.02*torch.randn(1,width,*support.shape[-2:],
                                                                         device=device,generator=gen))
                        supports.append(support)
                        support = F.max_pool2d(support.float(),3,2,1)>0
                    fixed[s.identifier] = geometry,desc,tuple(supports)
        latents_hash = parameter_sha(latents) if oracle else None
        bundle = torch.nn.ModuleDict({"model":model,"latents":latents})
        parameters = [p for p in bundle.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(parameters,lr=lr,weight_decay=0.)
        runner = dict(cursor=0,spent_updates=0,observations_seen=0)
        binding = dict(arm=arm,options=asdict(choice),seed=17,fit_ids=fit_ids,
                       initial_parameter_hash=parameter_sha(bundle),initial_latent_hash=latents_hash,
                       policy=dict(lr=lr,clip=1.,steps=steps,objective="native band-equal MSE"))
        write(path/"identity.json",binding)
        save(path/"init.pt",bundle,opt,runner,binding)

        def predict(sample):
            target_masks = [c.valid for c in sample.channels]
            if oracle:
                geometry,desc,support = fixed[sample.identifier]
                features = tuple(latents[f"{sample.identifier}_{w}"] for w in cfg.widths)
                encoded = Encoded(features,support,geometry,desc)
                specs = [TargetSpec(c.grid,t,d) for c,t,d in zip(sample.channels,target_masks,desc,strict=True)]
            else:
                _,encoded,specs = model.views(sample,target_masks,target_masks)
            return model.decoder.predict(encoded,specs)

        def objective(sample,pred):
            return torch.stack([(value-sample.channels[i].values[sample.channels[i].valid]).square().mean()
                                for i,value in pred.items()]).mean()

        def evaluate(step,export=False):
            bundle.eval()
            rows,arrays = [],{}
            with torch.no_grad():
                for role,group in members.items():
                    if oracle and role!="fit":
                        continue
                    for s in group:
                        pred = predict(s)
                        mses,gerr,zero_mse,zero_grad = [],[],[],[]
                        for i,raw in pred.items():
                            channel = s.channels[i]
                            mask = channel.valid
                            truth = channel.values
                            error = raw-truth[mask]
                            mses.append(float(error.square().mean()))
                            zero_mse.append(float(truth[mask].square().mean()))
                            full = torch.zeros_like(truth).masked_scatter(mask,error)
                            for a in (0,1):
                                valid = mask.narrow(a,1,mask.shape[a]-1)&mask.narrow(a,0,mask.shape[a]-1)
                                if valid.any():
                                    gerr.append(float(full.diff(dim=a)[valid].square().mean()))
                                    zero_grad.append(float(truth.diff(dim=a)[valid].square().mean()))
                            if export:
                                key = s.identifier+"_"+channel.name
                                arrays[key+"_truth"] = truth.cpu().numpy()
                                arrays[key+"_mask"] = mask.cpu().numpy()
                                arrays[key+"_prediction"] = torch.full_like(truth,torch.nan).masked_scatter(mask,raw).cpu().numpy()
                                arrays[key+"_affine"] = np.array(channel.grid.affine)
                                arrays[key+"_crs"] = np.array(channel.grid.crs)
                        result = dict(id=s.identifier,role=role,step=step,mse=float(np.mean(mses)),
                                      zero_mse=float(np.mean(zero_mse)),gradient_mse=float(np.mean(gerr)),
                                      zero_gradient_mse=float(np.mean(zero_grad)))
                        assert all(math.isfinite(result[k]) for k in ("mse","gradient_mse","zero_mse","zero_gradient_mse"))
                        rows.append(result)
            if export:
                np.savez_compressed(path/f"predictions_{step:03}.npz",**arrays)
            bundle.train()
            return rows

        evaluation = evaluate(0,True)
        best = math.inf
        best_step = None
        with (path/"updates.jsonl").open("x",encoding="utf-8",buffering=1) as stream:
            for step in range(1,steps+1):
                opt.zero_grad(set_to_none=True)
                before = [p.detach().clone() for p in parameters]
                losses = []
                for sample in members["fit"]:
                    value = objective(sample,predict(sample))
                    assert torch.isfinite(value)
                    (value/len(fit_ids)).backward()
                    losses.append(float(value.detach()))
                latent_grad = ({k:float(p.grad.norm()) for k,p in latents.items()} if oracle else {})
                assert not oracle or all(v>0 and math.isfinite(v) for v in latent_grad.values())
                norm = float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
                opt.step()
                change = float(torch.stack([(p.detach()-old).square().sum() for p,old in zip(parameters,before)]).sum().sqrt())
                assert math.isfinite(change) and change>0
                runner.update(cursor=step,spent_updates=step,observations_seen=step*len(fit_ids))
                stream.write(json.dumps(dict(step=step,losses=dict(zip(fit_ids,losses)),lr=lr,gradient_norm=norm,
                    latent_gradient_norm=latent_grad,parameter_change=change),allow_nan=False)+"\n")
                if step%10==0:
                    node = evaluate(step,step==steps)
                    evaluation.extend(node)
                    score = float(np.mean([r["mse"] for r in node if r["role"]=="fit"]))
                    if score<best:
                        best,best_step = score,step
                        save(path/"best.pt",bundle,opt,runner,binding)
                    if step%100==0:
                        print("S5",arm,step,round(score,6),flush=True)
                    write(path/"evaluation.json",evaluation)
                if step%25==0:
                    save(path/"last.pt",bundle,opt,runner,binding)
        from encoder_v3.hierarchical_revised.train import resume
        resume(path/"best.pt",bundle,opt,runner,binding)
        replayed = evaluate(best_step,True)
        previous = [r for r in evaluation if r["step"]==best_step]
        for a,b in zip(replayed,previous,strict=True):
            assert a["id"]==b["id"]
            assert abs(a["mse"]-b["mse"]) <= 1e-6+1e-5*abs(b["mse"])
        refs = [r for r in replayed if r["role"]=="fit"]
        passed = [r["id"] for r in refs if r["zero_mse"]>1e-6 and r["zero_gradient_mse"]>1e-6
                  and r["mse"]<=.5*r["zero_mse"] and r["gradient_mse"]<=.9*r["zero_gradient_mse"]]
        result = dict(status="COMPLETE",arm=arm,updates=steps,fit_views_per_member=steps,best_step=best_step,
                      initial_latent_hash=latents_hash,screening_fit_members=passed,
                      initial=[r for r in evaluation if r["step"]==0],best=refs,
                      last=[r for r in evaluation if r["step"]==steps and r["role"]=="fit"],
                      artifacts={p.name:hash_file(p) for p in path.iterdir() if p.suffix in (".pt",".npz")},
                      wall_seconds=time.perf_counter()-start)
        outcomes[arm] = result
        write(path/"status.json",result)
    assert outcomes["OR-D0"]["initial_latent_hash"]==outcomes["OR-D1"]["initial_latent_hash"]
    write(out/"fit_gate.json",dict(status="FIT_RUNS_VALID",results=outcomes,
          effect_screened=bool(outcomes["AE-F"]["screening_fit_members"] or outcomes["OR-D1"]["screening_fit_members"]),
          ridge_probe_status="PENDING"))
