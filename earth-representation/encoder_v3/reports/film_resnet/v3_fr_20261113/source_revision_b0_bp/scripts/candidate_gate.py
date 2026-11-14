"""Real fixed-weight compatibility, input permissions and full native capacity."""
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import torch

from run_v3_fr import read, write, mask_sha


def options(arm):
    from encoder_v3.film_resnet.model import Options
    registered = {"B0":(False,0,False,False,False),"BP":(False,0,False,True,False),
                  "N0":(True,0,False,True,False),"N1":(True,2,False,True,False),
                  "N2":(True,2,True,True,False),"C":(True,2,True,True,True),
                  "PD":(False,0,False,True,True)}
    ordered,blocks,film,position,d1 = registered[arm]
    return Options("ordered" if ordered else "historical",blocks,film,
                   "qk_physical" if position else "none",
                   "learned_subpixel_residual" if d1 else "legacy_continuous",True)


def load_legacy_weights(model, state):
    mapped = {("legacy.detail_gain."+key.removeprefix("decoder.detail_gain.")
               if key.startswith("decoder.detail_gain.") else key):value for key,value in state.items()}
    model.load_state_dict(mapped,strict=True)


def config_device(args):
    from train_patch_latent import Config
    config = read(args.release/"config.json")
    settings = config["training"]
    device = settings["device"]
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return Config(**config["model"]), settings, device


def run(args):
    from encoder_v3.film_resnet.model import FRPretraining
    from encoder_v3.hierarchical_revised.tasks import task_masks
    from encoder_v3.patch_latent.model import Pretraining
    from encoder_v3.shared_mae.data import load_data,sample_from_dataset

    out = args.output/"s4_decoder_execution"
    out.mkdir(exist_ok=False)
    policy = read(args.output/"acceptance_policy.json")
    assert read(args.output/"s1_baseline/executor_reference.json")["status"]=="PASS"
    assert (args.output/"candidate_unit_tests.xml").is_file()
    pred_tol,grad_tol = policy["gpu_prediction"],policy["gpu_gradient"]
    cfg,settings,device = config_device(args)
    data,resolver = load_data()
    old,compat = Pretraining(cfg).to(device),FRPretraining(cfg,options("B0"),seed=17).to(device)
    allowed = [np._core.multiarray._reconstruct,np.ndarray,np.dtype,np.dtypes.UInt32DType]
    with torch.serialization.safe_globals(allowed):
        state = torch.load(args.release/"best.pt",map_location="cpu",weights_only=True)["model"]
    old.load_state_dict(state,strict=True)
    load_legacy_weights(compat,state)
    panel = read(args.output/"data_panel.json")
    val = data["VAL"]
    checks, detail, positions = [],[],{}
    candidate = FRPretraining(cfg,options("C"),seed=17).to(device)
    for index,row in enumerate(val.rows):
        if row["id"] not in panel["fixed_val"]:
            continue
        sample = sample_from_dataset(val,index,resolver,device)
        for task in ("A","B1"):
            key = dict(id=sample.identifier,sensor=row["sensor"],task=task)
            for model in (old,compat):
                model.zero_grad(set_to_none=True)
            a,b = [model(sample,kind=task,seed=17) for model in (old,compat)]
            torch.testing.assert_close(a.predictions,b.predictions,**pred_tol)
            torch.testing.assert_close(a.loss,b.loss,**pred_tol)
            assert mask_sha(a.targets)==mask_sha(b.targets)
            a.loss.backward()
            b.loss.backward()
            gradients = dict(compat.named_parameters())
            error = 0.
            for name,p in old.named_parameters():
                mapped = ("legacy.detail_gain."+name.removeprefix("decoder.detail_gain.")
                          if name.startswith("decoder.detail_gain.") else name)
                other = gradients[mapped]
                assert (p.grad is None)==(other.grad is None),name
                if p.grad is not None:
                    torch.testing.assert_close(p.grad,other.grad,**grad_tol)
                    error = max(error,float((p.grad-other.grad).abs().max()))
            checks.append(key | dict(max_prediction_delta=float((a.predictions-b.predictions).detach().abs().max()),
                                     max_gradient_delta=error))
            with torch.no_grad():
                m = compat(sample,kind=task,seed=17,main_path_only=True)
                torch.testing.assert_close(b.main_prediction,m.predictions,**pred_tol)
                if task=="B1":
                    count = int(torch.count_nonzero(b.detail_contribution))
                    assert (count>0)==(row["sensor"]=="S2"),key
                    detail.append(key | dict(nonzero_contribution_pixels=count))
                original = candidate(sample,kind=task,seed=17)
                keeps,targets = task_masks(sample,cfg,task,17,0)
                for label,mask in (("hidden",targets),("qa_invalid",[~c.valid for c in sample.channels])):
                    changed = replace(sample,channels=[replace(c,values=torch.where(m,1024-c.values,c.values))
                                                        for c,m in zip(sample.channels,mask,strict=True)])
                    alt = candidate(changed,kind=task,seed=17)
                    torch.testing.assert_close(original.predictions,alt.predictions,**pred_tol)
                    torch.testing.assert_close(original.main_prediction,alt.main_prediction,**pred_tol)
                    assert mask_sha(original.targets)==mask_sha(alt.targets)==mask_sha(targets)
                    if label=="qa_invalid":
                        torch.testing.assert_close(original.loss,alt.loss,**pred_tol)
                    else:
                        assert float((original.loss-alt.loss).abs())>0
                bands,encoded,specs = candidate.views(sample,keeps,targets)
                expected = candidate.decoder.predict(encoded,specs)
                selected = next(iter(expected))
                reduced = [replace(s,selection=torch.zeros_like(s.selection)) for s in specs]
                selection = specs[selected].selection.clone()
                ids = selection.flatten().nonzero().flatten()
                selection.flatten()[ids[1::2]] = False
                reduced[selected] = replace(specs[selected],selection=selection)
                actual = candidate.decoder.predict(encoded,reduced)
                torch.testing.assert_close(actual[selected],expected[selected][::2],**pred_tol)
                if task=="A":
                    slots = candidate.encoder.spectral(bands,encoded.descriptor,encoded.geometry)
                    direct = slots.observed.any(1).reshape(*slots.shape)
                    for level,(f,s,blocks) in enumerate(zip(encoded.features,encoded.support,candidate.encoder.stages)):
                        q,k,cov = candidate.position[level].coordinates(encoded.geometry,s,level,blocks[0].reduction)
                        tag = f"{sample.identifier}_{level}"
                        positions[tag+"_query_xy"] = q.cpu().numpy()
                        positions[tag+"_key_xy"] = k.cpu().numpy()
                        positions[tag+"_key_coverage"] = cov.cpu().numpy()
                        positions[tag+"_feature"] = f.cpu().numpy()
                        positions[tag+"_observed"] = direct.cpu().numpy()
                        direct = torch.nn.functional.max_pool2d(direct[None,None].float(),3,stride=2,padding=1)[0,0]>0
            print("S4 compatibility",key,flush=True)
            del a,b,m,original,alt,bands,encoded,specs,expected,actual
    write(out/"decoder_terminal_gate.json",dict(status="PASS",checks=checks,
          target_query_subset="Every fixed validation observation/task; identical requested target entries"))
    write(out/"legacy_detail_gate.json",dict(status="PASS",checks=detail))
    np.savez_compressed(out/"position_diagnostics.npz",**positions)
    del old,compat,state
    torch.cuda.empty_cache()
    capacity(args,candidate,data,resolver,settings,device,out)


def capacity(args,model,data,resolver,settings,device,out):
    from encoder_v3.shared_mae.data import sample_from_dataset
    from encoder_v3.hierarchical_revised.tasks import available_tasks

    maxima = read(args.output/"geometry_census_summary.json")["maximum"]
    train = data["TRAIN"]
    lookup = {r["id"]:i for i,r in enumerate(train.rows)}
    cases = []
    tasks = [(sensor,records["native_values"]["id"]) for sensor,records in maxima.items()]
    tasks += [(s,r["bands"]["id"]) for s,r in maxima.items() if r["bands"]["id"]!=r["native_values"]["id"]]
    model.train()
    try:
        for sensor,id_ in tasks:
            sample = sample_from_dataset(train,lookup[id_],resolver,device)
            for task in available_tasks(sample,model.config):
                model.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                metrics = {}
                result = model(sample,kind=task,seed=settings["seed"],metrics=metrics)
                torch.cuda.synchronize(device)
                forward = time.perf_counter()-started
                assert torch.isfinite(result.loss) and torch.isfinite(result.predictions).all()
                result.loss.backward()
                torch.cuda.synchronize(device)
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),float("inf"),error_if_nonfinite=True))
                cases.append(dict(sensor=sensor,id=id_,task=task,loss=float(result.loss.detach()),gradient_norm=norm,
                                  targets=len(result.predictions),forward_seconds=forward,
                                  backward_seconds=time.perf_counter()-started-forward,
                                  peak_bytes=torch.cuda.max_memory_allocated(device),metrics=metrics))
                write(out/"capacity_progress.json",dict(cases=cases))
                print("S4 capacity",sensor,id_,task,"passed",flush=True)
                del result
            del sample
    except Exception as error:
        write(out/"capacity.json",dict(status="FAILED_CAPACITY" if isinstance(error,torch.OutOfMemoryError)
              else "FAILED_CORRECTNESS",sensor=sensor,id=id_,task=task,cases=cases,error=repr(error)))
        raise
    write(out/"capacity.json",dict(status="PASS",cases=cases,optimizer_updates=0,
          parameters=sum(p.numel() for p in model.parameters()),
          trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)))


def update_and_resume(args):
    from copy import deepcopy
    from dataclasses import asdict
    from encoder_v3.film_resnet.model import FRPretraining
    from encoder_v3.hierarchical_revised.train import resume,save,update
    from encoder_v3.latent_response.checkpoint import parameter_sha
    from encoder_v3.patch_latent.batching import ProportionalEpochPlan
    from encoder_v3.patch_latent.parallel import ParallelBatch
    from encoder_v3.shared_mae.data import load_data
    from executor_gate import compare_update
    from run_v3_fr import hash_file

    out = args.output/"s4_decoder_execution"
    assert read(out/"capacity.json")["status"]=="PASS"
    policy = read(args.output/"acceptance_policy.json")
    cfg,settings,device = config_device(args)
    settings = settings | {"offload_native_values":float("inf"),"spatial_cuda_graphs":False}
    data,resolver = load_data()
    train = data["TRAIN"]
    other,other_resolver = load_data()
    plan = ProportionalEpochPlan(train.rows,17,settings["batch_size"])
    lookup = {r["id"]:i for i,r in enumerate(train.rows)}
    choices = options("C")
    initial = FRPretraining(cfg,choices,seed=17)
    binding = dict(arm="C", options=asdict(choices), initial_parameter_hash=parameter_sha(initial),
                   frozen_identity=read(args.output/"baseline_identity.json"),
                   candidate_files={p.name:hash_file(p) for p in
                                    (Path(__file__).resolve().parents[1]/"src/encoder_v3/film_resnet").glob("*.py")})
    attempts = 0

    def setup():
        model = deepcopy(initial).to(device).train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=settings["lr"],weight_decay=settings["weight_decay"])
        actual = [id(p) for g in opt.param_groups for p in g["params"]]
        assert len(actual)==len(set(actual)) and set(actual)=={id(p) for p in model.parameters() if p.requires_grad}
        return model,opt

    def state(index):
        return dict(cursor=index,epoch=0,spent_updates=0,observations_seen=0)

    def one(model,opt,runner,dual):
        nonlocal attempts
        parallel = (ParallelBatch(model,train,resolver,settings,settings["devices"],
                    secondary_data=(other["TRAIN"],other_resolver)) if dual else None)
        try:
            attempts += 1
            log = update(model,train,plan.batch(runner["cursor"]),lookup,resolver,opt,runner,settings,6990,
                         backward_batch=parallel.backward if parallel else None)
            return model,opt,log
        finally:
            if parallel:
                parallel.close()
    cases = []
    try:
        for index in (0,len(plan)-1):
            a,oa = setup()
            b,ob = setup()
            left,right = state(index),state(index)
            single = one(a,oa,left,False)
            dual = one(b,ob,right,True)
            comparison = compare_update(single,dual,policy)
            assert left==right
            cases.append(dict(batch_index=index, observations=single[2]["batch_size"],errors=comparison))
            print("S4 candidate optimizer single vs dual",single[2]["batch_size"],flush=True)
            del a,b,oa,ob,single,dual
        write(out/"one_step_equivalence.json",dict(status="PASS",cases=cases,optimizer_updates=attempts))
        live,opt = setup()
        active = state(0)
        one(live,opt,active,True)
        saved = out/"resume_gate_checkpoint.pt"
        save(saved,live,opt,active,binding)
        restarted,new_opt = setup()
        restored = {}
        resume(saved,restarted,new_opt,restored,binding)
        assert restored==active and parameter_sha(restarted)==parameter_sha(live)
        continuous = one(live,opt,active,True)
        replay = one(restarted,new_opt,restored,True)
        assert restored==active
        comparison = compare_update(continuous,replay,policy)
        write(out/"resume_gate.json",dict(status="PASS",optimizer_updates=3,
              checkpoint_sha256=hash_file(saved),next_update=active["cursor"],errors=comparison,
              optimizer_registration="Every trainable parameter registered exactly once"))
    except Exception as error:
        write(out/"candidate_update_failure.json",dict(status="FAILED_CORRECTNESS",
              attempted_optimizer_updates=attempts,cases=cases,error=repr(error)))
        raise
