"""Compare full and tail updates of the historical model on the actual devices."""
from copy import deepcopy
import time

import torch

from run_v3_fr import read, write


def compare_update(reference, candidate, limits):
    left, lo, lr = reference
    right, ro, rr = candidate
    assert (lr["cursor"], lr["batch_size"], lr["eligible"], lr["lr"]) == (
        rr["cursor"], rr["batch_size"], rr["eligible"], rr["lr"])
    assert [(r["id"],[(k,v["targets"]) for k,v in r["tasks"].items()]) for r in lr["members"]] == [
        (r["id"],[(k,v["targets"]) for k,v in r["tasks"].items()]) for r in rr["members"]]
    error = {}
    def check(label, a,b,tol):
        delta = (a.detach()-b.detach()).float()
        stats = error.setdefault(label, dict(max_abs=0.,max_relative_l2=0.))
        stats["max_abs"] = max(stats["max_abs"], float(delta.abs().max()))
        stats["max_relative_l2"] = max(stats["max_relative_l2"],
            float(delta.norm()/a.detach().float().norm().clamp_min(1e-12)))
        torch.testing.assert_close(b.detach(),a.detach(),**tol)

    check("loss", torch.tensor(lr["loss"]),torch.tensor(rr["loss"]),limits["gpu_prediction"])
    for (name,a),(other,b) in zip(left.named_parameters(),right.named_parameters(),strict=True):
        assert name == other and (a.grad is None) == (b.grad is None), name
        check("parameter",a,b,limits["gpu_prediction"])
        if a.grad is not None:
            try:
                check("gradient",a.grad,b.grad,limits["gpu_gradient"])
                for field in ("step","exp_avg","exp_avg_sq"):
                    check(field,lo.state[a][field],ro.state[b][field],limits["gpu_gradient"])
            except AssertionError as exc:
                raise AssertionError(f"{name}: {exc}") from exc
    return error


def run(args):
    from encoder_v3.hierarchical_revised.train import update
    from encoder_v3.patch_latent.acceleration import common_shapes, prepare_spatial_graphs
    from encoder_v3.patch_latent.batching import ProportionalEpochPlan
    from encoder_v3.patch_latent.parallel import ParallelBatch
    from encoder_v3.shared_mae.data import load_data
    from train_patch_latent import Config, Pretraining
    from encoder_v3.shared_mae.geometry import metric_geometry
    from encoder_v3.latent_response.types import NativeGrid
    from types import SimpleNamespace

    out = args.output / "s1_baseline"
    assert read(out/"baseline_replay.json")["status"]=="PASS"
    policy = read(args.output/"acceptance_policy.json")
    config = read(args.release/"config.json")
    settings = config["training"] | {"offload_native_values":float("inf")}
    device = settings["device"]
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(17)
    initial = Pretraining(Config(**config["model"]))
    data,resolver = load_data()
    train = data["TRAIN"]
    other, other_resolver = load_data()
    plan = ProportionalEpochPlan(train.rows,17,settings["batch_size"])
    shapes,_ = common_shapes(train.rows,40.)
    lookup = {r["id"]:i for i,r in enumerate(train.rows)}
    started = time.perf_counter()
    cases = []
    attempts = 0
    try:
        for index in (0,len(plan)-1):
            rows = plan.batch(index)
            row_shapes = {metric_geometry([SimpleNamespace(grid=NativeGrid(tuple(b["transform"]),tuple(b["shape"]),b["crs"]))
                                           for b in row["bands"]],40.).shape for row in rows}
            captured = [shape for shape in shapes if shape in row_shapes]
            assert captured
            def one(mode):
                nonlocal attempts
                model = deepcopy(initial).to(device).train()
                opt = torch.optim.AdamW(model.parameters(),lr=settings["lr"],weight_decay=settings["weight_decay"])
                state = dict(cursor=index,epoch=0,spent_updates=0,observations_seen=0)
                parallel = None
                try:
                    if mode!="single_eager":
                        parallel = ParallelBatch(model,train,resolver,settings,settings["devices"],
                                                 secondary_data=(other["TRAIN"],other_resolver))
                    if mode=="dual_historical_graph":
                        prepare_spatial_graphs(model,captured)
                        prepare_spatial_graphs(parallel.replica,captured)
                    attempts += 1
                    log = update(model,train,rows,lookup,resolver,opt,state,settings,6990,
                                 backward_batch=parallel.backward if parallel else None)
                    assert state["spent_updates"] == 1 and state["cursor"]==index+1
                    return model,opt,log
                finally:
                    if parallel:
                        parallel.close()
            single = one("single_eager")
            repeated = one("single_eager")
            record = dict(batch_index=index, observations=len(rows), captured_shapes=captured,
                          single_repeat=compare_update(single,repeated,policy))
            del repeated
            for mode in ("dual_eager","dual_historical_graph"):
                record[mode] = compare_update(single,one(mode),policy)
            cases.append(record)
            print("S1 executor",len(rows),"observations passed",flush=True)
            del single
    except Exception as exc:
        write(out/"executor_reference.json",dict(status="FAILED_CORRECTNESS",cases=cases,
              batch_index=index, attempted_optimizer_updates=attempts,error=repr(exc)))
        raise
    write(out/"executor_reference.json",dict(status="PASS",cases=cases,optimizer_updates=attempts,
                                             wall_seconds=time.perf_counter()-started))
