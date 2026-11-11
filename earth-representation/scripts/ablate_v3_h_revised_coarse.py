"""Bounded paired engineering ablation; never promotes a candidate or starts a formal run."""
import ssl
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

import argparse
import hashlib
import json
import os
import time

import torch

from encoder_v3.shared_mae.data import load_data, sample_from_dataset
from encoder_v3.hierarchical_revised.coarse_shared import SharedCoarseDecoder
from encoder_v3.hierarchical_revised.model import Config, Decoder
from encoder_v3.hierarchical_revised.profile import hashes
from encoder_v3.hierarchical_revised.tasks import Pretraining


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config_v3_h_revised_shared_coarse_candidate.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--geometry-cache", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("New output directory required")
    document = json.loads(args.config.read_text())
    if document.get("formal_training_authorized") is not False or document.get("candidate_only") is not True:
        parser.error("An independent engineering candidate configuration is required")
    plan = document["ablation"]
    if plan["updates"] > 16 or len(plan["seeds"]) > 3:
        parser.error("This launcher is capped at three paired 16-update engineering runs")
    args.output.mkdir(parents=True)
    os.environ["V3_H_R_GEOMETRY_CACHE"] = str(args.geometry_cache.resolve())
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    datasets, resolver = load_data()
    cfg = Config(**document["model"])
    samples, selection = {}, []
    for role, sensors, fractions in (
        ("TRAIN", ("S2","L8","L9","EnMAP"), (.5,)),
        ("VAL", ("S2","L8"), (.25,.5,.75)),
    ):
        dataset = datasets[role]
        for sensor in sensors:
            choices = sorted((sum(b["shape"][0]*b["shape"][1] for b in row["bands"]), row["id"], i)
                             for i,row in enumerate(dataset.rows) if row["sensor"] == sensor)
            for fraction in fractions:
                values, identifier, index = choices[min(int(len(choices)*fraction), len(choices)-1)]
                sample = sample_from_dataset(dataset, index, resolver, device)
                assert sample.role == role
                samples[identifier] = sample
                selection.append(dict(id=identifier,role=role,sensor=sensor,quantile=fraction,native_values=values))
    train = [(row["id"], task) for row in selection if row["role"] == "TRAIN" for task in ("A","B1")]
    validate = [(row["id"], task) for row in selection if row["role"] == "VAL" for task in ("A","B1")]
    diagnostic = [(row["id"], task) for row in selection if row["role"] == "TRAIN" and
                  row["sensor"] in ("EnMAP","L9") for task in ("A","B1")]
    report = dict(status="RUNNING",source_sha256=hashes(args.config),selection=selection,
                  model_seeds=plan["seeds"],mask_seed=17,updates=plan["updates"],
                  validation_mask_epoch=11,training_mask_epochs=[0,1],variants={},
                  scope="Small-panel engineering ablation, not convergence or generalization acceptance")
    def save():
        (args.output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    def evaluate(model):
        rows = []
        model.eval()
        with torch.no_grad():
            for identifier,task in validate+diagnostic:
                sample = samples[identifier]
                result = model(sample,kind=task,seed=17,epoch=11)
                rows.append(dict(id=identifier,role=sample.role,task=task,loss=float(result.loss),
                                 channel_mse={str(c):float(v) for c,v in result.channel_loss.items()},
                                 target_pixels=len(result.predictions)))
                del result
        model.train()
        return rows
    save()
    try:
        for seed in plan["seeds"]:
            for variant, decoder in (("per_pixel",Decoder),("shared_coarse",SharedCoarseDecoder)):
                torch.manual_seed(seed)
                model = Pretraining(cfg,decoder_class=decoder).to(device)
                optimizer = torch.optim.AdamW(model.parameters(),lr=document["engineering"]["lr"],
                                              weight_decay=document["engineering"]["weight_decay"])
                record = dict(seed=seed,variant=variant,initial_hash=state_hash(model),
                              before=evaluate(model),steps=[])
                report["variants"][f"{seed}:{variant}"] = record
                for step in range(plan["updates"]):
                    identifier, task = train[step % len(train)]
                    optimizer.zero_grad(set_to_none=True)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    result = model(samples[identifier],kind=task,seed=17,epoch=step//len(train))
                    result.loss.backward()
                    if not torch.isfinite(result.loss) or not all(
                        p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                        raise ValueError("Nonfinite engineering update")
                    optimizer.step()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    record["steps"].append(dict(update=step+1,id=identifier,task=task,loss=float(result.loss.detach()),
                                                wall_seconds=time.perf_counter()-start))
                    del result
                    if (step+1)%8 == 0:
                        print(seed,variant,step+1,flush=True)
                        save()
                record["after"] = evaluate(model)
                record["final_hash"] = state_hash(model)
                checkpoint = args.output/f"{seed}_{variant}_engineering.pt"
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=document,
                                variant=variant,seed=seed,updates=plan["updates"],formal_training=False),checkpoint)
                record["checkpoint"] = str(checkpoint)
                del model,optimizer
                torch.cuda.empty_cache()
                save()
            assert report["variants"][f"{seed}:per_pixel"]["initial_hash"] == report["variants"][f"{seed}:shared_coarse"]["initial_hash"]
        report["status"] = "COMPLETED_ENGINEERING_ONLY"
        report["adopted"] = False
    except Exception as error:
        report.update(status="FAIL",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
    print(report["status"],str(args.output/"report.json"),flush=True)


if __name__ == "__main__":
    main()
