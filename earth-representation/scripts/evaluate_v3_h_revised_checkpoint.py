"""Evaluate a frozen checkpoint on the complete original VAL without training."""
import ssl
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    release=args.release.resolve()
    if args.output.exists():
        parser.error("New evaluation output directory required")
    manifest=json.loads((release/"source_manifest.json").read_text())
    for name,expected in manifest.items():
        if hashlib.sha256((release/"runtime"/name).read_bytes()).hexdigest()!=expected:
            raise ValueError("Frozen runtime changed: "+name)
    sys.path.insert(0,str(release/"runtime/src"))
    import torch
    from encoder_v3.hierarchical_revised.train import evaluate,binding_for
    from encoder_v3.hierarchical_revised.model import Config
    from encoder_v3.hierarchical_revised.tasks import Pretraining
    from encoder_v3.shared_mae.data import load_data
    from encoder_v3.shared_mae.train import write
    from encoder_v3.latent_response.checkpoint import parameter_sha
    config=json.loads((release/"config.json").read_text(encoding="utf-8"))
    os.environ["V3_H_R_GEOMETRY_CACHE"]=config["geometry_cache"]
    torch.set_num_threads(2)
    device=torch.device(args.device)
    if device.type=="cuda":
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    datasets,resolver=load_data()
    payload=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    binding=binding_for(config,datasets,resolver)
    if payload.get("schema")!="v3_h_revised_formal_v1" or payload.get("identity")!=binding:
        raise ValueError("Checkpoint binding does not match frozen evaluation")
    model=Pretraining(Config(**config["model"])).to(device)
    model.load_state_dict(payload["model"],strict=True)
    before=parameter_sha(model)
    args.output.mkdir(parents=True)
    started=time.perf_counter()
    def progress(value):
        write(args.output/"status.json",dict(status="EVALUATING",**value,elapsed_seconds=time.perf_counter()-started))
    progress(dict(validation_observations=0,validation_total=len(datasets["VAL"].rows)))
    try:
        result=evaluate(model,datasets["VAL"],resolver,dict(config["training"],device=str(device)),progress=progress)
        assert before==parameter_sha(model)
        result.update(status="COMPLETE",checkpoint=str(args.checkpoint.resolve()),
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            parameter_sha=before,checkpoint_cursor=payload["runner"]["cursor"],identity=binding,
            device=str(device),elapsed_seconds=time.perf_counter()-started,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),optimizer_updates=0)
        write(args.output/"report.json",result)
        write(args.output/"status.json",dict(status="COMPLETE",observations=result["observations"],
                                             tasks=result["tasks"],elapsed_seconds=result["elapsed_seconds"]))
        print("FULL_VAL_COMPLETE",result["tasks"],flush=True)
    except Exception as error:
        write(args.output/"status.json",dict(status="FAILED",error=repr(error),elapsed_seconds=time.perf_counter()-started))
        raise


if __name__=="__main__":
    main()
