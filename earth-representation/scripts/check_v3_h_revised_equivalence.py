"""Compare compressed decoder routes to independently expanded baseline routes."""
import ssl
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

import argparse
import copy
import json
import time
import numpy as np
import torch
from encoder_v3.hierarchical_revised import model as module
from encoder_v3.hierarchical_revised import readout
from encoder_v3.hierarchical_revised.model import Config
from encoder_v3.hierarchical_revised.profile import CONFIG, real_case, hashes
from encoder_v3.hierarchical_revised.routes import combine
from encoder_v3.hierarchical_revised.tasks import Pretraining
from encoder_v3.hierarchical_revised.attention import Metrics


class Expanded:
    def __init__(self, sparse, groups, group_keys, offset, latents, globals=0):
        def expanded(start,end):
            query, cells = groups(start,end)
            if group_keys is None:
                return query,cells
            rows,keys = group_keys.take(cells)
            return query[rows],keys
        self.graph = combine(sparse, expanded, offset, latents, globals)

    def __call__(self,start,end):
        return self.graph(start,end)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor",required=True)
    parser.add_argument("--rank",default="max")
    parser.add_argument("--task",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--warmup",action="store_true",help="Warm each path before the measured forward/backward")
    args=parser.parse_args()
    if args.output.exists():
        parser.error("New output path required")
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    cfg=Config(**json.loads(CONFIG.read_text())["model"])
    sample=real_case(args.sensor,args.rank,torch.device("cuda:0"))
    torch.manual_seed(17)
    fast=Pretraining(cfg).cuda()
    original=copy.deepcopy(fast)
    records=[]
    for model in (fast,original):
        if model is original:
            module.Factorized=Expanded
            def original_tile(decoder, descriptor, geography, local, coarse, chunk, metrics, inverse):
                q = decoder.condition(torch.cat((descriptor[inverse], geography), dim=-1))
                q = decoder.local.read_projected(q,*local[0],local[1],chunk,"decoder_local",metrics)
                q = decoder.global_read.read_projected(q,*coarse[0],coarse[1],chunk,"decoder_coarse",metrics)
                return decoder.output(q).squeeze(-1)
            readout.tile = original_tile
            original.config = module.Config(**{**vars(cfg), "decoder_query_chunk":4096})
            original.decoder.config = original.config
        if args.warmup:
            warm = model(sample,kind=args.task,seed=17)
            warm.loss.backward()
            model.zero_grad(set_to_none=True)
            del warm
            torch.cuda.synchronize()
        metrics=Metrics()
        start=time.perf_counter()
        result=model(sample,kind=args.task,seed=17,metrics=metrics)
        result.loss.backward()
        torch.cuda.synchronize()
        records.append((result.predictions.detach().cpu(),float(result.loss.detach()),
                        {name:p.grad.detach().cpu() for name,p in model.named_parameters() if p.grad is not None},
                        time.perf_counter()-start,metrics.summarize()))
        del result
    a,b=records
    errors={}
    for name in a[2]:
        x,y=a[2][name],b[2][name]
        errors[name]=dict(max_abs=float((x-y).abs().max()),
                          relative_l2=float((x-y).norm()/y.norm().clamp_min(1e-12)),
                          close=bool(torch.allclose(x,y,atol=1e-5,rtol=2e-4)))
    pairs_equal=all(a[4]["attention"][name]["pairs"]==row["pairs"] for name,row in b[4]["attention"].items())
    report=dict(sample=sample.identifier,task=args.task,warmed=args.warmup,source_sha256=hashes(CONFIG),
                prediction_max_abs=float((a[0]-b[0]).abs().max()),losses=[a[1],b[1]],
                gradients=errors,attention_pairs_equal=pairs_equal,
                fast_seconds=a[3],reference_seconds=b[3])
    report["status"]="PASS" if (pairs_equal and torch.allclose(a[0],b[0],atol=1e-5,rtol=1e-5)
                               and all(e["close"] for e in errors.values())) else "FAIL"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(report["status"],report["prediction_max_abs"],report["fast_seconds"],report["reference_seconds"])
    if report["status"]!="PASS":
        raise SystemExit(1)


if __name__=="__main__":
    main()
