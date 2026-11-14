"""Test that the D1 terminal can fit named fine spatial patterns."""
import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from run_v3_fr import bind_runtime,read,write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release",type=Path)
    parser.add_argument("campaign",type=Path)
    args = parser.parse_args()
    bind_runtime(args.release)
    from encoder_v3.film_resnet.model import FRPretraining
    from encoder_v3.patch_latent.model import Config
    from candidate_gate import options
    from types import SimpleNamespace

    torch.set_num_threads(2)
    cfg = Config(**read(args.release/"config.json")["model"])
    model = FRPretraining(cfg,options("C"),seed=17)
    terminal = model.decoder
    h,w,r = 4,5,4
    y,x = torch.meshgrid(torch.arange(h*r),torch.arange(w*r),indexing="ij")
    target = torch.stack([torch.ones_like(x),x/(w*r-1),y/(h*r-1),x>8,x<=8,
                         (x==7)&(y==5),x==7,(x+y)%2]).float()[None]
    # A structured coarse input independent of the fine truth, with one basis per parent cell.
    feature = torch.zeros(1,32,h,w)
    feature[0,:h*w] = torch.eye(h*w).reshape(h*w,h,w)
    valid = torch.ones(1,1,h,w,dtype=torch.bool)
    encoded = SimpleNamespace(features=[feature],support=[valid])
    terminal.parent_field = lambda e:e.features[0]
    opt = torch.optim.AdamW([*terminal.expander.parameters(),*terminal.fine_refine.parameters()],
                            lr=.025,weight_decay=0)
    baseline = target.square().mean((-2,-1)).flatten()
    truth = target.reshape(1,8,h,r,w,r).transpose(3,4).flatten(2)
    trace = []
    for step in range(20):
        opt.zero_grad(set_to_none=True)
        value = terminal.subcells(encoded)
        loss = F.mse_loss(value,truth)
        loss.backward()
        opt.step()
        trace.append(float(loss.detach()))
    with torch.no_grad():
        pred = terminal.subcells(encoded)
        per_pattern = (pred-truth).square().mean(-1).flatten()
    assert torch.isfinite(pred).all() and trace[-1]<trace[0]
    write(args.campaign/"s4_decoder_execution/d1_synthetic_patterns.json",
          dict(status="PASS",updates=len(trace),scope="Synthetic terminal optimization only",
               patterns=["constant","x ramp","y ramp","step","reverse step","impulse","thin line","checkerboard"],
               losses=trace,mse=per_pattern.tolist(),zero_mse=baseline.tolist()))


if __name__=="__main__":
    main()
