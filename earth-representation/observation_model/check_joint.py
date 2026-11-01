"""One real mixed-sensor correctness batch, including both cross directions."""
import json
import math
from pathlib import Path
import numpy as np
import torch

if __name__=='__main__':torch.cuda.init()

from .data import NativeSet, RELEASE
from .paired import NativePairs
from .train_joint import joint_model, masked_loss, cross_loss, balanced_masks


def main():
    torch.manual_seed(0);torch.set_num_threads(4)
    stats=json.loads((RELEASE/'statistics.json').read_text())['bands']
    data=NativeSet('train');model=joint_model(stats).cuda();model.eval();plans={}
    for sensor,c in [('S2',10),('L8',7)]:
        mask=np.zeros((3,c),bool)
        for i in range(3):mask[i,i::3]=True
        plans[sensor]=(list(range(3)),mask,torch.arange(16,device='cuda').expand(3,-1))
    value,logs,exposure,pixels=masked_loss(model,data,stats,plans,True)
    expected=sum(sum(logs[b['name']] for b in data.metadata[s])/len(data.metadata[s]) for s in ['S2','L8'])/2
    assert math.isclose(value,expected,rel_tol=1e-6)
    assert exposure==17 and pixels>0
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)
    assert sum(float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None)>0
    ids=torch.tensor([0],device='cuda');hidden=torch.zeros(1,10,dtype=torch.bool,device='cuda');hidden[:,[0,3]]=True
    inputs=data.bands('S2',ids,stats,hidden);query=data.query('S2',0,ids);tiles=torch.arange(4,device='cuda')
    with torch.no_grad():
        before=model.decoder(model.encode(inputs),query,tiles)[0]
        for i in [0,3]:inputs[i].value+=100
        after=model.decoder(model.encode(inputs),query,tiles)[0]
        permuted=model.decoder(model.encode(list(reversed(inputs))),query,tiles)[0]
    assert torch.equal(before,after),'hidden numerical target leaked'
    assert torch.allclose(after,permuted,atol=2e-6,rtol=1e-5),'band order affected prediction'
    model.zero_grad(set_to_none=True);pairs=NativePairs(data)
    cross,targets=cross_loss(model,pairs,stats,torch.tensor([0],device='cuda'),True)
    assert len(targets)==17 and math.isfinite(cross)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    result=dict(status='PASSED_JOINT_CRITICAL_PATHS',masked_loss=value,manual_equal_band_loss=expected,
        supervised_targets=exposure,hidden_target_leakage_max=float((before-after).abs().max()),
        band_permutation_max=float((after-permuted).abs().max()),cross_loss=cross,cross_targets=targets,
        optimizer_updates=0,data_release=str(RELEASE),boundary='one real mixed batch and one real pair; not task efficacy')
    out=Path(__file__).parent/'outputs/joint_v3';(out/'critical_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
