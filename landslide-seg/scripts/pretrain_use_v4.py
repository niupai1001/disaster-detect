"""v4 initialization, differential LR and the inherited v3 five-objective contract."""
from pretrain_use_v3 import (ROOT,inputs,load_rows,sample_rows,task_view,deterministic,lr,
    checkpoint,restore,atomic_save,mean_error,spatial_error,gradient_error,consistency,views,calibrate)
import pretrain_use_v3 as v3
import torch
from landslide_seg.models.use_v4_mit import USEv4
from landslide_seg.models.use_v4_reconstruction import Reconstruction


def initialize(device='cpu',task=None,encoder=None):
    deterministic(); torch.manual_seed(1234)
    encoder=USEv4() if encoder is None else encoder
    torch.manual_seed(1235)
    return Reconstruction(encoder,task).to(device)


def optimizer(model):
    if model.task is not None:
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=6e-4,weight_decay=.01)
    front=[]; new=[]
    for name,p in model.named_parameters():
        if p.requires_grad: (front if name.startswith('use.') and not name.startswith('use.adapter.') else new).append(p)
    return torch.optim.AdamW([dict(params=front,lr=1e-4,peak_lr=1e-4),dict(params=new,lr=6e-4,peak_lr=6e-4)],weight_decay=.01)


def update(model,opt,split,stats,step,calibration,device):
    # Hooks read gradients, without copying any frozen weights in forward.
    paths={}
    handles=[]
    for name,module in [('spectral',model.use.readout),('adapter',model.use.adapter)]:
        def hook(mod,args,out,key=name):
            value=out[0] if isinstance(out,tuple) else out
            if value.requires_grad:
                value.register_hook(lambda g: paths.setdefault(key,[]).append(float(g.detach().norm())))
        handles.append(module.register_forward_hook(hook))
    try: result=v3.update(model,opt,split,stats,step,calibration,device)
    finally:
        for h in handles: h.remove()
    if any(p.grad is not None or p.requires_grad for p in model.use.mit.parameters()):
        raise RuntimeError('MiT freeze violated')
    result['path_gradients']=paths
    return result
