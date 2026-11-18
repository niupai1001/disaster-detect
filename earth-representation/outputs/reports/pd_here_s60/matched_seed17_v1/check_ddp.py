import json,sys,tempfile,sysconfig
sys.path.insert(0,sysconfig.get_path('stdlib'))
import ssl
from pathlib import Path
sys.path.insert(0,str(Path('scripts').resolve()))
from train_pd_here_s60 import BASE,Study,models,update,write_json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from pd_here_geotoken_dualvit_r3.model import masked_reconstruction_loss

def worker(rank,root):
    torch.set_num_threads(2)
    device=f'cuda:{rank}'; torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    d=Path(root)
    dist.init_process_group('gloo',init_method=(d/'store').as_uri(),rank=rank,world_size=2,timeout=__import__('datetime').timedelta(seconds=35))
    try:
        study=Study(BASE/'texture_study',device)
        run=Path('outputs/pd_here_s60/matched_seed17_v1')
        state=torch.load(run/'s60_last.pt',map_location='cpu',weights_only=True)
        results={}
        for n,rows in [(16,study.rows['train'][:16]),(12,study.rows['train'][-12:])]:
            m,o=models(study,([.15]*10,[.1]*10))['s60']
            m.load_state_dict(state['model']);o.load_state_dict(state['optimizer'])
            ddp=DistributedDataParallel(m,device_ids=[rank],broadcast_buffers=False)
            x,q,views=study.inputs(rows,0)
            sl=slice(rank*n//2,(rank+1)*n//2)
            xx=tuple(v[sl] for v in x); qq=tuple(v[sl] for v in q)
            with ddp.no_sync():
                keep,bands=(a[sl] for a in views[0])
                (.5*masked_reconstruction_loss(ddp(xx,qq,keep,bands)['prediction'],xx,qq,keep,bands,'A')).backward()
            keep,bands=(a[sl] for a in views[1])
            (.5*masked_reconstruction_loss(ddp(xx,qq,keep,bands)['prediction'],xx,qq,keep,bands,'B1')).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);o.step()
            if rank==0:
                ref,ro=models(study,([.15]*10,[.1]*10))['s60']
                ref.load_state_dict(state['model']);ro.load_state_dict(state['optimizer'])
                update(ref,ro,x,q,views,'paired')
                results[n]=max(float((a.detach()-b.detach()).abs().max()) for a,b in zip(m.parameters(),ref.parameters(),strict=True))
            dist.barrier()
        if rank==0:
            write_json(d/'ddp.json',dict(status='PASS' if max(results.values())<=2e-5 else 'FAIL',max_parameter_difference=results,tolerance=2e-5,backend='gloo',ranks=2))
    finally:dist.destroy_process_group()

if __name__=='__main__':
    with tempfile.TemporaryDirectory() as temp:
        result=Path('outputs/pd_here_s60/matched_seed17_v1/ddp_check.json')
        try:
            torch.multiprocessing.spawn(worker,args=(temp,),nprocs=2,join=True)
            write_json(result,json.loads((Path(temp)/'ddp.json').read_text()))
        except Exception as error:
            write_json(result,dict(status='BLOCKED',backend='gloo',error=str(error)))
            raise


