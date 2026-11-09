"""New mechanism-B high-resolution CUDA engineering capacity; zero updates."""
import gc,json,hashlib,math,time
from pathlib import Path
import torch
from .participation_encoder import ParticipationEncoder
from .participation_layout import AreaEquivalentHillRate
from .model import BandObservation


def main():
    torch.manual_seed(92);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.75)
    curve=AreaEquivalentHillRate(.8092248954506467,4.010953026221307,1.219268281571556)
    model=ParticipationEncoder(rate_curve=curve,selection_seed=0,hyper_hidden=64).cuda().train();r=Path(__file__).resolve().parent/'reports'
    report=dict(status='RUNNING',architecture=model.get_extra_state(),curve_status='explicit unconfirmed candidate; anisotropic area-GSD rule also not frozen',scope='One generated visible band; full Encoder forward/backward, latent objective only',optimizer_updates=0,device_uuid=str(torch.cuda.get_device_properties(0).uuid),cases=[],limits=['No real high-resolution observation admitted','Synthetic SRF/reflectance are test fixtures, never native supervision','No Decoder, effective batch8, optimizer, checkpoint or data pipeline cost acceptance','Forward wall time includes CPU geometry; not pure GPU kernel timing'])
    def save():(r/'participation_gpu_capacity.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    save()
    for fov in (60.,120.):
        for gsd in (.5,1.21):
            side=math.ceil(fov/gsd);setup=time.perf_counter();b=BandObservation(torch.rand(side,side,device='cuda'),torch.ones(side,side,dtype=torch.bool,device='cuda'),torch.tensor([450.,500.,550.],device='cuda'),torch.tensor([0.,1.,0.],device='cuda'),(gsd,0.,100.,0.,-gsd,200.),'EPSG:32650','generated')
            torch.cuda.synchronize();setup=time.perf_counter()-setup;origin=(100.,200.-side*gsd);timed=[]
            for repetition in range(3):
                model.zero_grad(set_to_none=True);gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();z=model([b],origin_xy=origin);torch.cuda.synchronize();forward=time.perf_counter()-start
                start=time.perf_counter();z.tokens.square().mean().backward();torch.cuda.synchronize();backward=time.perf_counter()-start
                if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise FloatingPointError('Invalid mechanism-B gradients')
                if repetition:timed.append(dict(forward_seconds=forward,backward_seconds=backward,total_seconds=forward+backward,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
                audit=z.participation_audit[0];del z
            row=dict(gsd_m=gsd,requested_fov_m=fov,actual_fov_m=side*gsd,native_shape=[side,side],setup_seconds=setup,participation=audit,measurements=timed,padding_ratio=0.)
            report['cases'].append(row);save();print(json.dumps(row),flush=True);del b;model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    report['status']='MECHANISM_B_CAPACITY_COMPLETE_NOT_FORMAL_ACCEPTANCE';root=Path(__file__).resolve().parent;report['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),root/'participation_encoder.py',root/'participation_layout.py',root/'adaptive_window.py')};save()


if __name__=='__main__':main()
