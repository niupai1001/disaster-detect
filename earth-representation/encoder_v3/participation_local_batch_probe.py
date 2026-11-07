"""Matched local-group batching for mechanism B; no learned or sampling changes."""
import gc,hashlib,json,time,statistics
from pathlib import Path
import torch
from .participation_encoder import ParticipationEncoder
from .participation_layout import AreaEquivalentHillRate
from .model import BandObservation


def main():
    torch.manual_seed(93);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.75);root=Path(__file__).resolve().parent
    curve=AreaEquivalentHillRate(.8092248954506467,4.010953026221307,1.219268281571556);model=ParticipationEncoder(rate_curve=curve,selection_seed=0,hyper_hidden=64,local_batch_size=64).cuda().train()
    b=BandObservation(torch.rand(240,240,device='cuda'),torch.ones(240,240,dtype=torch.bool,device='cuda'),torch.tensor([450.,500.,550.],device='cuda'),torch.tensor([0.,1.,0.],device='cuda'),(.5,0.,100.,0.,-.5,200.),'EPSG:32650','generated');reference=None;rows=[]
    for batch in (64,256,1024):
        model.local_batch_size=batch;times=[];peaks=[]
        for repeat in range(3):
            model.zero_grad(set_to_none=True);gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();z=model([b],origin_xy=(100.,80.));loss=z.tokens.square().mean();loss.backward();torch.cuda.synchronize();seconds=time.perf_counter()-start
            if repeat:times.append(seconds);peaks.append(torch.cuda.max_memory_allocated())
            output=z.tokens.detach().cpu();grad=torch.cat([p.grad.flatten() for p in model.parameters()]).detach().cpu();signature=hashlib.sha256(json.dumps(z.group_ids).encode()).hexdigest();audit=z.participation_audit
            if not bool(torch.isfinite(output).all()&torch.isfinite(grad).all()):raise FloatingPointError('Invalid batching output/gradient')
            del z,loss
        if reference is None:reference=(output,grad,signature,audit)
        else:
            if signature!=reference[2] or audit!=reference[3]:raise ValueError('Grouping or participation changed')
        delta=grad.double()-reference[1].double();rows.append(dict(local_group_batch=batch,seconds=times,median_seconds=statistics.median(times),peak_allocated_bytes=max(peaks),tokens=len(output),group_signature=signature,participation=audit,prediction_max_abs=float((output-reference[0]).abs().max()),gradient_relative_l2=float(delta.norm()/reference[1].double().norm().clamp_min(1e-12)),gradient_max_abs=float(delta.abs().max()),default_adopted=False))
        print(json.dumps(rows[-1]),flush=True)
    for row in rows:row['speed_ratio_vs_64']=rows[0]['median_seconds']/row['median_seconds']
    report=dict(status='MATCHED_LOCAL_GROUP_BATCH_MEASURED_NOT_FORMAL_ACCEPTANCE',optimizer_updates=0,device_uuid=str(torch.cuda.get_device_properties(0).uuid),curve=model.rate_curve.get_config(),curve_frozen=False,global_effective_batch_changed=False,scope='same generated single visible band, all identical merge/native groups, full Encoder forward/backward, latent fixture objective',rows=rows,limits=['Local group execution batch is not observation effective batch','No optimizer update/recovery or Decoder supervision comparison','One explicit candidate curve; broader band/P/QA/native-grid resources still pending'],source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),root/'participation_encoder.py',root/'participation_layout.py',root/'adaptive_window.py')})
    (root/'reports/participation_local_batch_comparison.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
