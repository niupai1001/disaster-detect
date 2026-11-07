"""Matched real A readout packed-local execution, zero parameter updates."""
import json,time,statistics,hashlib
from pathlib import Path
import torch
from .data import NativeOpticalDataset,sha256
from .pilot_protocol import load_study
from .prepare_pilot import bounds,member
from .tasks import spatial_task
from .participation_layout import AreaEquivalentHillRate
from .participation_encoder import ParticipationEncoder,prepare_participation_task
from .participation_decoder import MatrixADecoder
from .adaptive_streaming import stream_adaptive_task
from .joint import PlannedView
from .plan_prefetch import move_views


def main():
    r=Path(__file__).resolve().parent/'reports';torch.set_num_threads(2);torch.manual_seed(0);torch.cuda.set_per_process_memory_fraction(.75);study=load_study(r/'pilot_study_candidate.json');ds=NativeOpticalDataset(study['release'],study['contracts'],role='TRAIN');identifier='obs_01ef02ab37643eb26fca2ccd';record=next(x for x in ds.rows if x['id']==identifier);frozen=next(x for x in study['training_members'] if x['id']==identifier)
    if ds.identities!=study['source_identities'] or member(record)!=frozen:raise ValueError('Reviewed identity mismatch')
    obs=ds[next(i for i,x in enumerate(ds.rows) if x['id']==identifier)];spec=next(x for x in study['schedule'][0]['tasks'] if x['observation']==identifier);task=spatial_task(obs,spec['A_rectangle']);left,bottom,_,_=bounds(record);curve=AreaEquivalentHillRate(.8092248954506467,4.010953026221307,1.219268281571556);plan=prepare_participation_task(task,rate_curve=curve,selection_seed=0,origin_xy=(left,bottom),chunk_pixels=16384,cell_indexed=True);view=move_views([PlannedView(task,plan,ds.engineering_train_stds(obs),'optical')],'cuda:0')[0]
    encoder=ParticipationEncoder(rate_curve=curve,selection_seed=0,hyper_hidden=64,local_batch_size=1024).cuda();head=MatrixADecoder(indexed=True,cell_indexed=True,bulk_geometry=True).cuda()
    def run():
        encoder.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True)
        return stream_adaptive_task(encoder,head,view.task,view.plan,train_stds=view.train_stds,chunk_pixels=16384,loss_weight=1.,backward=True)

    initial=[p.detach().clone() for m in (encoder,head) for p in m.parameters()];rows=[];reference=None
    for packed in (False,True):
        encoder.packed_local_output=packed;times=[];peaks=[]
        for repeat in range(3):
            predictions=[];supports=[]
            def collect(module,args,result):predictions.append(result[0].detach().clone());supports.append(result[1].detach().clone())
            handle=head.register_forward_hook(collect);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            try:result=run();torch.cuda.synchronize();elapsed=time.perf_counter()-start
            finally:handle.remove()
            if repeat:times.append(elapsed);peaks.append(torch.cuda.max_memory_allocated())
            prediction=torch.cat(predictions).cpu();support=torch.cat(supports).cpu();gradient=torch.cat([torch.zeros_like(p).flatten() if p.grad is None else p.grad.flatten() for m in (encoder,head) for p in m.parameters()]).detach().cpu().double()
        if reference is None:reference=(prediction,support,gradient,result)
        if not torch.equal(support,reference[1]):raise ValueError('Support changed')
        maximum=float((prediction[support]-reference[0][support]).abs().max());relative=float((gradient-reference[2]).norm()/reference[2].norm());loss=abs(result['loss']-reference[3]['loss'])
        if not bool(torch.isfinite(gradient).all() & torch.isfinite(prediction[support]).all()):raise FloatingPointError('Nonfinite numerical comparison')
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:run();torch.cuda.synchronize()
        operators={e.key:dict(count=e.count,self_cpu_seconds=e.self_cpu_time_total/1e6) for e in profiler.key_averages() if e.key in ('aten::select','aten::unbind','aten::stack','aten::index_select','cudaLaunchKernel','aten::as_strided')}
        row=dict(packed_local_output=packed,seconds=times,median_seconds=statistics.median(times),peak_allocated_bytes=max(peaks),prediction_max_abs=maximum,gradient_relative_l2=relative,task_loss_abs=loss,tokens=result['tokens'],supported_pixels=int(support.sum()),operators=operators,default_adopted=False);rows.append(row);print(json.dumps(row),flush=True)
    for old,p in zip(initial,[p for m in (encoder,head) for p in m.parameters()],strict=True):
        if not torch.equal(old,p):raise ValueError('Parameter changed')
    report=dict(status='PACKED_LOCAL_REAL_MATCHED_MEASUREMENT_NOT_FORMAL_ACCEPTANCE',observation_id=identifier,kind='A',source_identities=ds.identities,curve_frozen=False,optimizer_updates=0,parameters_unchanged=True,device_uuid=str(torch.cuda.get_device_properties(0).uuid),rows=rows,speedup=rows[0]['median_seconds']/rows[1]['median_seconds'],source_sha256={n:sha256(Path(__file__).with_name(n)) for n in ('participation_packed_probe.py','participation_encoder.py','adaptive_window.py','participation_decoder.py','participation_geometry.py')},limits=['One reviewed full-native A view only, not effective batch8 throughput','Two timing samples after warmup; profiled passes excluded','No real optimizer-step/resume comparison; default remains off'])
    (r/'participation_packed_comparison.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
