"""Matched real TRAIN B1 full-target batching comparison; never updates weights."""
import gc,hashlib,json,time,statistics
from pathlib import Path
import torch
from .data import NativeOpticalDataset,to_device
from .geometry_views import common_metric_view
from .pilot_protocol import load_study
from .tasks import fixed_spectral_task
from .streaming import task_signature
from .participation_encoder import ParticipationEncoder,prepare_participation_task
from .participation_layout import AreaEquivalentHillRate
from .adaptive_streaming import stream_adaptive_task
from .model import ReflectanceDecoder


def main():
    root=Path(__file__).resolve().parent;r=root/'reports';study=load_study(r/'pilot_study_candidate.json');torch.manual_seed(study['seed']);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.75)
    dataset=NativeOpticalDataset(study['release'],study['contracts'],role='TRAIN');identifier=study['training_members'][0]['id'];index=next(i for i,x in enumerate(dataset.rows) if x['id']==identifier)
    start=time.perf_counter();obs=dataset[index];obs=common_metric_view(obs,obs.bands[0].crs);stds=dataset.engineering_train_stds(obs);load_seconds=time.perf_counter()-start
    if dataset.identities!=study['source_identities']:raise ValueError('Reviewed data identity differs')
    hidden=study['schedule'][0]['tasks'][0]['B1_hidden'];cpu_task=fixed_spectral_task(obs,hidden);origin=(min(b.affine[2] for b in obs.bands),min(b.affine[5]+b.affine[4]*b.reflectance.shape[0] for b in obs.bands))
    curve=AreaEquivalentHillRate(.8092248954506467,4.010953026221307,1.219268281571556);start=time.perf_counter();plan=prepare_participation_task(cpu_task,rate_curve=curve,selection_seed=study['seed'],origin_xy=origin,chunk_pixels=16384,cell_indexed=True);planning_seconds=time.perf_counter()-start
    start=time.perf_counter();gpu=to_device(obs,'cuda:0');torch.cuda.synchronize();transfer_seconds=time.perf_counter()-start;task=fixed_spectral_task(gpu,hidden)
    if task_signature(task)!=task_signature(cpu_task):raise ValueError('Transfer changed task exposure')
    encoder=ParticipationEncoder(rate_curve=curve,selection_seed=study['seed'],hyper_hidden=64).cuda().train();decoder=ReflectanceDecoder(indexed=True,cell_indexed=True).cuda().train();reference=None;rows=[]
    for batch in (64,1024):
        encoder.local_batch_size=batch;times=[];peaks=[]
        for repeat in range(3):
            encoder.zero_grad(set_to_none=True);decoder.zero_grad(set_to_none=True);preds=[];supports=[]
            def collect(module,inputs,output):preds.append(output[0].detach().clone());supports.append(output[1].detach().clone())
            handle=decoder.register_forward_hook(collect);gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            try:result=stream_adaptive_task(encoder,decoder,task,plan,train_stds=stds,chunk_pixels=16384,loss_weight=1.,backward=True)
            finally:handle.remove()
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            if result['status']!='OK':raise ValueError('Reviewed diagnostic task has no supported target')
            if repeat:times.append(elapsed);peaks.append(torch.cuda.max_memory_allocated())
            pred=torch.cat(preds).cpu();support=torch.cat(supports).cpu();grad=torch.cat([p.grad.flatten() for module in (encoder,decoder) for p in module.parameters()]).detach().cpu()
            if not bool(torch.isfinite(pred[support]).all()&torch.isfinite(grad).all()):raise FloatingPointError('Invalid pixel prediction/gradient')
        if reference is None:reference=(pred,support,grad,result)
        if not torch.equal(reference[1],support):raise ValueError('Support/exposure changed')
        delta=grad.double()-reference[2].double();rows.append(dict(local_group_batch=batch,seconds=times,median_seconds=statistics.median(times),peak_allocated_bytes=max(peaks),pixels=int(support.sum()),tokens=result['tokens'],task_loss=result['loss'],prediction_max_abs=float((pred[support]-reference[0][support]).abs().max()),task_loss_abs=abs(result['loss']-reference[3]['loss']),max_band_nmse_abs=max(abs(a['nmse']-b['nmse']) for a,b in zip(result['bands'],reference[3]['bands']) if a['nmse'] is not None),gradient_relative_l2=float(delta.norm()/reference[2].double().norm().clamp_min(1e-12)),gradient_max_abs=float(delta.abs().max()),per_band=result['bands'],default_adopted=False));print(json.dumps(rows[-1]),flush=True)
    report=dict(status='REAL_REVIEWED_TRAIN_B1_FULL_TARGET_MATCHED_MEASUREMENT',optimizer_updates=0,observation_id=identifier,member_cache_sha256=obs.metadata['cache_sha256'],source_identities=dataset.identities,hidden_bands=hidden,task_signature=task_signature(task),merge_recipe=encoder.merge_recipe,curve_frozen=False,CPU_load_seconds=load_seconds,CPU_planning_seconds=planning_seconds,transfer_seconds=transfer_seconds,rows=rows,speed_ratio=rows[0]['median_seconds']/rows[1]['median_seconds'],scope='single reviewed TRAIN B1 observation; all supported target pixels; full Encoder/Decoder gradients; no parameter update',limits=['Explicit candidate curve is not scientific freezing or formal admission','No optimizer-step/resume or effective batch8 throughput acceptance','Timing excludes initial load/planning/transfer, collected separately','Existing FP32 limits can be compared descriptively; no tolerance changed'],source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),root/'participation_encoder.py',root/'participation_layout.py',root/'adaptive_window.py')})
    (r/'participation_real_B1_batch_comparison.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
