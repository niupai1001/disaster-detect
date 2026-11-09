"""One reviewed full-native A view CPU-dispatch/Python profile, zero updates."""
import cProfile,json,pstats,time
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
    run();torch.cuda.synchronize();python=cProfile.Profile();start=time.perf_counter()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        python.enable();result=run();torch.cuda.synchronize();python.disable()
    elapsed=time.perf_counter()-start;stats=pstats.Stats(python);rows=[]
    for (file,line,name),(primitive,total,self_time,cumulative,callers) in stats.stats.items():rows.append(dict(file=file,line=line,function=name,calls=total,primitive_calls=primitive,self_seconds=self_time,cumulative_seconds=cumulative))
    events=[dict(operator=e.key,count=e.count,self_cpu_seconds=e.self_cpu_time_total/1e6,total_cpu_seconds=e.cpu_time_total/1e6) for e in profiler.key_averages()]
    report=dict(status='PROFILE_COMPLETE_NOT_THROUGHPUT_ACCEPTANCE',observation_id=identifier,kind='A',tokens=result['tokens'],supported_pixels=sum(b['count'] for b in result['bands']),optimizer_updates=0,curve_frozen=False,device_uuid=str(torch.cuda.get_device_properties(0).uuid),profiled_seconds=elapsed,python_by_self=sorted(rows,key=lambda x:x['self_seconds'],reverse=True)[:35],python_by_cumulative=sorted(rows,key=lambda x:x['cumulative_seconds'],reverse=True)[:25],operators_by_self=sorted(events,key=lambda x:x['self_cpu_seconds'],reverse=True)[:40],attention_operators=[e for e in events if 'attention' in e['operator']],scalar_sync_operators=[e for e in events if 'local_scalar_dense' in e['operator'] or e['operator']=='aten::item'],source_sha256={n:sha256(Path(__file__).with_name(n)) for n in ('participation_profile.py','participation_encoder.py','adaptive_window.py','participation_decoder.py','participation_geometry.py','streaming.py')},limits=['CPU dispatch and Python profile includes GPU waits; not GPU-kernel duration attribution','Profiler overhead prevents comparison with throughput timing','One reviewed A view only; not full-batch or optimizer/save profiling'])
    (r/'participation_profile.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=report['status'],profiled_seconds=elapsed,python_top=report['python_by_self'][:5])))


if __name__=='__main__':main()
