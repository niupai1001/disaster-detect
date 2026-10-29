"""Real initialization/data/gradient tests, full replay and bounded deployment timing."""
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
import yaml
import run_use_mit_dual_routes as run
from landslide_seg.models.use_mit_dual_routes import visible_rgb
from landslide_seg.models.use_v3_reconstruction import packed_state_sha256, FrozenUSE


def recovery(route, device, splits, stats):
    out=run.BASE/f'engineering/trial_{route}'
    target=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
    if target['step'] != 200: raise ValueError('Replay requires finished trial200')
    model=run.initialize(route,device);opt=run.optimizer(model)
    saved=run.pt.restore(model,opt,out/'recovery_pre.pt',target['identity'])
    assert saved['step']==199
    run.v4.schedule(opt,200,run.matrix()['pretrain_'+route])
    model.train()
    record=run.update(model,opt,splits['TRAIN'],stats,200,run.rt.read(run.CALIBRATION),device)
    current=run.pt.checkpoint(model,opt,200,None,target['identity'],[])
    for key in ('model','use','optimizer','rng'):
        if not run.rt.states_equal(current[key],target[key]): raise ValueError('Recovery mismatch: '+route+'/'+key)
    # Numerical update records exclude wall time, checkpoint/validation overhead.
    for key in ('microbatches','path_gradients','files','sensors','gradient_norm'):
        if not run.rt.states_equal(record[key],target['history'][-1][key]): raise ValueError('Recovery record mismatch: '+key)
    result=dict(status='PASS',restored_step=199,replayed_step=200,replay_updates=1,new_successful_updates=0,
        model_sha256=current['model_sha256'],checkpoint_sha256=run.sha(out/'last.pt'),
        optimizer_and_all_rng_exact=True)
    run.rt.write_json(out/'recovery_verification.json',result)
    return result


def real_consumers(device, splits, stats):
    config=yaml.safe_load(run.CONFIG.read_text());train,val=run.datasets(config)
    batch=run.rt.collate([train[0],train[1]])
    results={}
    for route in ('C0','A','B'):
        encoder=run.load_encoder(route);torch.manual_seed(1235)
        model=run.Segmenter(encoder).to(device).train()
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0006)
        before=packed_state_sha256(model.use.encoder)
        prediction=model(batch['image'].to(device))
        loss=run.rt._seg_loss(config,prediction,batch['mask'].to(device),batch['valid'].to(device))
        loss.backward()
        run.v4.verify_frozen(model,opt,'segmentation',before)
        if not torch.isfinite(loss) or model.decoder.classifier.weight.grad.norm()==0:
            raise ValueError('Real task head gradient missing')
        results[route]=dict(loss=float(loss.detach()),prediction_shape=list(prediction.shape),
            task_gradient=float(model.decoder.classifier.weight.grad.norm()),encoder_sha256=before,
            encoder_final_sha256=packed_state_sha256(model.use.encoder),optimizer_updates=0)
        del model,opt,encoder
    # Paired actual augmented rows prove the inverse normalization and view-level masks.
    rows=[splits['TRAIN']['s2'][0],splits['TRAIN']['s2'][1]]
    original=run.load_rows(rows,stats,device,451)
    rgb,support=visible_rgb(original)
    view,hidden,_=run.pt.task_view(original,'spectral',500,metadata_mode='known')
    visible,mask=visible_rgb(view)
    if (mask & ~support).any(): raise ValueError('Student gained support')
    changed=copy.deepcopy(original);changed['x'][hidden]=123456
    changed_view,_,_=run.pt.task_view(changed,'spectral',500,metadata_mode='known')
    torch.testing.assert_close(visible,visible_rgb(changed_view)[0],rtol=0,atol=0)
    return dict(segmentation=results,rgb_hidden_truth_invariant=True,train_members=len(train),dev_members=len(val),
                augmentations=original['spatial_metadata'])


def teacher_free_validation(device, splits, stats):
    out=run.BASE/'engineering/trial_B'
    if not (out/'summary.json').exists():return None
    model=run.initialize('B',device)
    saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model'],strict=True)
    before=packed_state_sha256(model)
    def forbidden(*args,**kwargs):raise AssertionError('Validation called teacher')
    model.teacher.forward=forbidden
    corrected=run.validate(model,splits,stats,device,200)
    original=run.rt.read(out/'validation_200.json')
    for sensor,values in corrected['sensors'].items():
        assert values['means']==original['sensors'][sensor]['means']
        for left,right in zip(values['rows'],original['sensors'][sensor]['rows']):
            assert left['file']==right['file'] and left['raw']==right['raw'] and left['counts']==right['counts']
            assert 'align' not in left and 'coverage' not in left
    assert packed_state_sha256(model)==before
    run.rt.write_json(out/'validation_no_teacher_200.json',corrected)
    return dict(status='PASS',teacher_calls=0,validation_members=599,
        all_five_raw_errors_and_counts_exact=True,model_sha256_unchanged=before,
        historical_record_preserved='validation_200.json',corrected_record='validation_no_teacher_200.json',
        new_optimizer_updates=0,
        source_change='Validation now skips frozen teacher diagnostics. Training three-view update and 0.20 objective unchanged; exact step200 replays verify numerical identity.')


def verify(device):
    run.preflight();run.pt.deterministic()
    out=run.BASE/'engineering';out.mkdir(parents=True,exist_ok=True)
    common=out/'common_initial.pt'
    if not common.exists():
        encoder=run.load_encoder('A');torch.manual_seed(1235)
        head=run.Segmenter(copy.deepcopy(encoder))
        run.pt.atomic_save(dict(frontend={k:v.clone() for k,v in encoder.state_dict().items() if k.split('.')[0] in run.FRONTEND},
            adapter=copy.deepcopy(encoder.adapter.state_dict()),
            task_head={k:v.clone() for k,v in head.state_dict().items() if not k.startswith('use.')},
            source_v3_sha256=run.sha(run.V3),source_mit_sha256=run.sha(run.MIT/'model.safetensors')),
            common)
        del encoder,head
    command=[sys.executable,'-X','utf8','-m','pytest','tests/test_use_mit_dual_routes.py','-q',
             '--basetemp',str(out/'pytest_verify_tmp')]
    process=subprocess.run(command,cwd=run.ROOT,text=True,capture_output=True)
    (out/'pytest.log').write_text(process.stdout+process.stderr,encoding='utf8')
    if process.returncode: raise ValueError('Numerical/interface tests failed; see pytest.log')
    splits,stats,_,_=run.pt.inputs('multisensor')
    replays={}
    with run.v4.guarded_device(device) as gate:
        real=real_consumers(device,splits,stats)
        for route in ('A','B'):
            if (out/f'trial_{route}/summary.json').exists(): replays[route]=recovery(route,device,splits,stats)
        validation=teacher_free_validation(device,splits,stats)
    result=dict(status='PASS',pytest_command=command,pytest_log_sha256=run.sha(out/'pytest.log'),
        device_gate=gate,real_consumers=real,recovery=replays,formal_updates=0,
        teacher_free_validation=validation,
        extra_optimizer_updates=sum(v['replay_updates'] for v in replays.values()),
        limitation='Formal 59000-step execution and independent training-agent acceptance remain outstanding')
    run.rt.write_json(out/'verification.json',result)
    print(json.dumps(dict(status='PASS',recovery=list(replays))),flush=True)
    return result


def final_benchmark_available():
    paths=[run.BASE/f'runs/pretrain_{route}/u20000.pt' for route in ('A','B')]
    paths += [run.BASE/f'runs/segment_{route}/best.pt' for route in ('C0','A','B')]
    paths += [run.BASE/f'runs/segment_{route}/summary.json' for route in ('C0','A','B')]
    if not all(path.is_file() for path in paths):return False
    for route in ('C0','A','B'):
        summary=run.rt.read(run.BASE/f'runs/segment_{route}/summary.json')
        if summary.get('complete') is not True or summary.get('updates')!=5000:
            raise ValueError('Final benchmark requires completed 5000-step task heads')
    return True


def benchmark_segmenter(encoder, route, device, checkpoint=None, release_sha=None):
    torch.manual_seed(1235)
    model=run.Segmenter(copy.deepcopy(encoder).cpu())
    encoder_sha=packed_state_sha256(model.use.encoder)
    checkpoint_sha=None
    if checkpoint is not None:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['identity']['job']!=run.matrix()['segment_'+route] or saved['identity']['engineering']:
            raise ValueError('Wrong formal segmentation checkpoint identity')
        if saved['identity']['release_sha256']!=release_sha:
            raise ValueError('Final task head belongs to another release')
        if not 500<=saved['step']<=5000 or saved['step']%500:
            raise ValueError('Best task head must be from a registered 500-step evaluation')
        model.load_state_dict(saved['model'],strict=True)
        if packed_state_sha256(model.use.encoder)!=encoder_sha or saved['use_sha256']!=encoder_sha:
            raise ValueError('Task head checkpoint encoder differs from fixed u20000/C0')
        if packed_state_sha256(model)!=saved['model_sha256']:
            raise ValueError('Task head checkpoint full model hash differs')
        checkpoint_sha=run.sha(checkpoint)
    model=model.to(device).eval()
    return model,checkpoint_sha


def benchmark(device, release):
    """No optimizer steps; initial or fixed-final models, one physical card only."""
    manifest=run.verify_release(release);run.pt.deterministic()
    if torch.device(device).type!='cuda': raise ValueError('Same-card CUDA benchmark required')
    sys.path.insert(0,str(run.ROOT/'tests'))
    from test_use_mit_dual_routes import observation
    records=[];separate=[]
    final=final_benchmark_available()
    release_sha=run.sha(Path(release)/'manifest.json')
    class FrozenConsumer(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__();self.encoder=encoder.requires_grad_(False).eval()
        @torch.no_grad()
        def forward(self, obs):return self.encoder(obs)
    with run.v4.guarded_device(device) as gate:
        for route in ('C0','A','B'):
            source=run.BASE/f'runs/pretrain_{route}/u20000.pt' if final and route!='C0' else None
            encoder=run.load_encoder(route,source).to(device).eval()
            for consumer in ('encoder','frozen','segmenter'):
                task_checkpoint=None;task_checkpoint_sha=None
                if consumer=='encoder': model=encoder
                elif consumer=='frozen': model=FrozenConsumer(copy.deepcopy(encoder)).to(device).eval()
                else:
                    task_checkpoint=run.BASE/f'runs/segment_{route}/best.pt' if final else None
                    model,task_checkpoint_sha=benchmark_segmenter(encoder,route,device,task_checkpoint,release_sha)
                model_sha=packed_state_sha256(model)
                for size in (128,256,512):
                    obs=observation(10,size)
                    obs={k:v.to(device) if torch.is_tensor(v) else v for k,v in obs.items()}
                    # Static reserve avoids intentional OOM. 512 only runs with generous headroom.
                    free,total=torch.cuda.mem_get_info(device)
                    if size==512 and free<12*1024**3:
                        records.append(dict(route=route,consumer=consumer,size=size,unmeasured_reason='free CUDA memory below 12 GiB safety reserve'));continue
                    rounds=[]
                    activation_bytes=[]
                    handles=[]
                    def capture(module,args,output):
                        if torch.is_tensor(output):activation_bytes.append(output.numel()*output.element_size())
                    for module in model.modules():
                        if not list(module.children()):handles.append(module.register_forward_hook(capture))
                    with torch.no_grad():model(obs)
                    for handle in handles:handle.remove()
                    with torch.no_grad():
                        for _ in range(30): model(obs)
                        torch.cuda.synchronize(device)
                        for repeat in range(3):
                            values=[];torch.cuda.reset_peak_memory_stats(device)
                            for _ in range(100):
                                began=time.perf_counter();output=model(obs);torch.cuda.synchronize(device)
                                values.append(time.perf_counter()-began)
                            def byte_count(x):
                                if torch.is_tensor(x):return x.numel()*x.element_size()
                                if isinstance(x,dict):return sum(byte_count(v) for v in x.values())
                                if isinstance(x,(list,tuple)):return sum(byte_count(v) for v in x)
                                return 0
                            rounds.append(dict(median_seconds=float(np.median(values)),p90_seconds=float(np.percentile(values,90)),
                                throughput=1/float(np.median(values)),peak_allocated=torch.cuda.max_memory_allocated(device),
                                peak_reserved=torch.cuda.max_memory_reserved(device),output_bytes=byte_count(output)))
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],profile_memory=True) as profiler:
                        with torch.no_grad(): model(obs)
                        torch.cuda.synchronize(device)
                    records.append(dict(route=route,consumer=consumer,size=size,rounds=rounds,
                        parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                        leaf_activation_bytes=sum(activation_bytes),activation_counting='sum of leaf tensor outputs, not unique live storage',
                        top10=profiler.key_averages().table(sort_by='self_device_time_total',row_limit=10),
                        model_sha256=model_sha,task_head_selection='best' if final and consumer=='segmenter' else None,
                        task_checkpoint=str(task_checkpoint) if task_checkpoint else None,task_checkpoint_sha256=task_checkpoint_sha,
                        source_checkpoint_sha256=run.sha(source) if source else None))
                    torch.cuda.synchronize(device);began=time.perf_counter();packed_state_sha256(model)
                    hash_seconds=time.perf_counter()-began
                    if packed_state_sha256(model)!=model_sha:raise ValueError('Benchmark model changed')
                    cpu=observation(10,size);began=time.perf_counter()
                    moved={k:v.to(device) if torch.is_tensor(v) else v for k,v in cpu.items()}
                    torch.cuda.synchronize(device);transfer_seconds=time.perf_counter()-began
                    began=time.perf_counter();visible_rgb(moved);torch.cuda.synchronize(device)
                    separate.append(dict(route=route,consumer=consumer,size=size,full_hash_seconds=hash_seconds,
                        cpu_to_gpu_seconds=transfer_seconds,rgb_preprocessing_seconds=time.perf_counter()-began))
                if model is not encoder: del model
            del encoder
        splits,stats,_,_=run.pt.inputs('multisensor');rows=[splits['TRAIN']['s2'][0]]
        began=time.perf_counter();run.load_rows(rows,stats,'cpu');io_seconds=time.perf_counter()-began
    run.rt.write_json(run.BASE/'benchmarks'/('final.json' if final else 'initial.json'),
        dict(records=records,separate_overheads=separate,io_hash_decompress_normalize_cpu_seconds=io_seconds,
             device_gate=gate,release_sha256=run.sha(Path(release)/'manifest.json'),
             precision='FP32 TF32 off C10 B1',optimizer_updates=0,io_preprocessing_included=False,
             phase='final' if final else 'initial',formal_task_head_selection='best.pt for all three routes' if final else None,
             limitation='1024 not run for resource safety; 2048+ static only. Full hashes measured separately from neural scopes. Teacher training overhead is recorded in engineering update curves, not deployment encoder timings.',
             static_scale=dict(note='Attention reduction is not strictly linear; extrapolation is not measured latency',
                 input_bytes={str(s):10*s*s*4 for s in (1024,2048,4096)})))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:1');a=p.parse_args()
    torch.set_num_threads(4);verify(a.device)
