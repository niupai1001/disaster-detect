"""Actual pretrained segmentation path, strict reload, BN and stream acceptance."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import torch
from landslide_seg.config import load_segformer_config
from landslide_seg.data.segformer_dataset import read_manifest, load_stats, collate
from landslide_seg.data.use_dataset import USESegDataset
from landslide_seg.models.factory import build_model
from landslide_seg.training.update_training import FullBatchStream
from landslide_seg.training.segformer_trainer import load_checkpoint


def main():
    torch.set_num_threads(4)
    cfg=load_segformer_config(ROOT/'configs/segformer/use/physical.yaml')
    s=cfg['segformer']
    rows=read_manifest(ROOT/s['data']['manifest'])
    stats=load_stats(ROOT/s['data']['stats'])
    ds=USESegDataset(rows[:8],stats,augment=True,seed=1234,preload=True)
    ds.mixed=True
    stream=FullBatchStream(ds,8,1234)
    first=next(stream)
    state=stream.state_dict()
    expected=next(stream)
    restored=next(FullBatchStream(ds,8,1234,state))
    for key in ('x','band_valid','pixel_valid','band_metadata'):
        torch.testing.assert_close(expected['image'][key],restored['image'][key],equal_nan=True)
    assert expected['image']['band_ids']==restored['image']['band_ids']
    model=build_model(s).to('cuda:1').train()
    batch=first['image'].to('cuda:1')
    # Verify the actual supervised logits depend on every new component.
    pred=model(batch)['logits']
    loss=torch.nn.functional.binary_cross_entropy_with_logits(pred[:,0],first['mask'].to('cuda:1'))
    loss.backward()
    gradients={n:float(p.grad.norm()) for n,p in model.named_parameters() if n.startswith(('use.','bridge.')) and p.grad is not None}
    assert all(any(n.startswith(prefix) and v>0 for n,v in gradients.items()) for prefix in ('use.tokenizer','use.attention','use.pool','use.metadata','bridge'))
    assert all(not m.training for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm))
    before=float(loss.detach())
    opt=torch.optim.AdamW(model.parameters(),lr=6e-4)
    for _ in range(100):
        opt.zero_grad(set_to_none=True)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(model(batch)['logits'][:,0],first['mask'].to('cuda:1'))
        loss.backward();opt.step()
    after=float(loss.detach())
    assert after<before, (before,after)
    path=ROOT/'outputs/use/s2_10band_v4r1_p20/smoke_physical/1234/best.pt'
    loaded,_=load_checkpoint(path,device='cpu')
    loaded.eval()
    torch.testing.assert_close(loaded(first['image'])['logits'],loaded(first['image'])['logits'])
    out=dict(gradients=gradients,overfit_bce_before=before,overfit_bce_after=after,
             stream_resume=True,strict_checkpoint_load=True,bn_frozen=True,formal_result=False)
    (ROOT/'outputs/use/engineering_acceptance.json').write_text(json.dumps(out,indent=1),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='gradients'}),flush=True)


def diagnose():
    torch.set_num_threads(4)
    cfg=load_segformer_config(ROOT/'configs/segformer/use/mean.yaml')['segformer']
    rows=read_manifest(ROOT/cfg['data']['manifest'])[:8]
    stats=load_stats(ROOT/cfg['data']['stats'])
    ds=USESegDataset(rows,stats,augment=False,seed=1234,preload=True)
    batch=collate([ds[i] for i in range(len(ds))])['image']
    model,_=load_checkpoint(ROOT/'outputs/use/s2_10band_v4r1_p20/use_mean/1234/best.pt',device='cpu')
    model.eval()
    with torch.no_grad():
        encoded=model.use(batch)['features'];bridge=model.bridge(encoded)
    valid=batch['pixel_valid'].any(1)
    def summary(x):
        values=x.permute(0,2,3,1)[valid]
        return dict(mean=values.mean(0).tolist(),std=values.std(0).tolist(),
                    max_abs=float(values.abs().max()),finite=bool(torch.isfinite(values).all()))
    out=dict(sample_ids=[r['sample_id'] for r in rows],normalized_input=summary(batch['x']),
             use_feature=summary(encoded),bridge_feature=summary(bridge),
             interpretation='仅分布诊断，不证明退化因果；不修改实验结构或训练预算。')
    (ROOT/'outputs/use/input_distribution.json').write_text(json.dumps(out,indent=1,ensure_ascii=False),encoding='utf-8')


def evaluate_queue(include_pretrained=False):
    import subprocess
    import time
    names=('baseline','mean','self_mean','self_query','physical','random','capacity','mean_mixed','physical_mixed')
    if include_pretrained:names=(*names,'pretrained_mixed')
    results={}
    while len(results)<len(names):
        for name in names:
            directory=ROOT/f'outputs/use/s2_10band_v4r1_p20/use_{name}/1234'
            summary=directory/'train_summary.json'
            if not summary.exists() or name in results:continue
            training=json.loads(summary.read_text())
            if not training.get('complete') or training['global_step']!=2000:raise ValueError('Incomplete formal run')
            reports={}
            for subset in (('full',) if name=='baseline' else ('full','rgb','rgb_nir','six','unseen5')):
                target=directory/f'eval/eval_dev_{subset}.json'
                existing=json.loads(target.read_text()) if target.exists() else {}
                if 'use_bins' not in existing.get('overall',{}).get('objects',{}):
                    command=[sys.executable,'-X','utf8','scripts/evaluate_segformer.py','--checkpoint',str(directory/'best.pt'),
                        '--split','val','--device','cpu','--skip-efficiency','--band-subset',subset,'--tag','dev_'+subset]
                    with (ROOT/f'outputs/use/eval_{name}_{subset}.log').open('w',encoding='utf-8') as log:
                        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                report=json.loads(target.read_text())
                if report['n_samples']!=669:raise ValueError('Incomplete DEV evaluation')
                reports[subset]=report
            if abs(reports['full']['event_mean_iou']-training['best_val_iou'])>1e-4:
                raise ValueError('Independent evaluation differs from training selection score')
            results[name]=reports
            (ROOT/'outputs/use/supervised_evaluations.json').write_text(json.dumps(results,indent=1),encoding='utf-8')
            print('Evaluated',name,flush=True)
        if len(results)<len(names):
            state=json.loads((ROOT/'outputs/use/queue_status.json').read_text())
            if state['status']=='failed':raise RuntimeError(state.get('error','Training queue failed'))
            time.sleep(30)
    base=results['baseline']['full']
    alarms={}
    for name,views in results.items():
        full=views['full']
        alarms[name]=dict(iou_drop=base['event_mean_iou']-full['event_mean_iou'],
            tiny_recall_drop=base['overall']['objects']['use_bins']['1_4']['object_recall_evaluable']-
                full['overall']['objects']['use_bins']['1_4']['object_recall_evaluable'])
    (ROOT/'outputs/use/supervised_acceptance.json').write_text(json.dumps(dict(ok=True,
        accepted='数据、预算及独立评价完整；不是性能改善验收',alarms=alarms,models=list(results)),indent=1,ensure_ascii=False),encoding='utf-8')


def mechanism():
    torch.set_num_threads(4)
    path=ROOT/'outputs/use/s2_10band_v4r1_p20/use_physical/1234/best.pt'
    if not path.exists():path=ROOT/'outputs/use/s2_10band_v4r1_p20/smoke_physical/1234/best.pt'
    model,ck=load_checkpoint(path,device='cpu');model.eval()
    cfg=ck['config']['segformer'];rows=read_manifest(ROOT/cfg['data']['manifest'])[:2]
    ds=USESegDataset(rows,load_stats(ROOT/cfg['data']['stats']),augment=False,preload=True)
    a=collate([ds[i] for i in range(2)])['image'];captured=[]
    hook=model.use.attention.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach().clone()))
    with torch.no_grad():
        y=model(a)['logits'];changed=dict(a);changed['band_metadata']=a['band_metadata'].flip(1)
        mismatched=model(changed)['logits'];changed=dict(a);changed['x']=a['x']*1.1;model(changed)
    hook.remove()
    with torch.no_grad():
        _,wa=model.use.attention(captured[0],captured[0],captured[0],need_weights=True)
        _,wb=model.use.attention(captured[-1],captured[-1],captured[-1],need_weights=True)
    md=float((y-mismatched).abs().max());ad=float((wa-wb).abs().max())
    if md<=0 or ad<=0:raise ValueError('Metadata or content intervention has no observable response')
    out=dict(metadata_mismatch_logits_max_abs=md,content_intervention_attention_max_abs=ad,
        checkpoint=str(path),sample_ids=[r['sample_id'] for r in rows],
        command='python -X utf8 scripts/audit_use.py --mechanism',formal_result=False,
        note='机制连通检查，不是性能增益或因果证明')
    (ROOT/'outputs/use/mechanism_acceptance.json').write_text(json.dumps(out,indent=1,ensure_ascii=False),encoding='utf-8')


def artifacts():
    import hashlib
    paths=set()
    for name in ('baseline','mean','self_mean','self_query','physical','random','capacity','mean_mixed','physical_mixed','pretrained_mixed'):
        root=ROOT/f'outputs/use/s2_10band_v4r1_p20/use_{name}/1234'
        for pattern in ('*.pt','*.json','*.jsonl','eval/*.json','eval/*.csv'):paths.update(root.glob(pattern))
    for root,patterns in ((ROOT/'configs/segformer/use',('*.yaml',)),(ROOT/'outputs/use/pretraining',('*.pt','*.json','*.jsonl')),
        (ROOT/'outputs/use/external',('*manifest.json','*acceptance.json','*contract.json'))):
        for pattern in patterns:paths.update(root.glob(pattern))
    paths.update((ROOT/'outputs/use').glob('*acceptance.json'))
    lines=[hashlib.sha256(path.read_bytes()).hexdigest()+'  '+str(path.relative_to(ROOT)).replace('\\','/') for path in sorted(paths) if path.is_file()]
    (ROOT/'outputs/use/artifacts.sha256').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def pretraining_acceptance():
    import numpy as np
    import pretrain_use as pretrain
    from landslide_seg.config import set_seed
    set_seed(1234,True);torch.set_num_threads(4)
    root=ROOT/'outputs/use/external';rows=json.loads((root/'pilot_manifest.json').read_text())
    groups={sensor:[r for r in rows if r['sensor']==sensor] for sensor in ('s2','landsat')}
    core=json.loads((ROOT/'manifests/v4r1/stats_core25.json').read_text())
    stats={'s2':(np.asarray(core['mean'])/10000,np.asarray(core['std'])/10000),
        'landsat':(np.full(6,.2),np.full(6,.1))}  # Engineering constants; no fitted formal statistics.
    model=pretrain.Reconstruction().to('cuda:1');opt=torch.optim.AdamW(model.parameters(),lr=.0006,weight_decay=.01)
    rng=np.random.default_rng(1234)
    def update(net,optimizer,gen):
        chosen=[groups[s][i] for s in ('s2','landsat') for i in gen.integers(len(groups[s]),size=4)]
        inp,target,valid,metadata,known=pretrain.make_batch(chosen,gen,stats,root,'cuda:1')
        optimizer.zero_grad(set_to_none=True);loss=pretrain.masked_l1(net(inp,metadata,known),target,valid)
        loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1,error_if_nonfinite=True);optimizer.step()
        return [r['file'] for r in chosen],float(loss.detach()),inp['band_ids']
    torch.cuda.reset_peak_memory_stats('cuda:1');update(model,opt,rng)
    path=ROOT/'outputs/use/pretraining_resume_engineering.pt'
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),path)
    expected=update(model,opt,rng)
    restored=pretrain.Reconstruction().to('cuda:1');optimizer=torch.optim.AdamW(restored.parameters(),lr=.0006,weight_decay=.01)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    restored.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
    rng.bit_generator.state=saved['rng'];torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    actual=update(restored,optimizer,rng)
    assert expected==actual
    max_abs=0.
    for name,value in model.state_dict().items():
        other=restored.state_dict()[name];max_abs=max(max_abs,float((value-other).abs().max()))
        torch.testing.assert_close(value,other,atol=1e-7,rtol=1e-6)
    (ROOT/'outputs/use/pretraining_engineering_acceptance.json').write_text(json.dumps(dict(ok=True,
        real_pilot_samples=True,batch_size=8,sensor_exposures=[4,4],resume_next_samples_bands_and_loss_exact=True,
        next_parameters_max_abs=max_abs,next_parameters_atol=1e-7,next_parameters_rtol=1e-6,
        next_loss=actual[1],peak_allocated_mib=torch.cuda.max_memory_allocated('cuda:1')/2**20,
        formal_result=False,note='仅工程常量归一化，未拟合正式统计，不计入10000次预训练预算。'),indent=1,ensure_ascii=False),encoding='utf-8')


def interfaces():
    """Check the complete segmenter at odd spatial sizes with synthetic band sets."""
    from landslide_seg.data.optical_observation import OpticalObservation,collate_optical
    torch.set_num_threads(4);torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    model,_=load_checkpoint(ROOT/'outputs/use/s2_10band_v4r1_p20/use_physical/1234/best.pt',device='cuda:1')
    model.eval();report={}
    def observation(c):
        return OpticalObservation(torch.randn(c,127,129),tuple('synthetic_'+str(i) for i in range(c)),
            torch.tensor([[.5,0,10.]]*c),torch.tensor([[True,False,True]]*c),
            torch.ones(c,dtype=torch.bool),torch.ones(c,127,129,dtype=torch.bool),{})
    for c in (3,4,6,10,13):
        sample=observation(c);batch=collate_optical([sample]).to('cuda:1')
        model.zero_grad(set_to_none=True);output=model(batch)['logits'];output.square().mean().backward()
        assert output.shape==(1,1,127,129) and torch.isfinite(output).all()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        with torch.inference_mode():
            expected=model(batch)['logits'];reordered=model(collate_optical([sample.subset(sample.band_ids[::-1])]).to('cuda:1'))['logits']
            padded=model(collate_optical([sample,observation(13)]).to('cuda:1'))['logits'][:1]
        torch.testing.assert_close(expected,reordered,atol=1e-5,rtol=1e-4)
        torch.testing.assert_close(expected,padded,atol=1e-5,rtol=1e-4)
        report[c]=dict(shape=list(output.shape),permutation_max_abs=float((expected-reordered).abs().max()),padding_max_abs=float((expected-padded).abs().max()))
    (ROOT/'outputs/use/segmentation_interface_acceptance.json').write_text(json.dumps(dict(ok=True,
        synthetic_only=True,atol=1e-5,rtol=1e-4,results=report),indent=1),encoding='utf-8')


def transfer():
    """Verify that downstream initialization changes only USE, not the other modules."""
    import hashlib
    torch.manual_seed(1234)
    random=build_model(load_segformer_config(ROOT/'configs/segformer/use/physical_mixed.yaml')['segformer'])
    torch.manual_seed(1234)
    pretrained=build_model(load_segformer_config(ROOT/'configs/segformer/use/pretrained_mixed.yaml')['segformer'])
    reference=random.state_dict();loaded=pretrained.state_dict()
    outside=[name for name in reference if not name.startswith('use.')]
    assert all(torch.equal(reference[name],loaded[name]) for name in outside)
    source=torch.load(ROOT/'outputs/use/pretraining/best.pt',map_location='cpu',weights_only=False)['use']
    assert all(torch.equal(source[name],pretrained.use.state_dict()[name]) for name in source)
    assert any(not torch.equal(reference[name],loaded[name]) for name in reference if name.startswith('use.'))
    path=ROOT/'outputs/use/pretraining/best.pt'
    (ROOT/'outputs/use/transfer_acceptance.json').write_text(json.dumps(dict(ok=True,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        identical_non_use_tensors=len(outside),strict_use_tensors=len(source),seed=1234),indent=1),encoding='utf-8')


def costs():
    """Measure complete inference using real observations, with a shared device/batch."""
    import time
    torch.set_num_threads(4)
    from landslide_seg.config import set_seed
    precision_settings=set_seed(1234,True)
    from run_use import NAMES
    records=[]
    for name in (*NAMES,'pretrained_mixed'):
        checkpoint=ROOT/f'outputs/use/s2_10band_v4r1_p20/use_{name}/1234/best.pt'
        if not checkpoint.exists():continue
        model,saved=load_checkpoint(checkpoint,device='cpu');model.eval()
        cfg=saved['config']['segformer'];rows=read_manifest(ROOT/cfg['data']['manifest'])[:8]
        from landslide_seg.data.segformer_dataset import S2SegDataset
        dataset_type=USESegDataset if name!='baseline' else S2SegDataset
        ds=dataset_type(rows,load_stats(ROOT/cfg['data']['stats']),augment=False,preload=True)
        batch=collate([ds[i] for i in range(8)])['image']
        with torch.inference_mode(),torch.profiler.profile(with_flops=True) as profile:model(batch)
        supported_flops=sum(event.flops for event in profile.key_averages())/8
        model=model.to('cuda:1');batch=batch.to('cuda:1')
        with torch.inference_mode():
            for _ in range(10):model(batch)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats('cuda:1');start=time.perf_counter()
            for _ in range(30):model(batch)
            torch.cuda.synchronize();seconds=time.perf_counter()-start
        records.append(dict(model=name,parameters=sum(p.numel() for p in model.parameters()),
            supported_operator_flops_per_image=supported_flops,images_per_second=240/seconds,
            peak_allocated_mib=torch.cuda.max_memory_allocated('cuda:1')/2**20,
            checkpoint=str(checkpoint.relative_to(ROOT))))
        del model,batch;torch.cuda.empty_cache()
    out=dict(command='python -X utf8 scripts/audit_use.py --cost',device=torch.cuda.get_device_name(1),
        batch_size=8,precision='FP32',precision_settings=precision_settings,warmup=10,repeats=30,sample_ids=[r['sample_id'] for r in rows],
        scope='完整前向，含观测校验；不含文件读取与CPU到GPU搬运。FLOPs仅计profiler支持的算子，不是总计算量。',records=records)
    (ROOT/'outputs/use/costs.json').write_text(json.dumps(out,indent=1,ensure_ascii=False),encoding='utf-8')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group()
    for option in ('diagnose','mechanism','interfaces','artifacts','pretraining-acceptance','evaluate-queue','evaluate-pretrained','transfer','cost'):group.add_argument('--'+option,action='store_true')
    args=parser.parse_args()
    if args.diagnose:diagnose()
    elif args.mechanism:mechanism()
    elif args.interfaces:interfaces()
    elif args.artifacts:artifacts()
    elif args.pretraining_acceptance:pretraining_acceptance()
    elif args.evaluate_queue:evaluate_queue()
    elif args.evaluate_pretrained:evaluate_queue(include_pretrained=True)
    elif args.transfer:transfer()
    elif args.cost:costs()
    else:main()
