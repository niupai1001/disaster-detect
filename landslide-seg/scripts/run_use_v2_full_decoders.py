"""Two authorized full-network consumers of the same frozen USE v2; old runs untouched."""
import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path
import torch
import yaml
from torch import nn
from torch.nn import functional as F
import run_use_v2 as rt
from pretrain_use_v2 import atomic_save, checkpoint, deterministic, encoder
from landslide_seg.models.frozen_use import FrozenUSE, state_sha256
from landslide_seg.models.use_v2 import canonical_context
from landslide_seg.models.use_downstream import build_visual

OUT=rt.ROOT/'outputs/use_v2_full_decoders'
CONTRACT=rt.ROOT/'configs/use_v2_full_decoders'
SOURCE=rt.BASE/'runs/pretrain_attention_tail/best_unlabeled.pt'
PRETRAINED=rt.ROOT/'weights/segformer/segformer-b0-finetuned-ade-512-512'
RELEASE=OUT/'releases/engineering_v1'
KINDS=('unet','segformer')
DEPENDENCIES=('torch','numpy','scipy','PyYAML','rasterio','pyproj','shapely','matplotlib','transformers')

class FullDecoder(nn.Module):
    def __init__(self,kind):
        super().__init__()
        self.use=FrozenUSE(encoder('attention',SOURCE,'cpu'))
        torch.manual_seed(1235)
        self.visual,_=build_visual('unet' if kind=='unet' else 'patch32',str(PRETRAINED))
    def train(self,mode=True):
        super().train(mode)
        # Match the existing SegFormer frozen-BN comparison protocol.
        for m in self.visual.modules():
            if isinstance(m,nn.modules.batchnorm._BatchNorm): m.eval()
        return self
    def fused(self,obs):
        out=self.use(obs); fine=out['fine']
        context=sum(F.interpolate(canonical_context(x),size=fine.shape[-2:],mode='bilinear',align_corners=False) for x in out['context'])/4
        return fine+context
    def forward(self,obs):
        return self.visual(self.fused(obs))['logits']

def build(job,device='cpu',engineering=False):
    return FullDecoder(job['candidate']).to(device)

def source(job,engineering=False):
    summary=rt.read(SOURCE.parent/'summary.json')
    # Same 30k unlabeled-selected weight for both groups; never re-picked by a downstream score.
    if (not summary['complete'] or summary['updates']!=30000 or rt.sha(SOURCE)!=summary['checkpoints'][SOURCE.name]
            or summary['checkpoints'].get('selected_30000.pt')!=summary['checkpoints'][SOURCE.name]):
        raise ValueError('Frozen 30k source integrity failed')
    return SOURCE

def prepare():
    CONTRACT.mkdir(exist_ok=True)
    config=yaml.safe_load((rt.ROOT/'configs/use_v2/execution.yaml').read_text())
    config['use_v2']['jobs']={k:dict(mode='downstream',candidate=k,steps=5000,warmup=250,task=None,fine=True,source=str(SOURCE),validation_interval=500) for k in KINDS}
    config['use_v2']['budget']=10000
    config['use_v2']['consumer']='fine + mean(upsampled canonical32 contexts); identical parameter-free input to both networks'
    config['segformer']['output_dir']=str(OUT)
    with (CONTRACT/'execution.yaml').open('x',encoding='utf8') as f: yaml.safe_dump(config,f,allow_unicode=True,sort_keys=False)
    shutil.copyfile(rt.ROOT/'configs/use_v2/calibration.json',CONTRACT/'calibration.json')
    from check_research_protocol import audit_config
    rt.write_json(CONTRACT/'preflight.json',audit_config(CONTRACT/'execution.yaml'),True)

def freeze():
    from check_research_protocol import audit_config
    proof=audit_config(CONTRACT/'execution.yaml'); rt.write_json(OUT/'engineering'/'protocol_preflight.json',proof)
    files=[rt.ROOT/'scripts'/n for n in rt.SCRIPTS]
    files+=[*(CONTRACT/n for n in ('execution.yaml','calibration.json')),
            *(rt.ROOT/'manifests/v4r1'/n for n in ('pool_100.jsonl','stats_core25.json','protocol.json')),
            SOURCE,SOURCE.parent/'summary.json',
            *sorted((rt.ROOT/'src/landslide_seg').rglob('*.py')),
            *sorted(p for p in PRETRAINED.rglob('*') if p.is_file())]
    manifest=dict(status='FROZEN',files={p.relative_to(rt.ROOT).as_posix():rt.sha(p) for p in files},
        environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
            dependencies={n:importlib.metadata.version(n) for n in DEPENDENCIES}),
        protocol_preflight=proof.get('ok'),formal_updates=0,
        note='Frozen source, dependencies, config, data members, statistics, USE source and SegFormer initialization; '
             'formal training needs verify() plus a passed acceptance at this digest.')
    RELEASE.mkdir(parents=True,exist_ok=False)
    rt.write_json(RELEASE/'manifest.json',manifest,True)
    (RELEASE/'manifest.sha256').write_text(rt.sha(RELEASE/'manifest.json')+'\n',encoding='utf8')
    return dict(release=str(RELEASE),sha256=verify(),files=len(manifest['files']))

def verify():
    manifest=rt.read(RELEASE/'manifest.json'); digest=rt.sha(RELEASE/'manifest.json')
    if (RELEASE/'manifest.sha256').read_text().strip()!=digest: raise ValueError('Frozen manifest changed')
    for name,expected in manifest['files'].items():
        if rt.sha(rt.ROOT/name)!=expected: raise ValueError('Frozen file changed: '+name)
    for package,version in manifest['environment']['dependencies'].items():
        if importlib.metadata.version(package)!=version: raise ValueError('Dependency changed: '+package)
    if sys.version!=manifest['environment']['python']: raise ValueError('Python changed')
    return digest

def acceptance_path(device):
    return OUT/'engineering'/f"acceptance_{'cuda' if torch.device(device).type=='cuda' else 'cpu'}.json"

def initialization_report(model):
    visual=model.visual; summary=rt.read(SOURCE.parent/'summary.json')
    return dict(network=type(visual).__name__,load_info=getattr(visual,'load_info',None),
        adapter=getattr(visual,'adapter_report',None),classifier=getattr(visual,'cls_report',None),
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        use_sha256=model.use.reference_sha256,use_source=SOURCE.relative_to(rt.ROOT).as_posix(),
        use_source_sha256=rt.sha(SOURCE),use_source_updates=summary['updates'],
        use_source_unlabeled_score=summary['best_score'])

def accept(device):
    """Real forward, backward, strict reload and resume checks; GPU and CPU use the same code path."""
    index=int(torch.device(device).index or 0) if torch.device(device).type=='cuda' else None
    report=dict(passed=False,device=device,config_sha256=rt.sha(CONTRACT/'execution.yaml'),
        release_sha256=verify() if (RELEASE/'manifest.json').exists() else None,
        hardware=dict(cuda_available=torch.cuda.is_available(),torch=torch.__version__,cuda=torch.version.cuda,
            name=torch.cuda.get_device_name(index) if index is not None else None),
        source_sha256={**{n:rt.sha(rt.ROOT/'scripts'/n) for n in rt.SCRIPTS},
            'models/use_v2.py':rt.sha(rt.ROOT/'src/landslide_seg/models/use_v2.py')},
        groups={},recovery={})
    try:
        deterministic(); config=rt.load_configuration()[0]; train_ds,_=rt.datasets(config)
        batch=rt.collate([train_ds[0],train_ds[1]]); x=batch['image'].to(device)
        models={}
        for kind in KINDS:
            job=config['use_v2']['jobs'][kind]; model=build(job,device)
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0006,weight_decay=.01)
            frozen=[model.use.verify(optimizer)]
            model.train()
            if any(m.training for m in model.use.modules()): raise ValueError(f'{kind}: USE left inference mode after train()')
            models[kind]=model.fused(x)
            record=rt.downstream_update(model,optimizer,train_ds,1,config,device)
            silent=[n for n,p in model.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
            if silent: raise ValueError(f'{kind}: trainable parameters without a finite gradient: {silent[:3]}')
            frozen.append(model.use.verify(optimizer))
            payload=checkpoint(model,optimizer,0,0.0,dict(job_id=kind,job=job),[])
            path=OUT/'engineering'/'accept'/kind/'roundtrip.pt'; path.parent.mkdir(parents=True,exist_ok=True); atomic_save(payload,path)
            reloaded=build(job,'cpu'); reloaded.load_state_dict(payload['model'],strict=True)
            if state_sha256(reloaded)!=payload['model_sha256'] or payload['use_sha256']!=frozen[0]:
                raise ValueError(f'{kind}: strict reload differs from the saved state')
            report['groups'][kind]=dict(initialization=initialization_report(model),
                trainable_tensors=len([p for p in model.parameters() if p.requires_grad]),loss=record['loss'],
                gradient_norm=record['gradient_norm'],use_sha256=frozen,use_inference_mode=True,
                strict_reload_sha256=payload['model_sha256'],
                peak_cuda_bytes=torch.cuda.max_memory_allocated(index) if index is not None else None)
        report['shared_use']=report['groups']['unet']['use_sha256'][0]==report['groups']['segformer']['use_sha256'][0]
        if not report['shared_use']: raise ValueError('The two groups loaded different USE states')
        if not torch.equal(models['unet'],models['segformer']): raise ValueError('Identical-fusion input differs between the two groups')
        for kind in KINDS:
            out=OUT/'engineering'/'accept'/kind
            whole,split=out/'whole',out/'split'
            for path in (whole,split): shutil.rmtree(path,ignore_errors=True)
            rt.train(kind,whole,device,engineering=True,engineering_steps=2)
            rt.train(kind,split,device,engineering=True,engineering_steps=2,stop_at=1)
            resumed=rt.train(kind,split,device,engineering=True,engineering_steps=2,resume=True)
            left=torch.load(whole/'last.pt',map_location='cpu',weights_only=False)
            right=torch.load(split/'last.pt',map_location='cpu',weights_only=False)
            same=left['model_sha256']==right['model_sha256'] and left['history']==right['history'] \
                and rt.states_equal(left['optimizer'],right['optimizer']) and rt.states_equal(left['rng'],right['rng'])
            if not same: raise ValueError(f'{kind}: resumed run differs from the uninterrupted run')
            report['recovery'][kind]=dict(exact=True,updates=resumed['updates'],model_sha256=left['model_sha256'])
        report['passed']=True
    except BaseException as error:
        report['error']=repr(error); raise
    finally:
        rt.write_json(acceptance_path(device),report)
    return report

def evaluate(kind,run,device):
    result={}
    for name in ('best','last'):
        result[name]=rt.evaluate_job(kind,run/f'{name}.pt',run/'eval'/name,device)
    return result

def cases(kind,run,device):
    """Six fixed validation cases; no re-selection by result."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans']
    summary=rt.read(run/'summary.json'); ckpt=run/'best.pt'
    if rt.sha(ckpt)!=summary['checkpoints']['best.pt']: raise ValueError('Best checkpoint changed')
    saved=torch.load(ckpt,map_location='cpu',weights_only=False)
    config=rt.load_configuration()[0]; job=config['use_v2']['jobs'][kind]
    model=build(job,'cpu'); model.load_state_dict(saved['model'],strict=True); model.eval()
    before=model.use.verify()
    model.to(device)
    _,dataset=rt.datasets(config); stats=rt.load_stats(rt.ROOT/config['segformer']['data']['stats'])
    fixed=rt.read(rt.ROOT/'docs/assets/research/use/case_provenance.json')['cases']
    lookup={r['sample_id']:i for i,r in enumerate(dataset.records)}
    dest=run/'cases'; dest.mkdir(exist_ok=True); rows=[]
    figure,axes=plt.subplots(len(fixed),4,figsize=(12,3*len(fixed)),layout='constrained')
    for i,case in enumerate(fixed):
        sample=dataset[lookup[case['sample_id']]]
        with torch.no_grad(): probability=model(rt.collate([sample])['image'].to(device))[0,0].sigmoid().cpu().numpy()
        gt=sample['mask'].numpy().astype(bool); valid=sample['valid'].numpy().astype(bool)
        x=sample['image'].x.numpy()*np.array(stats['std'])[:,None,None]+np.array(stats['mean'])[:,None,None]
        rgb=np.clip(x[[2,1,0]].transpose(1,2,0)*.0001/.3,0,1); prediction=probability>=.5
        for ax,data,title in zip(axes[i],[rgb,gt,probability,prediction],['原RGB（固定反射率0–0.3）','滑坡真值（白色）','预测概率（固定0–1）','预测掩膜（阈值0.5）']):
            if data.ndim==3: ax.imshow(data,interpolation='nearest')
            else:
                cmap=plt.get_cmap('magma' if data.dtype!=bool else 'gray').copy(); cmap.set_bad('#607d8b')
                ax.imshow(np.ma.masked_where(~valid,data),cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
            ax.set_title(title if i==0 else ''); ax.set_xticks([]); ax.set_yticks([])
        axes[i,0].set_ylabel(case['sample_id'],fontsize=9)
        figure.colorbar(axes[i,2].images[0],ax=axes[i,2],shrink=.7)
        rows.append(dict(sample_id=case['sample_id'],gt_pixels=int((gt&valid).sum()),predicted_pixels=int((prediction&valid).sum()),
            true_positive=int((prediction&gt&valid).sum()),max_probability=float(probability[valid].max())))
        np.savez_compressed(dest/(case['sample_id']+'.npz'),prob=probability,mask=gt,valid=valid,rgb=rgb)
    if before!=model.use.verify(): raise ValueError('USE changed during case inference')
    figure.suptitle(f"{type(model.visual).__name__}：冻结USE v2＋全网络下游；固定案例，非按预测挑选；{device}只读推理",fontsize=13)
    figure.savefig(dest/'segmentation_cases.png',dpi=130); plt.close(figure)
    rt.write_json(dest/'provenance.json',dict(kind=kind,run=run.relative_to(rt.ROOT).as_posix(),checkpoint_sha256=rt.sha(ckpt),
        use_sha256=before,cases=rows,training_updates=summary['updates']),True)
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','train','accept','freeze','verify','evaluate','cases'])
    p.add_argument('--kind',choices=KINDS);p.add_argument('--device',default='cpu');p.add_argument('--resume',action='store_true')
    p.add_argument('--run');args=p.parse_args()
    if args.action=='prepare': prepare();return
    rt.CONTRACT=CONTRACT;rt.build=build;rt.source_checkpoint=source
    rt.SCRIPTS=(*rt.SCRIPTS,'run_use_v2_full_decoders.py')
    if args.action=='verify': print(verify());return
    if args.action=='freeze': print(json.dumps(freeze(),indent=2));return
    if args.action=='accept': accept(args.device);return
    if args.action in ('smoke','train','evaluate','cases'):
        if args.kind is None: p.error('--kind is required')
    run=Path(args.run).resolve() if args.run else OUT/('engineering' if args.action=='smoke' else 'runs')/args.kind
    if args.action=='evaluate': print(json.dumps(evaluate(args.kind,run,args.device),indent=2));return
    if args.action=='cases': print(json.dumps(cases(args.kind,run,args.device),ensure_ascii=False));return
    if args.action=='train':
        if torch.device(args.device).type!='cuda':
            raise SystemExit('Formal training needs an explicit CUDA device; CPU acceptance stays engineering only')
        digest=verify(); accepted=acceptance_path(args.device)
        if not accepted.exists(): raise SystemExit(f'Acceptance missing on this device: {accepted}')
        proof=rt.read(accepted)
        if proof.get('passed') is not True or proof.get('release_sha256')!=digest or proof.get('config_sha256')!=rt.sha(CONTRACT/'execution.yaml'):
            raise SystemExit('Acceptance does not match the frozen release')
        for k in KINDS:
            s=rt.read(OUT/'engineering'/k/'summary.json')
            if not s['complete'] or s['engineering_updates']!=2: raise ValueError('Both real-data smokes required')
    with rt.gpu_guard(args.device) as gate:
        result=rt.train(args.kind,run,args.device,
            engineering=args.action=='smoke',engineering_steps=2 if args.action=='smoke' else None,
            resume=args.resume,release_hash=rt.sha(CONTRACT/'execution.yaml') if args.action=='smoke' else verify(),gate=gate)
    print(result['complete'],result['best_score'])

if __name__=='__main__':main()
