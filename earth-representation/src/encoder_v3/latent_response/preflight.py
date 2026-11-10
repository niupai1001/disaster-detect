"""Independent E1 check/no-update entry; complete caches, explicit identities and limits."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback
import torch
from encoder_v3.data import sha256
from .data import LRDataset,load_catalogs,make_view,pool_recipe
from .geometry import FullGeometry
from .model import LatentResponseModel
from .execution import view_loss
from .checkpoint import parameter_sha,digest,atomic_replace

ROOT=Path(__file__).resolve().parents[3]


def write(path,document):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(document,ensure_ascii=False,indent=2),encoding='utf-8')
    atomic_replace(temporary,path)


def configure_fp32():
    # Match the controlled equivalence/resource checks, rather than implicit TF32 defaults.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def build_inputs():
    configure_fp32()
    config=json.loads((ROOT/'configs/config_lr_e0_v1.json').read_text())
    legacy=json.loads((ROOT/'src/encoder_v3/config_optical_selfsup_v1.json').read_text())['data']
    catalog_path=ROOT/'outputs/encoder_v3/latent_response/channel_catalog_frozen.json'
    catalogs=load_catalogs(catalog_path)
    datasets={role:LRDataset(legacy['release'],legacy['contracts'],role=role,catalogs=catalogs) for role in ('TRAIN','VAL')}
    if config.get('enmap_release'):
        from .enmap_data import KEY,NativeEnmapDataset,CombinedDataset
        from .registry import ChannelCatalog
        catalogs[KEY]=ChannelCatalog.from_ordered([f'B{i:03d}' for i in range(1,225)])
        enmap=NativeEnmapDataset(config['enmap_release'],catalogs)
        datasets['TRAIN']=CombinedDataset(datasets['TRAIN'],enmap)
    recipe=pool_recipe(datasets['TRAIN'].rows)
    model=LatentResponseModel(catalogs,seed=config['seed'])
    code={str(p.relative_to(ROOT)):sha256(p) for folder in ('latent_response',) for p in (ROOT/'src/encoder_v3'/folder).glob('*.py')}
    # The loader and legacy task implementation are also executable dependencies.
    for name in ('data.py','tasks.py','model.py'):
        p=ROOT/'src/encoder_v3'/name;code[str(p.relative_to(ROOT))]=sha256(p)
    identity=dict(code=digest(code),code_files=code,config=digest(config),registry=model.registry.identity_sha256(),
                  catalog=digest({key.uid:value.as_dict() for key,value in catalogs.items()}),catalog_file=sha256(catalog_path),data=datasets['TRAIN'].identities,sampling=digest(recipe),
                  interfaces=sha256(ROOT/'src/encoder_v3/latent_response/interface_confirmation.json'))
    return config,datasets,recipe,model,identity


def area(row):return max(b['shape'][0]*b['shape'][1] for b in row['bands'])


def cases(datasets):
    selected=[]
    for sensor,count in (('S2',4),('S2',10),('L8',7),('L8',7)):
        rows=[r for r in datasets['VAL'].rows if r['sensor']==sensor and len(r['bands'])==count and r['id'] not in {c['id'] for c in selected}]
        if not rows:raise ValueError(f'Missing VAL case {sensor}/{count}')
        row=min(rows,key=lambda r:(area(r),r['id']))
        selected.append(dict(role='VAL',id=row['id'],sensor=sensor,bands=count,area=area(row),scope='complete_A_B1'))
    for sensor,count in (('S2',10),('L8',6),('L9',6)):
        rows=[r for r in datasets['TRAIN'].rows if r['sensor']==sensor and len(r['bands'])==count and (area(r)>16384 if sensor=='S2' else True)]
        row=min(rows,key=lambda r:(area(r),r['id']))
        selected.append(dict(role='TRAIN',id=row['id'],sensor=sensor,bands=count,area=area(row),scope='engineering_complete_A_B1'))
    enmap_rows=[row for row in datasets['TRAIN'].rows if row['sensor']=='EnMAP']
    if enmap_rows:
        chosen={min(enmap_rows,key=area)['id'],max(enmap_rows,key=area)['id']}
        for row in enmap_rows:
            if row['id'] in chosen:selected.append(dict(role='TRAIN',id=row['id'],sensor='EnMAP',bands=len(row['bands']),area=area(row),scope='engineering_complete_A_B1'))
    maximum=max(datasets['TRAIN'].rows,key=area)
    selected.append(dict(role='TRAIN',id=maximum['id'],sensor=maximum['sensor'],bands=len(maximum['bands']),area=area(maximum),scope='maximum_complete_encoder_forward'))
    return selected


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('check','no-update'),default='check')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/encoder_v3/latent_response/plan_execution/e1')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(17);torch.set_num_threads(2)
    config,datasets,recipe,model,identity=build_inputs()
    selected=cases(datasets)
    write(args.output/'sampling_recipe.json',recipe)
    write(args.output/'validation_panel.json',dict(cases=[c for c in selected if c['role']=='VAL'],
          mask_rule='deterministic central Sub-Patch-aligned rectangle; B1 last loaded channel',identity=identity))
    report=dict(mode=args.mode,status='CHECK_PASS',identity=identity,parameters=sum(p.numel() for p in model.parameters()),
                train_members=len(datasets['TRAIN']),val_members=len(datasets['VAL']),pool_members=recipe['member_count'],
                device=args.device,real_optimizer_steps=0,cases=[],scope='E1 engineering only; no scientific validation score',
                max_scene_backward_tested=False)
    path=args.output/'readiness.json';write(path,report)
    if args.mode=='check':return
    device=torch.device(args.device);model.to(device)
    if device.type=='cuda':report['gpu']=torch.cuda.get_device_name(device)
    for case in selected:
        row=dict(case,status='RUNNING');report['cases'].append(row);write(path,report)
        start=time.perf_counter();before=parameter_sha(model)
        try:
            dataset=datasets[case['role']]
            index=next(i for i,r in enumerate(dataset.rows) if r['id']==case['id'])
            observation=dataset[index]
            channels=dataset.channels(observation,device)
            row['cache_sha256']=observation.metadata['cache_sha256']
            row['native_shapes']={b.record_id:list(b.reflectance.shape) for b in observation.bands}
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
            row['tasks']={}
            if case['scope']=='maximum_complete_encoder_forward':
                with torch.no_grad():
                    representation=model.encode(channels,FullGeometry.from_channels(channels),tile_rows=config['frontend_tile_rows'],chunk_rows=config['window_chunk_rows'])
                row['census']=representation.census
                del representation
                row['decoder_targets']=0
                row['note']='Full maximum cache encoder forward; no maximum-scene decoder or backward claim'
            else:
                for kind in ('A','B1'):
                    model.zero_grad(set_to_none=True)
                    view=make_view(observation,channels,kind)
                    row['tasks'][kind]=view_loss(model,view,backward=True,tile_rows=config['frontend_tile_rows'],
                        chunk_rows=config['window_chunk_rows'],chunk_pixels=config['decoder_chunk_pixels'])
                    row['tasks'][kind]['finite_gradient_parameters']=sum(p.grad is not None for p in model.parameters())
                    del view
                row['equal_weight_A_B1_loss']=.5*(row['tasks']['A']['loss']+row['tasks']['B1']['loss'])
            if device.type=='cuda':torch.cuda.synchronize(device)
            after=parameter_sha(model)
            row.update(status='PASS' if before==after else 'FAIL',parameter_sha_before=before,parameter_sha_after=after,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None)
        except Exception as exc:
            row.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc(),parameter_sha_before=before,
                       parameter_sha_after=parameter_sha(model))
        finally:
            row['seconds']=time.perf_counter()-start
            model.zero_grad(set_to_none=True)
            observation=None; channels=None
            gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
            report['status']='RUNNING';write(path,report)
            print(case['id'],case['scope'],row['status'],round(row['seconds'],2),flush=True)
    report['status']='PASS' if all(c['status']=='PASS' for c in report['cases']) else 'FAIL'
    report['stop_boundary']='E1 complete; E2/E3 remain unauthorized'
    write(path,report)


if __name__=='__main__':main()
