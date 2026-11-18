"""Train S60 and the previous DualViT under the same native-target protocol."""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import random
import time

import torch
from torch import nn

from train_pd_here_r3_texture_study import (
    BASE, Study, TASKS, RunWriter, batches, digest, initial_model, update, write_json,
)
from pd_here_geotoken_dualvit_r3.evaluation import metric_records
from pd_here_s60.model import DualAxisViT, Layout, inputs60, retile

OLD = 'fusion256_global2_zerohead'


class NativeTargetAdapter(nn.Module):
    """Only the comparison interface uses old packing. All network tokens are 60 m."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, values, qa, keep, bands):
        x,q,k = inputs60(values,qa,keep)
        output = self.model(x,q,k,bands)
        slabs = output['prediction'].split([h*w for h,w in self.model.layout.native_shapes],-1)
        return {**output,'prediction':torch.cat([
            retile(t.reshape(len(keep),64,h,w),8,4).flatten(2)
            for t,(h,w) in zip(slabs,self.model.layout.native_shapes,strict=True)],-1)}


def compare_records(prediction, values, qa, keep, bands, task):
    rows = metric_records(prediction,values,qa,keep,bands,task)
    keyed = {(r['parent'],r['band']):r for r in rows}
    truth,quality,k = inputs60(values,qa,keep)
    offset = 0
    for j,(y,q,old) in enumerate(zip(truth,quality,values,strict=True)):
        size = old.shape[-2]*old.shape[-1]
        p = retile(prediction[...,offset:offset+size].reshape_as(old),4,8)
        offset += size
        valid = q & ((~k)[:,:,None,None] if task=='A' else (~bands[:,j])[:,None,None,None])
        for i in range(len(y)):
            r = keyed.get((i,j))
            if r is None:
                continue
            z = valid[i].flatten(1)
            px,ty = p[i].flatten(1).double(),y[i].flatten(1).double()
            count = z.sum(-1)
            supported = count>0
            def split(x):
                mean = torch.where(z,x,0.).sum(-1)/count.clamp_min(1)
                return mean,torch.where(z,x-mean[:,None],0.)
            mx,cx = split(px)
            my,cy = split(ty)
            r['patch_mean_mse'] = float(((mx-my).square()*count).sum()/count.sum())
            r['within_patch_centered_mse'] = float((cx-cy).square().sum()/count.sum())
            for label,v in [('prediction',cx),('truth',cy)]:
                r[label+'_texture_rms'] = float((v.square().sum()/count.sum()).sqrt())
                full = z.all(-1)
                if full.sum()>=2:
                    texture = v[full]
                    energy = texture.square().mean()
                    if energy>1e-20:
                        r[label+'_common_template_fraction'] = float(texture.mean(0).square().mean()/energy)
            h,w = y.shape[-2:]
            error = (p[i]-y[i]).reshape(8,8,h,w).transpose(1,2).reshape(8*h,8*w).double()
            mask = valid[i].reshape(8,8,h,w).transpose(1,2).reshape_as(error)
            edge = []
            inside = []
            for axis,side in ((0,h),(1,w)):
                delta = error.diff(dim=axis)
                pair = (mask[1:] & mask[:-1]) if axis==0 else (mask[:,1:] & mask[:,:-1])
                seam = torch.arange(1,error.shape[axis],device=error.device)%side==0
                seam = seam[:,None] if axis==0 else seam[None,:]
                edge.append(delta[pair & seam])
                inside.append(delta[pair & ~seam])
            for name,parts in [('boundary',edge),('interior',inside)]:
                d = torch.cat(parts)
                if len(d):
                    r['gradient_mse_'+name] = float(d.square().mean())
    return rows


def evidence(study):
    layout = Layout()
    if study.layout.band_keys != layout.band_keys or study.layout.native_shapes != tuple((2*h,2*w) for h,w in layout.native_shapes):
        raise ValueError('Corrected release differs from canonical S60 band layout')
    geometry = {}
    support_hash = hashlib.sha256()
    for role,rows in study.rows.items():
        for row in rows:
            source = Path(row['cache'])
            if not row.get('cache_sha256') or digest(source)!=row['cache_sha256']:
                raise ValueError(f"Corrected cache hash differs for {row['id']}")
            entry = study.panel[row['id']]
            x,y = entry['origin']
            for band,gsd in zip(row['bands'],layout.band_gsd,strict=True):
                a,r,bx,s,e,by = band['transform']
                indices = ((x-bx)/a,(by-y)/-e)
                if (a,-e)!=(gsd,gsd) or min(indices)<-1e-7 or any(abs(t-round(t))>1e-7 for t in indices):
                    raise ValueError(f"Invalid window alignment for {row['id']}")
            geometry[row['id']] = dict(role=role,origin=[x,y],window_id=entry['window_id'],
                crs=row['bands'][0]['crs'],affines=[band['transform'] for band in row['bands']])
        for _,group in batches(rows):
            _,qa,views = study.inputs(group,0)
            for (keep,bands) in views:
                for j,q in enumerate(qa):
                    old = q & (~keep[:,:,None,None] | ~bands[:,j,None,None,None])
                    child_q = retile(q,4,8)
                    child_k = retile(keep[...,None,None].expand(-1,-1,2,2),4,8).flatten(1)
                    child = child_q & (~child_k[:,:,None,None] | ~bands[:,j,None,None,None])
                    if not torch.equal(retile(child,8,4),old):
                        raise AssertionError('Native target support changed')
                    support_hash.update(old.cpu().numpy().tobytes())
    return geometry,support_hash.hexdigest()


def models(study,stats):
    torch.manual_seed(17)
    new = NativeTargetAdapter(DualAxisViT(stats[0])).to(study.device)
    groups = [dict(params=[p for p in new.parameters() if p.ndim>1],weight_decay=.04),
              dict(params=[p for p in new.parameters() if p.ndim==1],weight_decay=0.)]
    # The two branch-type vectors are a 2-D learned embedding, so they receive decay.
    return {'s60':(new,torch.optim.AdamW(groups,lr=1e-4)),
            'r3':initial_model(study.layout,OLD,stats,study.device)}


def spatial_vit(study, stats):
    torch.manual_seed(17)
    reference = DualAxisViT(stats[0]).state_dict()
    model = NativeTargetAdapter(DualAxisViT(stats[0],spatial_only=True))
    model.model.load_state_dict({key:reference[key] for key in model.model.state_dict()})
    model.to(study.device)
    groups = [dict(params=[p for p in model.parameters() if p.ndim>1],weight_decay=.04),
              dict(params=[p for p in model.parameters() if p.ndim==1],weight_decay=0.)]
    return model,torch.optim.AdamW(groups,lr=1e-4)


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def benchmark(study,stats,panel):
    import statistics
    import numpy as np
    report = {}
    x,q,views = study.inputs(panel,0)
    for task in ('A','B1','paired'):
        for name,(model,opt) in models(study,stats).items():
            durations = []
            for i in range(70):
                sync(study.device)
                if i==20:
                    torch.cuda.reset_peak_memory_stats(study.device)
                start = time.perf_counter()
                update(model,opt,x,q,views,task)
                sync(study.device)
                if i>=20:
                    durations.append(time.perf_counter()-start)
            report.setdefault(name,{})[task] = dict(p50_seconds=statistics.median(durations),
                p95_seconds=float(np.quantile(durations,.95)),warmup=20,measured=50,
                peak_allocated=torch.cuda.max_memory_allocated(study.device),
                peak_reserved=torch.cuda.max_memory_reserved(study.device),parents=len(panel),
                valid_window_pixels=sum(int(a.sum()) for a in q),
                support={t:dict(
                    input_pixels=sum(int((a & keep[:,:,None,None] & bands[:,j,None,None,None]).sum()) for j,a in enumerate(q)),
                    target_pixels=sum(int((a & (~keep[:,:,None,None] | ~bands[:,j,None,None,None])).sum()) for j,a in enumerate(q)))
                    for t,(keep,bands) in zip(TASKS,views,strict=True) if task in (t,'paired')})
            print('benchmark',name,task,report[name][task]['p50_seconds'],flush=True)
    return report


def checkpoint(path, model, optimizer, epoch, identity, history, device):
    tmp = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                    history=history,identity=identity,cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(device)),tmp)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--spatial-only',action='store_true',help='Run a single fresh 30-epoch spatial ViT arm')
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    status = args.out/'status.json'
    try:
        run(args,status)
    except Exception as error:
        write_json(status,dict(status='FAILED',error=str(error)))
        raise


def run(args,status):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    study = Study(BASE/'texture_study',args.device)
    geometry,target_hash = evidence(study)
    write_json(args.out/'windows.json',geometry)
    stats,thresholds,coefficients = study.train_references()
    pairs = {'vit':spatial_vit(study,stats)} if args.spatial_only else models(study,stats)
    primary = next(iter(pairs.values()))[0]
    layout = primary.model.layout
    identity = dict(model_id='PD-HERE-S60-SpatialViT' if args.spatial_only else 'PD-HERE-S60-DualAxis-v1',
        reference='PD-HERE-S60-DualAxis-v1' if args.spatial_only else OLD,
        layout=asdict(layout),layout_hash=layout.layout_hash,data=study.identity,
        windows_sha256=digest(args.out/'windows.json'),exposure0_target_sha256=target_hash,
        mask_source_sha256=digest(Path(__file__).resolve().parents[1]/'src/pd_here_geotoken_dualvit_r3/data.py'),
        seed=17,batch=16,lr=.0001,weight_decay=.04,max_epochs=30,precision='FP32',
        diagnostic_updates_per_task=0 if args.spatial_only else 200,
        renderer_scope='fixed native targets; 60m analysis grid for both')
    if args.spatial_only:
        identity.update(encoder='Linear native patch embedding + position + one global pre-LN ViT block + LN',
                        initialization='seed17 Dual-Axis initial tensors for common keys; no trained checkpoint')
    write_json(args.out/'identity.json',identity)
    panel = study.rows['train'][:16]
    write_json(status,dict(status='DIAGNOSTICS'))
    report = {}
    for task in (() if args.spatial_only else TASKS):
        for name,(model,opt) in models(study,stats).items():
            score = lambda: study.score(model,thresholds,coefficients,rows=panel,tasks=(task,),records_for=compare_records)
            before = score()
            x,q,views = study.inputs(panel,0)
            for _ in range(200):
                update(model,opt,x,q,views,task)
            after = score()
            report.setdefault(name,{})[task] = dict(before=before,after=after,parents=[r['id'] for r in panel])
            checkpoint(args.out/f'diagnostic_{name}_{task}.pt',model,opt,0,identity,[],args.device)
            write_json(args.out/'diagnostics.json',report)
            print('diagnostic',name,task,after['model'][task]['all']['metrics']['mse'],flush=True)
    if not args.spatial_only:
        write_json(status,dict(status='BENCHMARK'))
        write_json(args.out/'benchmark.json',benchmark(study,stats,panel))
    references = study.score(primary,thresholds,coefficients,records_for=compare_records)
    references.pop('model')
    write_json(args.out/'references.json',references)
    initial = {name:study.score(model,thresholds,records_for=compare_records)['model'] for name,(model,_) in pairs.items()}
    write_json(args.out/'initial.json',initial)
    history = []
    reason = 'budget_complete'
    definitions = {f'{n}_{t}_{k}':dict(label=f'{n} {t} {k}',direction='min',role='loss',unit='')
                   for n in pairs for t in TASKS for k in ('mse','gradient_mse','within_patch_centered_mse')}
    viewer = BASE/'viewer'/args.out.name
    with RunWriter(viewer,experiment=dict(id='pd_here_s60_spatial_vit' if args.spatial_only else 'pd_here_s60_dualaxis',
                   name='S60 Spatial ViT' if args.spatial_only else 'S60 vs previous DualViT'),
                   variant=dict(id='vit' if args.spatial_only else 'matched',
                                name='Spatial ViT' if args.spatial_only else 'Matched S60 and R3',kind='baseline'),metrics=definitions,
                   charts=[f'{n}_{t}_gradient_mse' for n in pairs for t in TASKS],config=identity) as log:
        for epoch in range(1,31):
            write_json(status,dict(status='RUNNING',completed_epochs=epoch-1,target_max_epochs=30))
            ordered = list(study.rows['train'])
            random.Random(17+epoch).shuffle(ordered)
            entry = dict(epoch=epoch,updates=44*epoch,arms={})
            for name,(model,opt) in pairs.items():
                start = time.perf_counter()
                train = dict(A=0.,B1=0.)
                for _,rows in batches(ordered):
                    losses,_ = update(model,opt,*study.inputs(rows,epoch-1),'paired')
                    for task,value in losses.items():
                        train[task] += value*len(rows)/700
                val = study.score(model,thresholds,records_for=compare_records)['model']
                entry['arms'][name] = dict(train=train,val=val,seconds=time.perf_counter()-start)
            history.append(entry)
            for name,(model,opt) in pairs.items():
                checkpoint(args.out/f'{name}_last.pt',model,opt,epoch,identity,history,args.device)
            write_json(args.out/'history.json',history)
            log.log(44*epoch,{key:entry['arms'][n]['val'][t]['all']['metrics'][k] for key in definitions
                              for n,t,k in [key.split('_',2)]},epoch=epoch)
            print('epoch',epoch,{n:{t:r['val'][t]['all']['metrics']['mse'] for t in TASKS}
                                  for n,r in entry['arms'].items()},flush=True)
            if epoch==10 and not args.spatial_only:
                now = entry['arms']['s60']['val']
                # Require both structural errors below matched R3, or both MSEs below same-input references.
                old = entry['arms']['r3']['val']
                structural = all(sum(now[t]['all']['metrics'][k] for t in TASKS)<
                                 sum(old[t]['all']['metrics'][k] for t in TASKS)
                                 for k in ('within_patch_centered_mse','gradient_mse'))
                baseline = all(now[t]['all']['metrics']['mse']<references[b][t]['all']['metrics']['mse']
                               for t,b in [('A','idw'),('B1','ols')])
                write_json(args.out/'epoch10_gate.json',dict(structural_improved=structural,baseline_pass=baseline,
                    definition='S60 vs same-epoch R3: both task-equal within-60m centered and native gradient MSE decrease; baseline pass requires A IDW and B1 OLS MSE'))
                if not structural and not baseline:
                    reason = 'epoch10_no_structural_improvement_and_baseline_not_passed'
                    break
    write_json(status,dict(status='COMPLETED',reason=reason,epochs=epoch,updates_per_arm=44*epoch,viewer=str(viewer)))


if __name__=='__main__':
    main()
