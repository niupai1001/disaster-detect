"""Ten-epoch full-scene probe on the frozen expanded 2,000 TRAIN subset."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import time

from pd_here_native_expanded_subset import CORE_SHA, EXTENSION_SHA, ExpandedTrainingData, digest, read
import torch

from encoder_v3.latent_response.tvlp_writer import RunWriter
from pd_here_geotoken_dualvit_r3.data import write_json
from pd_here_s60.whole_scene import WholeSceneViT, pack_scene, tile_bounds, tiles
from pd_here_s60.whole_scene_dual import WholeSceneDualViT
from pd_here_s60.whole_scene_dual_axis import WholeSceneDualAxisViT, MODEL_ID as DUAL_AXIS_ID

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT/'outputs/pd_here_native_patch/wp1_wp2/raw_native_full_preflight_v1'
VIEWER = ROOT/'outputs/pd_here_geotoken_dualvit_r3/viewer'
TASKS = ('A','B1')


def masks(scene, parent, epoch):
    support = torch.stack([q.any(-1) for q in scene.qa.split(
        [h*w for _,h,w in scene.layout],-1)],-1)
    legal = support.any(-1)
    gen = random.Random(f'17:{parent}:A:{epoch}')
    gh,gw = scene.grid
    patch_ids = torch.arange(gh*gw,device=legal.device)
    group_ids = patch_ids if scene.patch_m==120 else (patch_ids//gw//2)*((gw+1)//2)+(patch_ids%gw)//2
    groups = group_ids[legal].unique().tolist()
    if len(groups)>=2:
        chosen = gen.sample(groups,max(1,len(groups)//4))
        a = legal & torch.isin(group_ids,torch.tensor(chosen,device=legal.device))
    elif scene.subcells is not None:
        subcells = scene.subcells[scene.qa].unique().tolist()
        if len(subcells)<2:
            raise ValueError(f'{parent}: insufficient distinct original 60m cells for A')
        a = scene.subcells==gen.choice(subcells)
    else:
        # Tiny scenes can fit only one 120 m group. Mask their 60 m positions directly.
        valid_ids = patch_ids[legal].tolist()
        if len(valid_ids)<2:
            raise ValueError(f'{parent}: insufficient distinct 60m locations for A')
        a = legal & torch.isin(patch_ids,torch.tensor(gen.sample(valid_ids,max(1,len(valid_ids)//4)),device=legal.device))
    band_ids = support.any(0).nonzero().flatten().tolist()
    if len(band_ids)<2:
        raise ValueError(f'{parent}: insufficient valid bands for B1')
    offset = int(hashlib.sha256(f'17:{parent}:B1'.encode()).hexdigest()[:8],16)
    b = torch.ones(len(scene.layout),dtype=torch.bool,device=legal.device)
    b[band_ids[(offset+epoch)%len(band_ids)]] = False
    return (a,torch.ones_like(b)),(legal,b)


def objective(model, scene, view, task, diagnostics=False, backward_scale=None):
    keep,bands = view
    widths = [h*w for _,h,w in scene.layout]
    if task=='A':
        hidden = ~keep[:,None] if keep.ndim==1 else ~keep
    else:
        hidden = torch.repeat_interleave(~bands,torch.tensor(widths,device=bands.device))[None]
    selected = (scene.qa & hidden).split(widths,-1)
    counts = [q.sum() for q in selected]
    active = torch.stack(counts).gt(0).sum()
    if not active:
        raise ValueError(f'{task}: no valid targets')
    if not diagnostics:
        total = scene.values.new_zeros(())
        for tile,visible,index in tiles(scene,keep):
            structural = tile.qa.any(-1)
            supports = [q[index][structural] for q in selected]
            if not any(q.any() for q in supports):
                continue
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction = model(tile,visible,bands)
            errors = [torch.where(q,p.float()-y,0.).square().sum()/n.clamp_min(1)
                      for p,y,q,n in zip(prediction.split(widths,-1),tile.values[structural].split(widths,-1),
                                       supports,counts,strict=True)]
            loss = torch.stack(errors).sum()/active
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{task}: nonfinite tile loss')
            if backward_scale is not None:
                (loss*backward_scale).backward()
            total += loss.detach()
        return total

    prediction = torch.zeros_like(scene.values)
    for tile,visible,index in tiles(scene,keep):
        structural = tile.qa.any(-1)
        if structural.any():
            native = torch.zeros_like(tile.values)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                native[structural] = model(tile,visible,bands).float()
            prediction[index] = native
    if diagnostics:
        metrics = defaultdict(list)
    errors = []
    slabs = zip(prediction.split(widths,-1),scene.values.split(widths,-1),selected,strict=True)
    for p,y,target in slabs:
        if not target.any():
            continue
        error = torch.where(target,p.float()-y,0.)
        mse = error.square().sum()/target.sum()
        errors.append(mse)
        if diagnostics:
            metrics['mse'].append(float(mse))
            metrics['centered_mse'].append(float(mse-(error.sum()/target.sum()).square()))
            counts = target.sum(-1)
            mean = error.sum(-1)/counts.clamp_min(1)
            centered = torch.where(target,error-mean[:,None],0.)
            metrics['within_patch_mse'].append(float(centered.square().sum()/target.sum()))
            complete = target.all(-1)
            if complete.sum()>=2:
                texture = p[complete].float()
                texture = texture-texture.mean(-1,keepdim=True)
                energy = texture.square().mean()
                if energy>1e-20:
                    metrics['template_fraction'].append(float(texture.mean(0).square().mean()/energy))
    if not errors:
        raise ValueError(f'{task}: no valid targets')
    loss = torch.stack(errors).mean()
    if diagnostics:
        return {k:sum(v)/len(v) for k,v in metrics.items()}
    return loss


def save(path,model,optimizer,epoch,identity,history,device):
    temporary = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                    epoch=epoch,identity=identity,history=history,cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(device)),temporary)
    temporary.replace(path)


@torch.no_grad()
def evaluate(model, panel, cache, device):
    model.eval()
    records = []
    for row in panel:
        scene=cache[row['id']].to(device)
        views=masks(scene,row['id'],0)
        for task,view in zip(TASKS,views,strict=True):
            records.append(dict(id=row['id'],source=row['source'],domain=row['domain'],task=task,
                                metrics=objective(model,scene,view,task,True)))
    model.train()
    result={}
    for task in TASKS:
        sources={}
        for source in sorted({r['source'] for r in panel}):
            group=[r['metrics'] for r in records if r['task']==task and r['source']==source]
            sources[source]={k:sum(v)/len(v) for k in set().union(*(r.keys() for r in group))
                             for v in [[r[k] for r in group if k in r]]}
        macro={k:sum(v)/len(v) for k in ('mse','centered_mse','within_patch_mse','template_fraction')
               for v in [[r[k] for r in sources.values() if k in r]] if v}
        result[task]=dict(source_macro=macro,by_source=sources)
    return dict(aggregate=result,parents=records)


def run(args):
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(17)
    subset_file = FROZEN/'subset_expanded_2000_seed17.json'
    prior = read(FROZEN/'expanded2000_plan.json')
    subset = read(subset_file)
    rows = subset['parents']
    if (digest(subset_file)!=prior['subset_sha256'] or subset['core_release_sha256']!=CORE_SHA or
            subset['extension_release_sha256']!=EXTENSION_SHA or len({r['id'] for r in rows})!=2000):
        raise ValueError('Frozen expanded subset identity differs')
    data = ExpandedTrainingData(ROOT)
    patch_m = 120 if args.legacy_dual else 60
    model_type = WholeSceneDualViT if args.legacy_dual else (WholeSceneDualAxisViT if args.dual_axis else WholeSceneViT)
    model_id = 'R3-WholeScene-DualViT' if args.legacy_dual else (DUAL_AXIS_ID if args.dual_axis else 'S60-WholeScene-SpatialViT')
    cache={}; layouts=defaultdict(list); members=[]; panel=[]; selected=defaultdict(int)
    print('Packing all native scenes',flush=True)
    for i,row in enumerate(rows,1):
        key=row['id'];scale=subset['dn_scales'][row['domain']] if row['encoding']=='stored_dn' else 1.
        try:
            scene=pack_scene(data.load(row,'cpu'),scale,patch_m)
            masks(scene,key,0)
        except Exception as error:
            raise ValueError(f"Full scene {key} ({row['source']}): {error}") from error
        cache[key]=scene
        slots=[h*w for _,h,w in scene.layout]
        means=[float(torch.where(q,x,0.).sum()/q.sum().clamp_min(1)) for x,q in zip(
            scene.values.split(slots,-1),scene.qa.split(slots,-1),strict=True)]
        layouts[scene.layout].append(means)
        bounds=tile_bounds(scene)
        max_output=[max((r1-r0)*h for r0,r1,_,_ in bounds for _,h,_ in scene.layout),
                    max((c1-c0)*w for _,_,c0,c1 in bounds for _,_,w in scene.layout)]
        assert max(max_output)<=512
        members.append(dict(**row,grid=scene.grid,layout=scene.layout,valid_native_pixels=scene.valid_pixels,
                            origin=scene.origin,scale=scale,tiles=bounds,max_native_output_hw=max_output))
        if selected[row['source']]<2:
            panel.append(row);selected[row['source']]+=1
        if i%200==0:print('packed',i,flush=True)
    stats={k:[sum(x)/len(x) for x in zip(*v,strict=True)] for k,v in layouts.items()}
    write_json(args.out/'members.json',members)
    identity=dict(model_id=model_id,subset_sha256=digest(subset_file),
        core_sha256=CORE_SHA,extension_sha256=EXTENSION_SHA,members_sha256=digest(args.out/'members.json'),
        layouts=[dict(bands=k,train_means=v) for k,v in stats.items()],sources=subset['sample_counts'],
        precision='BF16 linear/attention, FP32 residual/loss/optimizer',attention='exact efficient SDPA over each whole image or output tile',
        native_output_max_hw=[512,512],split_parents=sum(len(x['tiles'])>1 for x in members),
        tile_protocol='non-overlapping physical patches; native output bound; counts and masks on parent before partition',
        epochs=10,parents=2000,batch=16,updates=1250,patch_m=patch_m,seed=17,lr=.0001,
        mask='A: select quarter of valid 120m groups; tiny-scene fallback select 60m cells; B1: cyclic eligible band',
        objective='paired A/B1 native target MSE, per band then per scene',eval='fixed TRAIN panel, at most 2 per source',
        panel=[r['id'] for r in panel],train_only=True)
    write_json(args.out/'identity.json',identity)
    write_json(args.out/'status.json',dict(status='PREFLIGHT'))
    # Exercise the largest full sequence and every native layout before any optimizer update.
    torch.manual_seed(17)
    model=model_type(stats).to(args.device)
    if args.dual_axis:
        identity.update(initialization='seed17, all common tensors identical to spatial ViT initialization',
                        encoder='one global spatial block, one per-patch spectral block, one per-patch fusion block')
        write_json(args.out/'identity.json',identity)
    if args.legacy_dual:
        identity['factorized_spatial_layouts'] = [list(k) for k,idx in model.layout_indices.items()
                                                 if model.projections[idx].factorized]
        write_json(args.out/'identity.json',identity)
    counts=defaultdict(int)
    for scene in cache.values():counts[scene.layout]+=1
    biggest=sorted(rows,key=lambda r:cache[r['id']].grid[0]*cache[r['id']].grid[1],reverse=True)
    measured=[];seen=set()
    for row in biggest:
        scene=cache[row['id']]
        if scene.layout in seen:continue
        seen.add(scene.layout)
        scene=scene.to(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        start=time.perf_counter()
        for task,view in zip(TASKS,masks(scene,row['id'],0),strict=True):
            loss=objective(model,scene,view,task,backward_scale=.5)
            if not torch.isfinite(loss):raise FloatingPointError(f"{row['id']}/{task}")
        torch.cuda.synchronize(args.device)
        report=dict(parent=row['id'],source=row['source'],layout=scene.layout,
            grid=scene.grid,tiles=len(tile_bounds(scene)),valid_tokens=int(scene.qa.any(-1).sum()),
            seconds=time.perf_counter()-start,peak_allocated=torch.cuda.max_memory_allocated(args.device))
        measured.append(report)
        model.zero_grad(set_to_none=True)
        print('full-scene check',report['source'],report['grid'],round(report['seconds'],3),flush=True)
    write_json(args.out/'preflight.json',dict(measured=measured,parameters=sum(p.numel() for p in model.parameters())))
    groups=[dict(params=[p for p in model.parameters() if p.ndim>1],weight_decay=.04),
            dict(params=[p for p in model.parameters() if p.ndim==1],weight_decay=0.)]
    optimizer=torch.optim.AdamW(groups,lr=.0001)
    viewer=VIEWER/args.out.name
    definitions={f'{t}_{k}':dict(label=f'{t} TRAIN {k}',direction='min',role='loss',unit='')
                 for t in TASKS for k in ('mse','centered_mse','within_patch_mse','template_fraction')}
    history=[]
    write_json(args.out/'initial.json',evaluate(model,panel,cache,args.device))
    with RunWriter(viewer,experiment=dict(id='pd_here_r3_whole_scene' if args.legacy_dual else 'pd_here_s60_whole_scene',
                   name='Full-scene 2000 TRAIN'),
                   variant=dict(id=model_id,name=model_id,kind='baseline'),
                   config=identity,metrics=definitions,charts=['A_mse','B1_mse','A_within_patch_mse','B1_within_patch_mse']) as log:
        for epoch in range(1,11):
            started=time.perf_counter()
            ordered=rows.copy();random.Random(17+epoch).shuffle(ordered)
            totals={task:0. for task in TASKS}
            updates=125*(epoch-1)
            write_json(args.out/'status.json',dict(status='RUNNING',epochs=epoch-1,current_epoch=epoch,updates=updates))
            for i in range(0,len(ordered),16):
                group=ordered[i:i+16]
                optimizer.zero_grad(set_to_none=True)
                for row in group:
                    parent=row['id'];scene=cache[parent].to(args.device)
                    for task,view in zip(TASKS,masks(scene,parent,epoch-1),strict=True):
                        loss=objective(model,scene,view,task,backward_scale=.5/len(group))
                        if not torch.isfinite(loss):raise FloatingPointError(f'{parent}/{task} nonfinite loss')
                        totals[task]+=float(loss.detach())
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                updates+=1
                if updates%25==0:
                    write_json(args.out/'status.json',dict(status='RUNNING',epochs=epoch-1,current_epoch=epoch,updates=updates))
                    print('updates',updates,'epoch',epoch,'elapsed_s',round(time.perf_counter()-started,1),flush=True)
            evaluated=evaluate(model,panel,cache,args.device)
            item=dict(epoch=epoch,updates=updates,train={k:v/len(rows) for k,v in totals.items()},
                      fixed_train=evaluated,seconds=time.perf_counter()-started)
            history.append(item)
            save(args.out/'last.pt',model,optimizer,epoch,identity,history,args.device)
            write_json(args.out/'history.json',history)
            log.log(updates,{f'{task}_{k}':value for task in TASKS for k,value in
                             evaluated['aggregate'][task]['source_macro'].items()},epoch=epoch)
            print('epoch',epoch,item['train'],round(item['seconds'],1),flush=True)
    write_json(args.out/'status.json',dict(status='COMPLETED',epochs=10,updates=1250,viewer=str(viewer)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--device',default='cuda:1')
    architectures=parser.add_mutually_exclusive_group()
    architectures.add_argument('--legacy-dual',action='store_true')
    architectures.add_argument('--dual-axis',action='store_true')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:
        run(args)
    except Exception as error:
        write_json(args.out/'status.json',dict(status='FAILED',error=str(error)))
        raise


if __name__=='__main__':
    main()
