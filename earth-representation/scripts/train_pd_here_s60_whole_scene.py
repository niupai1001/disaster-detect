"""Full-scene training on the frozen expanded 2,000 TRAIN subset."""
import argparse
from collections import Counter, defaultdict
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import random
import time

from pd_here_native_expanded_subset import CORE_SHA, EXTENSION_SHA, ExpandedTrainingData, digest, read
import torch

from encoder_v3.latent_response.tvlp_writer import RunWriter
from pd_here_geotoken_dualvit_r3.data import write_json
from pd_here_s60.whole_scene import WholeSceneViT, pack_scene, tile_bounds, tiles
from pd_here_s60.whole_scene_runtime import prepared_backward
from pd_here_s60.whole_scene_batch import batched_backward, scene_signature
from pd_here_s60.whole_scene_parallel import tune_packed_parallel
from pd_here_s60.mae_schedule import BASE_LR, BETAS, WEIGHT_DECAY, WARMUP_EPOCHS, after_warmup, mae_lr
from pd_here_s60.whole_scene_dual import WholeSceneDualViT
from pd_here_s60.whole_scene_dual_axis import WholeSceneDualAxisViT, MODEL_ID as DUAL_AXIS_ID
from pd_here_s60.whole_scene_spectral_conv import (
    MODEL_ID as SPECTRAL_CONV_ID, MODEL_NAME as SPECTRAL_CONV_NAME, REFERENCE_URLS,
    WholeSceneSpectralConv, WIDE_MODEL_ID, WIDE1024_ID, pack_spectral_conv_scene,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT/'outputs/pd_here_native_patch/wp1_wp2/raw_native_full_preflight_v1'
VIEWER = ROOT/'outputs/pd_here_geotoken_dualvit_r3/viewer'
TASKS = ('A','B1')
LEGACY_MODELS = {
    'R3-WholeScene-DualViT': WholeSceneDualViT,
    'R3-WholeScene-DualViT-Global1': partial(WholeSceneDualViT,global_depth=1),
    'R3-WholeScene-SpectralOnly': partial(WholeSceneDualViT,spectral_only=True),
}
SPATIAL_ID = 'S60-WholeScene-SpatialViT'
MODELS = {SPATIAL_ID:WholeSceneViT, DUAL_AXIS_ID:WholeSceneDualAxisViT,
          SPECTRAL_CONV_ID:WholeSceneSpectralConv,
          WIDE_MODEL_ID:partial(WholeSceneSpectralConv,fusion_width=512),
          WIDE1024_ID:partial(WholeSceneSpectralConv,fusion_width=1024), **LEGACY_MODELS}


class IneligiblePairedScene(ValueError):
    pass


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
            raise IneligiblePairedScene(f'{parent}: insufficient distinct original 60m cells for A')
        a = scene.subcells==gen.choice(subcells)
    else:
        # Tiny scenes can fit only one 120 m group. Mask their 60 m positions directly.
        valid_ids = patch_ids[legal].tolist()
        if len(valid_ids)<2:
            raise IneligiblePairedScene(f'{parent}: insufficient distinct 60m locations for A')
        a = legal & torch.isin(patch_ids,torch.tensor(gen.sample(valid_ids,max(1,len(valid_ids)//4)),device=legal.device))
    band_ids = support.any(0).nonzero().flatten().tolist()
    if len(band_ids)<2:
        raise IneligiblePairedScene(f'{parent}: insufficient valid bands for B1')
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


def transfer_full_heads(model, checkpoint):
    initial = model.state_dict()
    prior = checkpoint['identity']['layouts']
    mapping = {str(i):model.layout_indices[tuple(map(tuple,entry['bands']))]
               for i,entry in enumerate(prior)}
    weights = {}
    for key,value in checkpoint['model'].items():
        if key.startswith('projections.'):
            _,index,name = key.split('.',2)
            key = f'projections.{mapping[index]}.{name}'
        if key not in initial or value.shape!=initial[key].shape:
            raise ValueError(f'Incompatible warm-start weight: {key}')
        weights[key] = value
    missing = set(initial)-weights.keys()
    new_heads = set(model.layout_indices.values())-set(mapping.values())
    permitted = {key for key in initial if key.startswith('projections.') and key.split('.')[1] in new_heads}
    if missing!=permitted:
        raise ValueError('Unexpected missing parameters in full warm start')
    model.load_state_dict(initial | weights)
    return len(mapping)


def early_stop_state(previous, score, epoch, policy):
    if not math.isfinite(score):
        raise FloatingPointError('Nonfinite VAL MSE')
    state = dict(previous)
    best = not state or score < state['best_score']
    if best:
        state.update(best_score=score,best_epoch=epoch)
    if policy is None:
        return state,best
    if 'reference' not in state or (
            score < state['reference'] and score <= state['reference'] * (1-policy['relative_min_delta'])):
        state.update(reference=score,reference_epoch=epoch)
    state['stale_epochs'] = epoch-state['reference_epoch']
    state['triggered'] = epoch>=policy['min_epochs'] and state['stale_epochs']>=policy['patience']
    return state,best


def pack_validation(data, training_ids, layouts, patch_m):
    rows, cache, manifest = [], {}, []
    for member in data.dataset.core_roles['VAL']:
        key = member['id']
        if key in training_ids or key in cache:
            raise ValueError(f'VAL member overlap: {key}')
        source = f'Core {member["sensor"]}'
        obs = data.dataset.core.load(member,'cpu')
        scene = pack_spectral_conv_scene(obs,1.,source) if patch_m==120 else pack_scene(obs,1.,patch_m)
        if scene.layout not in layouts:
            raise ValueError(f'VAL {key}: no TRAIN output head for {scene.layout}')
        masks(scene,key,0)
        cache[key] = scene
        row = dict(id=key,source=source,domain='surface_reflectance')
        rows.append(row)
        manifest.append(dict(**row,cache_sha256=member['cache_sha256'],layout=scene.layout,
                             grid=scene.grid,origin=scene.origin,tiles=tile_bounds(scene),
                             valid_native_pixels=scene.valid_pixels,
                             wavelengths_um=(scene.metadata.wavelengths.tolist() if scene.metadata is not None else
                                             [band.physics.center.summary[0] for band in obs.bands])))
    if len(rows)!=275:
        raise ValueError(f'Expected the frozen 275 Core VAL members, found {len(rows)}')
    return rows,cache,manifest


def save_best(path,model,identity,item):
    temporary = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(),epoch=item['epoch'],identity=identity,
                    updates=item['updates'],validation=item['validation'],
                    validation_paired_mse=item['validation_paired_mse']),temporary)
    temporary.replace(path)


@torch.no_grad()
def evaluate(model, panel, cache, device, diagnostics=True):
    model.eval()
    records = []
    for row in panel:
        scene=cache[row['id']].to(device)
        views=masks(scene,row['id'],0)
        for task,view in zip(TASKS,views,strict=True):
            metrics = objective(model,scene,view,task,diagnostics)
            records.append(dict(id=row['id'],source=row['source'],domain=row['domain'],task=task,
                                metrics=metrics if diagnostics else {'mse':float(metrics)}))
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
    run_started = time.perf_counter()
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
    if args.full_train:
        rows = data.members()
        if len(rows)!=13021 or len({row['id'] for row in rows})!=len(rows):
            raise ValueError('Full paired TRAIN membership differs from the sealed release')
    model_id = args.model_id
    legacy = model_id in LEGACY_MODELS
    fusion_widths = {SPECTRAL_CONV_ID:256,WIDE_MODEL_ID:512,WIDE1024_ID:1024}
    spectral_conv = model_id in fusion_widths
    fusion_width = fusion_widths.get(model_id)
    peak_lr = BASE_LR*args.batch/256 if args.mae else .0001
    spectral_only = model_id=='R3-WholeScene-SpectralOnly'
    patch_m = 120 if legacy or spectral_conv else 60
    policy = None
    if args.early_stop_patience:
        policy = dict(patience=args.early_stop_patience,relative_min_delta=args.early_stop_min_delta,
                      min_epochs=args.early_stop_min_epochs,
                      metric='mean of A/B1 source-macro native MSE on Core VAL, mask exposure 0')
    validate = policy is not None or args.full_train
    prior_identity = read(args.out/'identity.json') if args.resume else None
    cache={}; layouts=defaultdict(list); members=[]; panel=[]; selected=defaultdict(int)
    wavelengths = []
    exclusions = []
    print('Packing all native scenes',flush=True)
    for i,row in enumerate(rows,1):
        key=row['id'];scale=subset['dn_scales'][row['domain']] if row['encoding']=='stored_dn' else 1.
        try:
            observation = data.load(row,'cpu')
            if spectral_conv:
                scene = pack_spectral_conv_scene(observation,scale,row['source'])
                wavelengths.append(dict(id=key, source=row['source'], bands=[
                    dict(band=b.key, center_um=center, provenance=provenance)
                    for b,center,provenance in zip(observation.bands,scene.metadata.wavelengths.tolist(),
                                                  scene.metadata.wavelength_sources,strict=True)]))
            else:
                scene = pack_scene(observation,scale,patch_m)
            masks(scene,key,0)
        except IneligiblePairedScene as error:
            if not args.full_train:
                raise
            exclusions.append(dict(id=key,source=row['source'],reason=str(error)))
            continue
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
    if args.full_train:
        rows = [row for row in rows if row['id'] in cache]
        write_json(args.out/'exclusions.json',exclusions)
        if len(rows)!=13020 or {entry['id'] for entry in exclusions}!={'FLAIR-HUB S2:4-14/22'}:
            raise ValueError('Paired full-scene eligibility differs from the audited release')
        print('Paired TRAIN',len(rows),'excluded',len(exclusions),flush=True)
    updates_per_epoch = math.ceil(len(rows)/args.batch)
    stats={k:[sum(x)/len(x) for x in zip(*v,strict=True)] for k,v in layouts.items()}
    write_json(args.out/'members.json',members)
    identity=dict(model_id=model_id,subset_sha256=digest(subset_file),
        core_sha256=CORE_SHA,extension_sha256=EXTENSION_SHA,members_sha256=digest(args.out/'members.json'),
        layouts=[dict(bands=k,train_means=v) for k,v in stats.items()],sources=dict(Counter(r['source'] for r in rows)),
        precision='BF16 linear/attention, FP32 residual/loss/optimizer',attention='exact efficient SDPA over each whole image or output tile',
        native_output_max_hw=[512,512],split_parents=sum(len(x['tiles'])>1 for x in members),
        tile_protocol='non-overlapping physical patches; native output bound; counts and masks on parent before partition',
        epochs=args.epochs,parents=len(rows),batch=args.batch,updates=updates_per_epoch*args.epochs,
        patch_m=patch_m,seed=17,lr=peak_lr,
        mask='A: select quarter of valid 120m groups; tiny-scene fallback select 60m cells; B1: cyclic eligible band',
        objective='paired A/B1 native target MSE, per band then per scene',eval='fixed TRAIN panel, at most 2 per source',
        panel=[r['id'] for r in panel],train_only=True)
    if args.full_train:
        identity['training_pool'] = 'all paired A/B1 TRAIN from core v4 and expansion v6'
        identity['exclusions_sha256'] = digest(args.out/'exclusions.json')
    if args.parallel_device:
        identity['parallel_device_candidate'] = args.parallel_device
    if args.mae:
        identity.update(schedule=dict(method='MAE step-wise linear warmup and half-cosine decay',
            base_lr=BASE_LR,peak_lr=peak_lr,min_lr=0.,warmup_epochs=WARMUP_EPOCHS,
            weight_decay=WEIGHT_DECAY,betas=BETAS,updates_per_epoch=updates_per_epoch,
            tails='retain; normalize by actual parent count',early_stop_clock='starts after warmup'))
    if spectral_conv:
        write_json(args.out/'wavelengths.json',dict(references=REFERENCE_URLS, parents=wavelengths))
        model_name = SPECTRAL_CONV_NAME if fusion_width==256 else f'{SPECTRAL_CONV_NAME} · {fusion_width}'
        identity.update(model_name=model_name, wavelengths_sha256=digest(args.out/'wavelengths.json'),
                        global_encoder_depth=2, branch_widths=[1024,1024], fusion_width=fusion_width,
                        encoder='wavelength-conditioned spectral ViT plus co-registered native residual spatial convolution')
    if validate:
        print('Packing frozen Core VAL-275',flush=True)
        val_rows,val_cache,manifest = pack_validation(data,cache,stats,patch_m)
        write_json(args.out/'val_members.json',manifest)
        identity.update(early_stopping=policy,validation_parents=len(val_rows),
                        validation_members_sha256=digest(args.out/'val_members.json'),train_only=False)
    starting_state = None
    if args.init_best is not None:
        starting_state = torch.load(args.init_best,map_location='cpu',weights_only=True)
        previous = starting_state['identity']
        bindings = ('model_id','core_sha256','extension_sha256','validation_members_sha256')
        if not args.full_train:
            bindings += ('members_sha256','wavelengths_sha256')
        for key in bindings:
            if previous[key]!=identity[key]:
                raise ValueError(f'Initial best checkpoint disagrees on {key}')
        identity['initialization'] = dict(kind='warm start from best; new AdamW and epoch schedule',
            path=str(args.init_best.resolve()),sha256=digest(args.init_best),
            source_epoch=starting_state['epoch'],source_updates=starting_state['updates'])
    preflight_started = time.perf_counter()
    # Exercise the largest full sequence and every native layout before any optimizer update.
    torch.manual_seed(17)
    model=MODELS[model_id](stats).to(args.device)
    if starting_state is not None:
        if args.full_train:
            matched = transfer_full_heads(model,starting_state)
            identity['initialization']['restored_layouts'] = matched
            identity['initialization']['new_layouts'] = len(stats)-matched
        else:
            model.load_state_dict(starting_state['model'])
        del starting_state
    if model_id==DUAL_AXIS_ID:
        identity.update(initialization='seed17, all common tensors identical to spatial ViT initialization',
                        encoder='one global spatial block, one per-patch spectral block, one per-patch fusion block')
    if legacy:
        identity['factorized_spatial_layouts'] = [list(k) for k,idx in model.layout_indices.items()
                                                 if not spectral_only and model.projections[idx].factorized]
        identity['global_encoder_depth'] = len(model.global_blocks)
        identity['spectral_only'] = spectral_only
        identity['initialization'] = 'seed17; unchanged tensors retain baseline initialization'
    if args.accelerated:
        identity['runtime'] = 'resident native scene cache; CPU masks; vectorized per-band objective'
    if args.packed:
        identity['packing'] = 'compatible native views; bulk host transfers; independent padded attention'
        identity['packed_max_patch_tokens'] = 1024 if fusion_width==512 else 256
    old_identity = None if not args.resume else {k:v for k,v in prior_identity.items() if k!='runtime'}
    new_identity = {k:v for k,v in json.loads(json.dumps(identity)).items() if k!='runtime'}
    if args.resume and new_identity!=old_identity:
        raise ValueError('Resume settings or dataset differ from the existing run')
    write_json(args.out/'identity.json',identity)
    write_json(args.out/'status.json',dict(status='PREFLIGHT'))
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
    breakdown = defaultdict(int)
    for name,p in model.named_parameters():
        breakdown[name.split('.')[0]] += p.numel()
    write_json(args.out/'preflight.json',dict(measured=measured,parameters=sum(breakdown.values()),
                                            parameter_groups=dict(breakdown)))
    print('parameters',sum(breakdown.values()),flush=True)
    decay = WEIGHT_DECAY if args.mae else .04
    groups=[dict(params=[p for p in model.parameters() if p.ndim>1],weight_decay=decay),
            dict(params=[p for p in model.parameters() if p.ndim==1],weight_decay=0.)]
    optimizer=torch.optim.AdamW(groups,lr=peak_lr,betas=BETAS if args.mae else (.9,.999))
    viewer=VIEWER/args.out.name
    definitions={f'{t}_{k}':dict(label=f'{t} TRAIN {k}',direction='min',role='loss',unit='')
                 for t in TASKS for k in ('mse','centered_mse','within_patch_mse','template_fraction')}
    charts = ['A_mse','B1_mse','A_within_patch_mse','B1_within_patch_mse']
    if validate:
        for key,label in (('A_mse','A VAL MSE'),('B1_mse','B1 VAL MSE'),('paired_mse','A/B1 VAL MSE')):
            definitions[f'val_{key}'] = dict(label=label,direction='min',role='loss',unit='')
        charts = ['val_paired_mse','val_A_mse','val_B1_mse']
    if policy:
        definitions['early_stop_stale_epochs'] = dict(label='Early-stop wait',direction='none',role='other',unit='epoch')
        charts.append('early_stop_stale_epochs')
    if args.mae:
        definitions['lr'] = dict(label='Learning rate',direction='none',role='other',unit='')
    history=[]
    stop_state={}
    best_path = args.out/'best.pt'
    initial_eval_started = time.perf_counter()
    if args.resume:
        state = torch.load(args.out/'last.pt',map_location='cpu',weights_only=True)
        saved = {k:v for k,v in json.loads(json.dumps(state['identity'])).items() if k!='runtime'}
        if saved!=old_identity:
            raise ValueError('Checkpoint identity differs from the run')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['cpu_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'],args.device)
        history = state['history']
        if not history or state['epoch']!=len(history) or history[-1]['updates']!=updates_per_epoch*len(history):
            raise ValueError('Checkpoint epoch and update counts disagree')
        stop_state = history[-1].get('early_stop',{})
        if stop_state and stop_state['best_epoch']==len(history):
            save_best(best_path,model,identity,history[-1])
        del state
        write_json(args.out/'history.json',history)
    else:
        write_json(args.out/'initial.json',evaluate(model,panel,cache,args.device))
        if validate:
            initial_val = evaluate(model,val_rows,val_cache,args.device,False)
            write_json(args.out/'initial_val.json',initial_val)
            if args.init_best is not None:
                val = initial_val['aggregate']
                score = sum(val[task]['source_macro']['mse'] for task in TASKS)/len(TASKS)
                stop_state,_ = early_stop_state({},score,0,policy)
                save_best(best_path,model,identity,dict(epoch=0,updates=0,validation=val,
                                                        validation_paired_mse=score))
    device_cache = {key:scene.to(args.device) for key,scene in cache.items()} if args.accelerated else None
    signatures = ({key:scene_signature(scene,max_patch_tokens=identity['packed_max_patch_tokens'])
                   for key,scene in cache.items()} if args.packed else None)
    replica = None
    if args.parallel_device:
        replica,measured = tune_packed_parallel(model,rows,args.batch,cache,device_cache,signatures,masks,
                                               args.device,args.parallel_device)
        write_json(args.out/'parallel_preflight.json',measured)
        print('packed parallel preflight',measured,flush=True)
    epochs_started = time.perf_counter()
    with RunWriter(viewer,experiment=dict(id='pd_here_r3_whole_scene' if patch_m==120 else 'pd_here_s60_whole_scene',
                   name=f'Full-scene {len(rows)} TRAIN'),
                   variant=dict(id=model_id,name=identity.get('model_name',model_id),
                                kind='modification' if spectral_conv else 'baseline',
                                description=identity.get('encoder','') if spectral_conv else ''),
                   config=identity,metrics=definitions,charts=charts,
                   numbers=['lr'] if args.mae else None,resume=args.resume) as log:
        epoch_end = len(history) if stop_state.get('triggered') else args.epochs
        for epoch in range(len(history)+1,epoch_end+1):
            started=time.perf_counter()
            ordered=rows.copy();random.Random(17+epoch).shuffle(ordered)
            totals={task:0. for task in TASKS}
            updates=updates_per_epoch*(epoch-1)
            progress = dict(status='RUNNING',epochs=epoch-1,current_epoch=epoch,updates=updates,
                            max_epochs=args.epochs,viewer=str(viewer),early_stop=stop_state or None)
            write_json(args.out/'status.json',progress)
            for batch_index,i in enumerate(range(0,len(ordered),args.batch)):
                group=ordered[i:i+args.batch]
                if args.mae:
                    lr = mae_lr(peak_lr,epoch-1+batch_index/updates_per_epoch,args.epochs)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                if signatures is not None:
                    losses = (replica.backward(group,epoch-1) if replica is not None else
                              batched_backward(model,group,cache,device_cache,signatures,masks,epoch-1))
                    values = losses.tolist()
                    if not all(math.isfinite(value) for value in values):
                        raise FloatingPointError('Nonfinite packed batch loss')
                    for task,value in zip(TASKS,values,strict=True):
                        totals[task] += value
                elif device_cache is not None:
                    losses = []
                    for row in group:
                        parent = row['id']
                        cpu = cache[parent]
                        losses.append(prepared_backward(model,cpu,device_cache[parent],
                                                        masks(cpu,parent,epoch-1),.5/len(group)))
                    values = torch.stack(losses).sum(0).tolist()
                    if not all(math.isfinite(value) for value in values):
                        raise FloatingPointError('Nonfinite prepared batch loss')
                    for task,value in zip(TASKS,values,strict=True):
                        totals[task] += value
                else:
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
                    write_json(args.out/'status.json',progress | {'updates':updates})
                    print('updates',updates,'epoch',epoch,'elapsed_s',round(time.perf_counter()-started,1),flush=True)
            evaluated=evaluate(model,panel,cache,args.device)
            item=dict(epoch=epoch,updates=updates,train={k:v/len(rows) for k,v in totals.items()},
                      fixed_train=evaluated)
            viewer_values={f'{task}_{k}':value for task in TASKS for k,value in
                           evaluated['aggregate'][task]['source_macro'].items()}
            if validate:
                val = evaluate(model,val_rows,val_cache,args.device,False)['aggregate']
                task_scores = {task:val[task]['source_macro']['mse'] for task in TASKS}
                score = sum(task_scores.values())/len(TASKS)
                stop_state,is_best = early_stop_state(stop_state,score,epoch,policy)
                if args.mae and policy:
                    after_warmup(stop_state,epoch)
                item.update(validation=val,validation_paired_mse=score,early_stop=stop_state)
                viewer_values.update({f'val_{task}_mse':value for task,value in task_scores.items()},
                                     val_paired_mse=score)
                if policy:
                    viewer_values['early_stop_stale_epochs'] = stop_state['stale_epochs']
                print('VAL',epoch,task_scores,'best',stop_state['best_epoch'],
                      'stale',stop_state.get('stale_epochs'),flush=True)
            if args.mae:
                item['lr'] = viewer_values['lr'] = optimizer.param_groups[0]['lr']
            item['seconds'] = time.perf_counter()-started
            history.append(item)
            save(args.out/'last.pt',model,optimizer,epoch,identity,history,args.device)
            if validate and is_best:
                save_best(best_path,model,identity,item)
            write_json(args.out/'history.json',history)
            log.log(updates,viewer_values,epoch=epoch)
            print('epoch',epoch,item['train'],round(item['seconds'],1),flush=True)
            if stop_state.get('triggered'):
                break
    if replica is not None:
        replica.close()
    finished = time.perf_counter()
    write_json(args.out/'timing.json',dict(packing_seconds=preflight_started-run_started,
        preflight_seconds=initial_eval_started-preflight_started,
        initial_eval_seconds=epochs_started-initial_eval_started,
        train_and_epoch_eval_seconds=sum(item['seconds'] for item in history),
        epochs_wall_seconds=finished-epochs_started,total_wall_seconds=finished-run_started,
        epoch_seconds=[item['seconds'] for item in history]))
    reason = 'early_stopping' if stop_state.get('triggered') and len(history)<args.epochs else 'max_epochs'
    write_json(args.out/'status.json',dict(status='COMPLETED',epochs=len(history),
        updates=history[-1]['updates'],viewer=str(viewer),stop_reason=reason,early_stop=stop_state or None))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--accelerated',action='store_true')
    parser.add_argument('--packed',action='store_true')
    parser.add_argument('--parallel-device',help='Use a second GPU when the packed benchmark is faster')
    parser.add_argument('--full-train',action='store_true',help='Use all paired TRAIN members')
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--mae',action='store_true',help='Use the MAE pretraining learning-rate and AdamW recipe')
    parser.add_argument('--init-best',type=Path,help='Initialize the 256 variant from an immutable best checkpoint')
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--early-stop-patience',type=int,default=0)
    parser.add_argument('--early-stop-min-delta',type=float,default=.001,help='Relative improvement')
    parser.add_argument('--early-stop-min-epochs',type=int,default=30)
    parser.add_argument('--resume',action='store_true',help='Continue from last.pt in the same run')
    architectures=parser.add_mutually_exclusive_group()
    for option,model_id in zip(('--legacy-dual','--legacy-dual-one-layer','--legacy-spectral-only','--dual-axis',
                               '--spectral-transformer-spatial-conv','--spectral-transformer-spatial-conv-512',
                               '--spectral-transformer-spatial-conv-1024'),
                               (*LEGACY_MODELS,DUAL_AXIS_ID,SPECTRAL_CONV_ID,WIDE_MODEL_ID,WIDE1024_ID),strict=True):
        architectures.add_argument(option,dest='model_id',action='store_const',const=model_id)
    parser.set_defaults(model_id='S60-WholeScene-SpatialViT')
    args=parser.parse_args()
    eligible = (SPATIAL_ID,SPECTRAL_CONV_ID,WIDE_MODEL_ID,WIDE1024_ID)
    if args.batch<1 or args.batch>32:
        parser.error('--batch must be between 1 and 32')
    if args.packed and (not args.accelerated or args.model_id not in eligible):
        parser.error('--packed requires --accelerated and a supported architecture')
    if args.mae and (args.epochs<=WARMUP_EPOCHS or (args.early_stop_patience and
            args.early_stop_min_epochs<WARMUP_EPOCHS+args.early_stop_patience)):
        parser.error('MAE requires a post-warmup budget and enough epochs for full patience')
    if args.epochs<1 or args.early_stop_patience<0 or not 0<=args.early_stop_min_delta<1:
        parser.error('Epochs must be positive, patience nonnegative and relative delta in [0, 1)')
    if args.early_stop_patience and (args.model_id not in eligible
                                    or not 1<=args.early_stop_min_epochs<=args.epochs):
        parser.error('Early stopping requires a supported variant and a valid minimum epoch')
    if args.parallel_device and not args.packed:
        parser.error('--parallel-device requires --packed')
    if args.full_train and (args.init_best is None or not args.packed or args.model_id not in eligible):
        parser.error('--full-train requires --init-best, --packed and a supported architecture')
    if args.init_best is not None and not args.full_train and (args.model_id!=SPECTRAL_CONV_ID or not args.early_stop_patience):
        parser.error('Subset --init-best requires the 256 spectral-conv variant and early stopping')
    if args.resume:
        if not (args.out/'last.pt').is_file() or read(args.out/'status.json')['status']=='COMPLETED':
            parser.error('Resume requires an unfinished run and last.pt')
    else:
        args.out.mkdir(parents=True,exist_ok=False)
    try:
        run(args)
    except Exception as error:
        write_json(args.out/'status.json',dict(status='FAILED',error=str(error)))
        raise


if __name__=='__main__':
    main()
