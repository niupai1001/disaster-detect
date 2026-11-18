"""Matched R3 texture experiments on an audited split."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from train_pd_here_geotoken_dualvit_r3 import ROOT, model_for, update
from audit_pd_here_r3_sources import BASE, digest, read
from pd_here_geotoken_dualvit_r3.data import OpticalDataset, batch, geometry, load_window, masks, write_json
from pd_here_geotoken_dualvit_r3.model import GeoTokenDualViT
from pd_here_geotoken_dualvit_r3.evaluation import (
    a_baseline, b_baseline, coarse, metric_records, reduce_records,
)
from encoder_v3.latent_response.tvlp_writer import RunWriter

VARIANTS = ('baseline', 'input_scale', 'decoder128')
TASKS = ('A', 'B1')


def batches(rows):
    for start in range(0, len(rows), 16):
        yield start, rows[start:start+16]


def initial_model(layout, variant, stats, device):
    if variant == 'band_tokens256':
        from pd_here_geotoken_dualvit_r3.band_tokens import BandTokenDualViT
        reference,_ = initial_model(layout,'fusion256_global2_zerohead',stats,'cpu')
        torch.manual_seed(17)
        model = BandTokenDualViT(layout)
        current = model.state_dict()
        for key,value in reference.state_dict().items():
            if key in current and current[key].shape == value.shape:
                current[key] = value
        model.load_state_dict(current)
        with torch.no_grad():
            for head in model.prediction.values():
                head.weight.zero_()
                head.bias.zero_()
            model.band_offset.copy_(torch.as_tensor(stats[0]))
        model.to(device)
        groups = [dict(params=[p for name,p in model.named_parameters() if p.ndim>1 and not name.endswith('bias')],weight_decay=.04),
                  dict(params=[p for name,p in model.named_parameters() if p.ndim==1 or name.endswith('bias')],weight_decay=0.)]
        return model,torch.optim.AdamW(groups,lr=1e-4)
    direct_concat = variant == 'concat256_global2_zerohead'
    linear_branches = variant == 'linear_branches256'
    wide_projection = variant in ('fusion256_global2_zerohead','linear_branches256')
    if linear_branches:
        reference,_ = initial_model(layout,'fusion256_global2_zerohead',stats,'cpu')
    else:
        reference, _ = (initial_model(layout,'global2_zerohead',stats,'cpu') if direct_concat or wide_projection
                        else model_for(layout, 'cpu'))
    torch.manual_seed(17)
    model = GeoTokenDualViT(layout, decoder_width=128 if variant in ('decoder128','decoder128_zerohead','global2_zerohead','concat256_global2_zerohead','fusion256_global2_zerohead','linear_branches256') else 64,
                           input_stats=stats if variant == 'input_scale' else None,
                           global_depth=2 if variant in ('global2_zerohead','concat256_global2_zerohead','fusion256_global2_zerohead','linear_branches256') else 1,
                           fusion_projection=not direct_concat, fusion_width=256 if wide_projection else None,
                           branch_transformer=not linear_branches)
    current = model.state_dict()
    for key, value in reference.state_dict().items():
        if key in current and current[key].shape == value.shape:
            current[key] = value
    model.load_state_dict(current)
    if variant in ('decoder128_zerohead','global2_zerohead','concat256_global2_zerohead','fusion256_global2_zerohead','linear_branches256'):
        # Begin with TRAIN band means so random per-pixel weights cannot impose a shared texture.
        with torch.no_grad():
            model.prediction.weight.zero_()
            offset = 0
            for mean,(height,width) in zip(stats[0],layout.native_shapes,strict=True):
                pixels = height*width
                model.prediction.bias[offset:offset+pixels].fill_(float(mean))
                offset += pixels
    model.to(device)
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if param.ndim == 1 or name.endswith('bias') else decay).append(param)
    optimizer = torch.optim.AdamW([dict(params=decay,weight_decay=.04),
                                   dict(params=no_decay,weight_decay=0.)],lr=1e-4)
    return model, optimizer


class Study:
    def __init__(self, release, device):
        self.release = release
        self.device = device
        self.identity = read(release/'dataset_identity.json')
        audit = release/'source_audit.json'
        if (self.identity['status'] != 'PASS' or read(audit)['status'] != 'PASS'
                or digest(audit) != self.identity['source_audit_sha256']):
            raise ValueError('Source audit must pass before real-data training')
        self.rows = {}
        for role, count in [('train',700),('val',300)]:
            manifest = release/f'{role}.jsonl'
            if digest(manifest) != self.identity[f'{role}_sha256']:
                raise ValueError('Audited split digest changed')
            self.rows[role] = [json.loads(line) for line in manifest.read_text().splitlines()]
            if len(self.rows[role]) != count:
                raise ValueError('Unexpected comparison split size')
        window_plan = BASE/'window_plan.json'
        if digest(window_plan) != self.identity['window_plan_sha256']:
            raise ValueError('Frozen window plan digest changed')
        self.panel = read(window_plan)
        self.layout, _ = geometry(self.rows['train'][0])
        dataset = OpticalDataset(ROOT)
        self.cache = {}
        for role,rows in self.rows.items():
            for row in rows:
                entry = self.panel[row['id']]
                if entry['role'] != role or geometry(row)[0] != self.layout:
                    raise ValueError('Audited window or layout changed')
                self.cache[row['id']] = load_window(dataset,row,entry['origin'],self.layout)

    def inputs(self, rows, exposure):
        ids = [row['id'] for row in rows]
        values, qa = batch([self.cache[i] for i in ids],self.device)
        return values,qa,masks(qa,ids,[self.panel[i]['window_id'] for i in ids],exposure)

    def train_references(self):
        moments = [[] for _ in self.layout.band_keys]
        energies = {task:defaultdict(list) for task in TASKS}
        grams = None
        for _,rows in batches(self.rows['train']):
            values,qa,views = self.inputs(rows,0)
            for j,(value,valid) in enumerate(zip(values,qa,strict=True)):
                for x,q in zip(value,valid,strict=True):
                    if q.any():
                        z=x[q].double()
                        moments[j].append((z.mean().item(),z.square().mean().item()))
            x,q = coarse(values,qa)
            x = x[q.all(-1)].double()
            x = torch.cat((torch.ones_like(x[:,:1]),x),dim=-1)
            gram = (x.T@x).cpu()
            grams = gram if grams is None else grams+gram
            targets = torch.cat([x.flatten(2) for x in values],dim=-1)
            for task,view in zip(TASKS,views,strict=True):
                for r in metric_records(targets,values,qa,*view,task):
                    if 'truth_gradient_energy' in r:
                        energies[task][r['band']].append(r['truth_gradient_energy'])
        mean, second = np.asarray([np.mean(x,axis=0) for x in moments]).T
        thresholds = {t:{j:float(np.quantile(x,.75)) for j,x in e.items()} for t,e in energies.items()}
        coefficients = []
        for j in range(len(mean)):
            indices = [k for k in range(len(mean)+1) if k != j+1]
            gram = grams[indices][:,indices]
            coefficients.append(torch.linalg.pinv(gram)@grams[indices,j+1:j+2])
        return (mean,np.sqrt(np.maximum(second-mean**2,0))),thresholds,coefficients

    @torch.no_grad()
    def score(self, model, thresholds, coefficients=None):
        records = defaultdict(list)
        model.eval()
        for start,rows in batches(self.rows['val']):
            values,qa,views = self.inputs(rows,0)
            for task,view in zip(TASKS,views,strict=True):
                keep,bands = view
                estimates = {'model':model(values,qa,*view)['prediction']}
                if coefficients is not None:
                    if task == 'A':
                        estimates.update({key:a_baseline(values,qa,keep,key)
                                          for key in ('constant','nearest','idw')})
                    else:
                        estimates['ols'] = b_baseline(values,qa,bands,coefficients)
                if task == 'A':
                    support = tuple(q & (q & keep[:,:,None,None]).flatten(1).any(-1)[:,None,None,None]
                                    for q in qa)
                else:
                    # Apply the same support at every epoch, for the model and OLS.
                    _,coarse_valid = coarse(values,qa)
                    available = (coarse_valid | ~bands[:,None,None,None,:]).all(-1)
                    support = tuple(q & (available.repeat_interleave(2,-1).repeat_interleave(2,-2)
                                          if q.shape[-1]==12 else available) for q in qa)
                for name,prediction in estimates.items():
                    for record in metric_records(prediction,values,support,*view,task):
                        record['parent'] += start
                        records[name,task].append(record)
        model.train()
        result = {}
        for (name,task),items in records.items():
            result.setdefault(name,{})[task] = {}
            for scope, subset in [('all',items),('high_texture',[
                    x for x in items if x.get('truth_gradient_energy',-1) >= thresholds[task].get(x['band'],float('inf'))])]:
                values,counts = reduce_records(subset,counts=True)
                result[name][task][scope] = dict(metrics=values, valid_parents=counts)
        return result


def train(study, variant, out, resume):
    stats, thresholds, coefficients = study.train_references()
    model, optimizer = initial_model(study.layout,variant,stats,study.device)
    identity = dict(variant=variant,seed=17,epochs=30,updates=1320,lr=1e-4,weight_decay=.04,
                    data=study.identity,layout_hash=study.layout.layout_hash,
                    statistics=[x.tolist() for x in stats],thresholds=thresholds,
                    precision='FP32',batch=16,schedule='shuffle seed17+epoch; train exposure epoch-1; VAL exposure 0')
    if variant == 'global2_zerohead':
        identity['cross_geographic_depth'] = 2
        identity['comparison'] = 'decoder128_zerohead, same initialization for common parameters'
    if variant == 'concat256_global2_zerohead':
        identity.update(cross_geographic_depth=2,geographic_width=256,fusion_projection=False,
                        comparison='global2_zerohead; concat directly into global self-attention')
    if variant == 'fusion256_global2_zerohead':
        identity.update(cross_geographic_depth=2,geographic_width=256,fusion_projection='Linear(256,256)',
                        comparison='global2_zerohead; 256to256 fusion projection then two global ViT blocks')
    if variant == 'linear_branches256':
        identity.update(cross_geographic_depth=2,geographic_width=256,fusion_projection='Linear(256,256)',
                        branch_transformer=False,
                        comparison='fusion256_global2_zerohead; remove local Attention and FFN; keep global ViT')
    if variant == 'band_tokens256':
        identity.update(cross_geographic_depth=2,global_token_width=256,
                        token_unit='(geographic patch, band)',spatial_embedding='whole native patch Linear; no spatial Transformer',
                        decoder='160 band/geo slots; shared native-shape Linear heads, trainable band means',
                        comparison='fusion256_global2_zerohead; keep per-band features before reducers')
    out.mkdir(parents=True,exist_ok=True)
    checkpoint = out/'last.pt'
    history = []
    if resume:
        state = torch.load(checkpoint,map_location='cpu',weights_only=True)
        if state['identity'] != identity:
            raise ValueError('Checkpoint identity differs')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'],study.device)
        history = state['history']
    elif checkpoint.exists():
        raise ValueError('Existing run requires --resume')
    if len(history) == 30:
        write_json(out/'history.json',history)
        write_json(out/'status.json',dict(status='COMPLETED',epochs=30,updates=1320))
        return
    write_json(out/'identity.json',identity)
    definitions = {f'{t}_{k}':dict(label=f'{t} VAL {k}',direction='min',role='loss',unit='')
                   for t in TASKS for k in ('mse','centered_mse','gradient_mse')}
    viewer = BASE/'viewer'/f'texture30_{out.name}_{variant}'
    with RunWriter(viewer,experiment=dict(id='pd_here_r3_texture30',name='R3 texture study'),
                   variant=dict(id=variant,name=variant,kind='baseline'),config=identity,metrics=definitions,
                   charts=['A_centered_mse','B1_centered_mse','A_gradient_mse','B1_gradient_mse'],
                   resume=resume and (viewer/'run.json').exists()) as log:
        if not resume:
            write_json(out/'reference_metrics.json',study.score(model,thresholds,coefficients))
        for epoch in range(len(history)+1,31):
            started = time.perf_counter()
            ordered = list(study.rows['train'])
            random.Random(17+epoch).shuffle(ordered)
            train_sum = {t:0. for t in TASKS}
            for _,rows in batches(ordered):
                losses,_ = update(model,optimizer,*study.inputs(rows,epoch-1),'paired')
                for task,value in losses.items():
                    train_sum[task] += value*len(rows)
            val = study.score(model,thresholds)['model']
            history.append(dict(epoch=epoch,updates=44*epoch,
                                train={t:v/700 for t,v in train_sum.items()},val=val,
                                seconds=time.perf_counter()-started))
            temporary = out/'last.tmp'
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(), identity=identity,
                            history=history,torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state(study.device)),temporary)
            temporary.replace(checkpoint)
            write_json(out/'history.json',history)
            log.log(44*epoch,{key:val[t]['all']['metrics'][k] for key in definitions
                             for t,k in [key.split('_',1)]},epoch=epoch)
            print(variant, epoch, history[-1]['train'], flush=True)
    write_json(out/'status.json',dict(status='COMPLETED',epochs=30,updates=1320))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('variant',choices=(*VARIANTS,'decoder128_zerohead','global2_zerohead','concat256_global2_zerohead','fusion256_global2_zerohead','linear_branches256','band_tokens256'))
    parser.add_argument('--release',type=Path,default=BASE/'texture_study')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    try:
        train(Study(args.release,args.device),args.variant,args.out,args.resume)
    except Exception as error:
        write_json(args.out/'status.json',dict(status='FAILED',error=str(error)))
        raise


if __name__ == '__main__':
    main()
