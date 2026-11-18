"""R3 staged single-GPU FP32 runner. G3 requires reviewed G2 evidence."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import ssl
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pd_here_geotoken_dualvit_r3.data import freeze,geometry,windows,load_window,batch,masks,write_json,OpticalDataset
from pd_here_geotoken_dualvit_r3.model import GeoTokenDualViT,masked_reconstruction_loss
from pd_here_geotoken_dualvit_r3.evaluation import metrics,a_baseline,fit_ols,b_baseline
from encoder_v3.latent_response.tvlp_writer import RunWriter


def model_for(layout,device):
    torch.manual_seed(17)
    model=GeoTokenDualViT(layout).to(device)
    decay=[];other=[]
    for name,p in model.named_parameters():
        (other if p.ndim==1 or name.endswith('bias') else decay).append(p)
    optimizer=torch.optim.AdamW([dict(params=decay,weight_decay=.04),dict(params=other,weight_decay=0.)],lr=1e-4,betas=(.9,.999))
    return model,optimizer


def update(model,opt,values,qa,views,task):
    opt.zero_grad(set_to_none=True);losses={}
    for name,view in zip(('A','B1'),views):
        if task not in (name,'paired'):continue
        output=model(values,qa,*view)
        loss=masked_reconstruction_loss(output['prediction'],values,qa,*view,name)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        (loss*(.5 if task=='paired' else 1.)).backward()
        losses[name]=loss.item()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    opt.step()
    return losses,float(norm)


def prepare(out):
    train,val,identity=freeze(ROOT,out)
    layout,_=geometry(train[0]);dataset=OpticalDataset(ROOT);panel={};excluded=[]
    for role,rows in [('train',train),('val',val)]:
        for index,row in enumerate(rows):
            current,origins=windows(row)
            if current!=layout:raise ValueError('Mixed band layouts')
            order=list(range(len(origins)));random.Random(f'17:{row["id"]}:windows').shuffle(order)
            for window_id in order:
                sample=load_window(dataset,row,origins[window_id],layout)
                _,q=batch([sample],'cpu')
                try:
                    # Require every scheduled B1 band to be legal on the initial panel.
                    for exposure in range(len(layout.band_keys)):masks(q,[row['id']],[window_id],exposure)
                except ValueError:continue
                panel[row['id']]=dict(role=role,window_id=window_id,origin=origins[window_id],window_order=order)
                break
            else:excluded.append(dict(id=row['id'],role=role,reason='no initial window supports A and every B1 band'))
            if index%100==0:print(f'prepare {role} {index}/{len(rows)}',flush=True)
    identity.update(layout=asdict(layout),layout_hash=layout.layout_hash,excluded=excluded,
                    train_eligible=sum(v['role']=='train' for v in panel.values()),val_eligible=sum(v['role']=='val' for v in panel.values()),
                    qa='frozen QA intersect finite and native window support',input_resampling=False,
                    initial_panel='deterministic first legal window; fixed throughout G1/G2',
                    rotation='G3 requires registered window rotation before launch')
    write_json(out/'window_plan.json',panel)
    identity['window_plan_sha256']=hashlib.sha256((out/'window_plan.json').read_bytes()).hexdigest()
    write_json(out/'dataset_identity.json',identity)
    write_json(out/'g0_data.json',dict(status='PASS',parents=len(panel),excluded=excluded,layout_hash=layout.layout_hash))
    return train,val,layout,panel,identity


def load_panel(out,device):
    rows=[json.loads(l) for l in (out/'train.jsonl').read_text().splitlines()]
    panel=json.loads((out/'window_plan.json').read_text());rows=[r for r in rows if r['id'] in panel][:16]
    if len(rows)!=16:raise ValueError('Need 16 eligible TRAIN parents')
    layout,_=geometry(rows[0]);dataset=OpticalDataset(ROOT)
    samples=[load_window(dataset,r,panel[r['id']]['origin'],layout) for r in rows]
    values,qa=batch(samples,device)
    ids=[r['id'] for r in rows];wins=[panel[i]['window_id'] for i in ids]
    return layout,values,qa,ids,wins


def writer(out,task,identity):
    definitions={key:dict(label=label,direction='min',role='loss',unit='') for key,label in
                 [('loss_a','A TRAIN MSE'),('loss_b','B1 TRAIN MSE'),('eval_a','A 固定 TRAIN MSE'),('eval_b','B1 固定 TRAIN MSE')]}
    definitions.update(seconds=dict(label='更新秒数',direction='none',role='other',unit='s'),
                       gradient_norm=dict(label='裁剪前梯度范数',direction='none',role='other',unit=''))
    return RunWriter(out/'viewer'/task,experiment=dict(id='pd_here_geotoken_dualvit_r3',name='GeoToken DualViT R3'),
                     variant=dict(id=task,name=f'G2 {task} · 16 TRAIN · FP32',kind='baseline'),metrics=definitions,
                     charts=['loss_a','loss_b','eval_a','eval_b'],numbers=['seconds','gradient_norm'],config=identity)


@torch.no_grad()
def evaluate(model,values,qa,views):
    model.eval();result={}
    for name,view in zip(('A','B1'),views):
        result[name]=metrics(model(values,qa,*view)['prediction'],values,qa,*view,name)
    model.train();return result


def benchmark(out,device):
    layout,values,qa,ids,wins=load_panel(out,device);views=masks(qa,ids,wins,0)
    report=dict(device=torch.cuda.get_device_name(device),precision='FP32',global_parent_batch=16,
                warmup=20,measured=50,ddp='NOT_RUN; single GPU selected',bf16='NOT_RUN',tasks={})
    for task in ('A','B1','paired'):
        model,opt=model_for(layout,device);times=[];torch.cuda.reset_peak_memory_stats(device)
        for step in range(70):
            torch.cuda.synchronize(device);start=time.perf_counter()
            update(model,opt,values,qa,views,task)
            torch.cuda.synchronize(device)
            if step>=20:times.append(time.perf_counter()-start)
        report['tasks'][task]=dict(p50=float(np.percentile(times,50)),p95=float(np.percentile(times,95)),
            parents_per_second=16/float(np.median(times)),peak_allocated=torch.cuda.max_memory_allocated(device),
            peak_reserved=torch.cuda.max_memory_reserved(device),trace={n:model(values,qa,*v)['trace'] for n,v in zip(('A','B1'),views)})
        print(f'G1 {task}: {report["tasks"][task]}',flush=True)
        del model,opt;torch.cuda.empty_cache()
    report['status']='PASS';write_json(out/'g1_profile.json',report)


def diagnostics(out,device):
    if json.loads((out/'g1_profile.json').read_text())['status']!='PASS':raise ValueError('G1 required')
    layout,values,qa,ids,wins=load_panel(out,device);fixed=masks(qa,ids,wins,0)
    identity=dict(stage='G2',diagnostic_only=True,parents=ids,windows=wins,layout_hash=layout.layout_hash,
                  seed=17,updates_per_arm=200,global_parent_batch=16,precision='FP32',optimizer='AdamW',lr=1e-4,
                  mask_rule='A and B1 deterministic exposure rotation; fixed evaluation exposure=0',
                  authorization='user requested implementation and training in this chat',pid=os.getpid())
    identity['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),*sorted((ROOT/'src/pd_here_geotoken_dualvit_r3').glob('*.py'))]}
    write_json(out/'training_identity.json',identity)
    coeff=fit_ols(values,qa)
    baselines={method:metrics(a_baseline(values,qa,fixed[0][0],method),values,qa,*fixed[0],'A') for method in ('constant','nearest','idw')}
    baselines['B1_OLS']=metrics(b_baseline(values,qa,fixed[1][1],coeff),values,qa,*fixed[1],'B1')
    write_json(out/'g2_baselines.json',dict(scope='same 16 TRAIN parents only; OLS TRAIN fit',mapping='10m to20m valid area average; predicted20m repeated to10m',metrics=baselines))
    summary={}
    for task in ('A','B1','paired'):
        if task=='paired' and not all(summary[t]['learning'] for t in ('A','B1')):
            write_json(out/'status.json',dict(status='STOPPED_G2',reason='single task learning gate failed',summary=summary));return
        model,opt=model_for(layout,device);initial=evaluate(model,values,qa,fixed)
        with writer(out,task,identity) as log:
            log.log(0,dict(eval_a=initial['A']['mse'],eval_b=initial['B1']['mse']))
            start=time.perf_counter()
            for step in range(1,201):
                views=masks(qa,ids,wins,step-1);torch.cuda.synchronize(device);tick=time.perf_counter()
                losses,norm=update(model,opt,values,qa,views,task)
                torch.cuda.synchronize(device)
                fields={('loss_a' if t=='A' else 'loss_b'):v for t,v in losses.items()}
                fields.update(seconds=time.perf_counter()-tick,gradient_norm=norm)
                if step%20==0:
                    ev=evaluate(model,values,qa,fixed);fields.update(eval_a=ev['A']['mse'],eval_b=ev['B1']['mse'])
                    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,identity=identity,task=task,
                                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),out/f'{task}_last.pt')
                    print(f'G2 {task} {step}/200: {fields}',flush=True)
                log.log(step,fields)
                write_json(out/'status.json',dict(status='TRAINING',stage='G2',task=task,step=step,pid=os.getpid(),losses=losses))
        final=evaluate(model,values,qa,fixed)
        relevant=('A','B1') if task=='paired' else (task,)
        summary[task]=dict(initial=initial,final=final,seconds=time.perf_counter()-start,
            learning=all(final[t]['mse']<initial[t]['mse'] and final[t]['centered_mse']<initial[t]['centered_mse'] for t in relevant))
        write_json(out/'g2_results.json',summary)
        del model,opt;torch.cuda.empty_cache()
    write_json(out/'status.json',dict(status='G2_COMPLETE_REVIEW_REQUIRED',summary=summary,
        g3='NOT_RUN pending reconstruction/structure gate review; no automatic long training'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','benchmark','diagnostics','all'])
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/pd_here_geotoken_dualvit_r3')
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        if args.stage in ('prepare','all'):prepare(out)
        if args.stage in ('benchmark','all'):benchmark(out,args.device)
        if args.stage in ('diagnostics','all'):diagnostics(out,args.device)
    except Exception as error:
        write_json(out/'status.json',dict(status='FAILED',error=repr(error),pid=os.getpid()));raise

if __name__=='__main__':main()
