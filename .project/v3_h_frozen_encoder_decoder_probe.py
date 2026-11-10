"""Matched Decoder-only controls using the original immutable physical Encoder."""
import ssl,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING','0')
import argparse,json,sys,time,hashlib
import numpy as np
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]/'earth-representation'
RELEASE=(ROOT/'outputs/encoder_v3/hierarchical/release_a_only_v1').resolve()
sys.path.insert(0,str(RELEASE/'runtime/src'))
from encoder_v3.hierarchical.train import load_data,build_model,sample_from_dataset,identity,write
from encoder_v3.hierarchical.verification import fingerprint
from encoder_v3.hierarchical.geometry import locate
from scipy.sparse import coo_matrix


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['original','band_film'],required=True)
    p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=128)
    p.add_argument('--analyze',action='store_true');p.add_argument('--engineering-checkpoint',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(17);torch.cuda.set_device(args.device)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=json.loads((RELEASE/'formal_run/config.json').read_text());datasets,resolver=load_data()
    checkpoint=torch.load(RELEASE/'formal_run/best.pt',map_location='cpu',weights_only=False)
    assert checkpoint['identity']==identity(config,datasets,resolver) and checkpoint['runner']['cursor']==641
    model=build_model(config).to(args.device);model.load_state_dict(checkpoint['model']);model.requires_grad_(False)
    decoder_names=('decoder','decoder_position','mask_tokens','topdown','output')
    for name in decoder_names:getattr(model,name).requires_grad_(True)
    model.eval();encoder_sha=fingerprint(model.online.state_dict())
    source_hashes={name:hashlib.sha256((RELEASE/'formal_run'/name).read_bytes()).hexdigest() for name in ('best.pt','last.pt','updates.jsonl')}
    keys=sorted({(r['instrument'],b['band_key']) for r in datasets['TRAIN'].rows for b in r['bands']});lookup={key:i for i,key in enumerate(keys)}
    gamma=nn.Embedding(len(keys),64).to(args.device);beta=nn.Embedding(len(keys),64).to(args.device)
    nn.init.zeros_(gamma.weight);nn.init.zeros_(beta.weight)
    cohort=json.loads((ROOT/'outputs/encoder_v3/hierarchical/structure_diagnosis/cohort.json').read_text())
    cache=[]
    for case in cohort['members']:
        idx=next(i for i,r in enumerate(datasets['TRAIN'].rows) if r['id']==case['id'])
        row=datasets['TRAIN'].rows[idx];sample=sample_from_dataset(datasets['TRAIN'],idx,resolver,args.device)
        layout=model.layout(sample);keeps,targets,_=model.masks(sample,layout,'A',17,0)
        with torch.no_grad():encoded,valids,conditions=model.online(sample,layout,keeps)
        query=[]
        for level,part in enumerate(layout[1]):
            supported=torch.zeros(len(part),device=args.device,dtype=torch.bool)
            for c,mapping in zip(sample.channels,layout[2]):supported|=mapping[level].support(c.valid)>0
            query.append(supported)
        ids=torch.tensor([lookup[(row['instrument'],c.name)] for c in sample.channels],device=args.device)
        cache.append((sample,layout,targets,encoded,valids,conditions,query,ids))
    def forward(item,film):
        sample,layout,targets,encoded,valids,conditions,query,ids=item
        decoded=model.decode(encoded,valids,layout,query);losses=[];predictions=[]
        for i,(c,mask) in enumerate(zip(sample.channels,targets)):
            if not bool(mask.any()):predictions.append(None);continue
            features=decoded[0]
            if film:features=features*(1+gamma(ids[i]))+beta(ids[i])
            field=model.readout(features,conditions['output'][i]);pred=layout[2][i][0].native(field).reshape(c.values.shape)
            losses.append((pred[mask]-c.values[mask]).square().mean());predictions.append(pred)
        return torch.stack(losses).mean(),predictions
    report={'status':'RUNNING','arm':args.arm,'source_run':'release_a_only_v1','source_encoder_has_physical_descriptor':True,
        'source_checkpoint_cursor':641,'source_checkpoint_sha256':source_hashes['best.pt'],'encoder_sha_before':encoder_sha,
        'formal_updates':0,'engineering_updates':0,'cohort':cohort['members'],'initial_losses':[],
        'steps':args.steps,'lr':1e-3,'weight_decay':0.,'clip_norm':3.,'mask_seed':17,'mask_epoch':0,
        'new_decoder_parameter_count':sum(p.numel() for p in (gamma.weight,beta.weight)) if args.arm=='band_film' else 0,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def flush():write(args.output/'report.json',report)
    with torch.no_grad():
        for item in cache:
            ordinary,_=forward(item,False);modulated,_=forward(item,True)
            expected=model([item[0]],kind='A',seed=17,epoch=0)[0][0]
            assert abs(float(ordinary)-float(expected))<1e-6 and abs(float(ordinary)-float(modulated))<1e-6
            report['initial_losses'].append(float(ordinary))
    if args.analyze:
        if args.engineering_checkpoint:
            fitted=torch.load(args.engineering_checkpoint,map_location=args.device,weights_only=False)
            assert fitted['source_encoder_sha256']==encoder_sha and fitted['arm']==args.arm and fitted['keys']==keys
            for name in decoder_names:getattr(model,name).load_state_dict(fitted['decoder'][name])
            gamma.load_state_dict(fitted['gamma']);beta.load_state_dict(fitted['beta'])
        rows={i:[] for i in range(len(keys))};measurements=[]
        with torch.no_grad():
            for sample,layout,targets,encoded,valids,conditions,query,ids in cache:
                decoded=model.decode(encoded,valids,layout,query)[0];f=decoded.cpu().numpy().astype(np.float64)
                per_case=[];native={}
                for i,(c,mask) in enumerate(zip(sample.channels,targets)):
                    if not bool(mask.any()):continue
                    m=layout[2][i][0];idx=int(ids[i]);selected=mask.flatten().cpu().numpy()
                    if c.grid not in native:
                        matrix=coo_matrix(((m.area/m.pixel_area).cpu().numpy().astype(np.float64),
                            (m.source.cpu().numpy(),m.destination.cpu().numpy())),shape=(m.pixels,len(f))).tocsr()
                        native[c.grid]=np.column_stack((matrix@f,np.ones(m.pixels)))
                    features=native[c.grid][selected];truth=c.values[mask].cpu().numpy().astype(np.float64)
                    h,w=c.values.shape;yy,xx=np.meshgrid(np.arange(h),np.arange(w),indexing='ij');a,_,ox,_,e,oy=c.grid.affine
                    points=np.stack(((ox+(xx.ravel()+.5)*a)-layout[0].origin[0],(oy+(yy.ravel()+.5)*e)-layout[0].origin[1]),-1)
                    patch=locate(points,layout[1][2].x,layout[1][2].y)[selected]
                    regions=np.zeros(len(layout[1][2]),dtype=bool)
                    chosen=np.random.default_rng(1729).permutation(len(regions))[:min(len(regions)-1,max(1,round(len(regions)*.2)))]
                    regions[chosen]=True;holdout=regions[patch]
                    assert holdout.any() and (~holdout).any()
                    errors={}
                    for variant in ('trained','gamma_only','beta_only','spatial_mean'):
                        values=decoded.mean(0,keepdim=True).expand_as(decoded) if variant=='spatial_mean' else decoded
                        if args.arm=='band_film':
                            if variant!='beta_only':values=values*(1+gamma(ids[i]))
                            if variant!='gamma_only':values=values+beta(ids[i])
                        field=model.readout(values,conditions['output'][i]);prediction=layout[2][i][0].native(field).reshape(c.values.shape)[mask].cpu().numpy()
                        errors[variant+'_mse']=float(np.mean((prediction-truth)**2))
                        errors[variant+'_holdout_mse']=float(np.mean((prediction[holdout]-truth[holdout])**2))
                    row={'sample':sample.identifier,'key':keys[idx],'x':features,'y':truth,'holdout':holdout,'errors':errors}
                    rows[idx].append(row);per_case.append(errors)
                measurements.append({'id':sample.identifier,**{k:float(np.mean([r[k] for r in per_case])) for k in per_case[0]}})
        probes=[]
        for idx,parts in rows.items():
            if not parts:continue
            xs=[];ys=[]
            for row in parts:
                take=~row['holdout'];weight=1/np.sqrt(int(take.sum()))
                xs.append(row['x'][take]*weight);ys.append(row['y'][take]*weight)
            coefficient=np.linalg.lstsq(np.concatenate(xs),np.concatenate(ys),rcond=1e-6)[0]
            for row in parts:
                prediction=row['x']@coefficient
                probes.append({'id':row['sample'],'key':keys[idx],
                    'trained_holdout_mse':row['errors']['trained_holdout_mse'],
                    'linear_probe_fit_mse':float(np.mean((prediction[~row['holdout']]-row['y'][~row['holdout']])**2)),
                    'linear_probe_holdout_mse':float(np.mean((prediction[row['holdout']]-row['y'][row['holdout']])**2))})
        report.update(status='PASS',analysis_only=True,engineering_updates=0,ablations=measurements,band_linear_probes=probes,
            encoder_sha_after=fingerprint(model.online.state_dict()),limits='No updates. Per-band affine probe fits only80% hidden geographic patches of the same two TRAIN observations, evaluates held-out20% patches. This is a within-TRAIN diagnostic, not independent VAL/generalization. Gamma/beta inference ablations do not isolate their training causal contribution.')
        assert report['encoder_sha_after']==encoder_sha
        for name,value in source_hashes.items():assert hashlib.sha256((RELEASE/'formal_run'/name).read_bytes()).hexdigest()==value
        report['source_artifacts_unchanged']=True;flush();print('DECODER_ANALYSIS_PASS',args.arm,measurements,flush=True);return
    parameters=[p for p in model.parameters() if p.requires_grad]
    if args.arm=='band_film':parameters += list(gamma.parameters())+list(beta.parameters())
    opt=torch.optim.AdamW(parameters,lr=1e-3,weight_decay=0.);history=[];flush()
    try:
        for step in range(args.steps):
            opt.zero_grad(set_to_none=True);value=0.;start=time.perf_counter()
            for item in cache:
                loss,_=forward(item,args.arm=='band_film');(loss/len(cache)).backward();value+=float(loss.detach())/len(cache)
            assert all(p.grad is None for p in model.online.parameters())
            norm=torch.nn.utils.clip_grad_norm_(parameters,3.,error_if_nonfinite=True);opt.step();report['engineering_updates']+=1
            history.append({'step':step+1,'loss':value,'gradient_norm':float(norm),'seconds':time.perf_counter()-start})
            if (step+1)%32==0:report['history']=history;flush();print('DECODER_FIT',args.arm,step+1,value,flush=True)
        with torch.no_grad():report['final_losses']=[float(forward(item,args.arm=='band_film')[0]) for item in cache]
        report['encoder_sha_after']=fingerprint(model.online.state_dict());assert report['encoder_sha_after']==encoder_sha
        for name,value in source_hashes.items():assert hashlib.sha256((RELEASE/'formal_run'/name).read_bytes()).hexdigest()==value
        assert json.loads((RELEASE/'formal_run/status.json').read_text())['status']=='STOPPED'
        report.update(status='PASS',encoder_unchanged=True,source_artifacts_unchanged=True,
            limits='Two fixed TRAIN observations, cached frozen physical-Encoder representations. Same original Decoder weights, masks, optimizer and zero-initialized FiLM preserve identical initial predictions. Not formal VAL, not evidence of generalization or whole-model speed.')
        torch.save({'decoder':{name:getattr(model,name).state_dict() for name in decoder_names},'gamma':gamma.state_dict(),'beta':beta.state_dict(),
            'keys':keys,'source_encoder_sha256':encoder_sha,'arm':args.arm},args.output/'diagnostic_decoder.pt')
    except BaseException as exc:report.update(status='FAILED',error=repr(exc));raise
    finally:flush()
    print('FROZEN_ENCODER_DECODER_PASS',args.arm,report['final_losses'],flush=True)


if __name__=='__main__':main()
