"""Isolated structural interventions and fixed-TRAIN controls; never resumes a run."""
import ssl,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING','0')
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
ROOT=Path(__file__).resolve().parents[1]/'earth-representation'
RELEASE=(ROOT/'outputs/encoder_v3/hierarchical/release_descriptor_free_v1').resolve()
sys.path.insert(0,str(RELEASE/'runtime/src'))
from encoder_v3.hierarchical.train import load_data,build_model,sample_from_dataset,identity,write
from encoder_v3.hierarchical.model import masked_softmax
from encoder_v3.latent_response.checkpoint import parameter_sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['intervention','fit'],required=True)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arm',choices=['original','value_film','fully_visible'],default='original')
    parser.add_argument('--steps',type=int,default=128)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(17);torch.cuda.set_device(args.device)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=json.loads((RELEASE/'formal_run/config.json').read_text());sets,resolver=load_data()
    payload=torch.load(RELEASE/'formal_run/best.pt',map_location='cpu',weights_only=False)
    assert identity(config,sets,resolver)==payload['identity']
    model=build_model(config).to(args.device);model.load_state_dict(payload['model']);model.eval()
    baseline_sha=parameter_sha(model)
    hashes={name:hashlib.sha256((RELEASE/'formal_run'/name).read_bytes()).hexdigest() for name in ('best.pt','last.pt','updates.jsonl')}
    report={'status':'RUNNING','mode':args.mode,'arm':args.arm,'checkpoint_cursor':233,'checkpoint_sha256':hashes['best.pt'],
        'identity':payload['identity'],'formal_updates':0,'engineering_updates':0,'cases':[],
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def flush():write(args.output/'report.json',report)
    def evaluate(sample,seed):
        with torch.no_grad():loss,details=model([sample],kind='A',seed=seed,epoch=0,return_predictions=True)
        return float(loss[0]),details[0]
    def prediction_delta(detail,reference):
        return float(torch.stack([(a[mask]-b[mask]).square().mean() for a,b,mask in zip(detail['predictions'],reference['predictions'],reference['target_masks']) if bool(mask.any())]).mean().sqrt())
    def visible_baseline(sample,detail):
        values=[]
        for c,mask in zip(sample.channels,detail['target_masks']):
            keep=c.valid&~mask
            if bool(mask.any()):
                assert bool(keep.any());values.append(float((c.values[mask]-c.values[keep].mean()).square().mean()))
        return float(np.mean(values))
    flush()
    try:
        if args.mode=='intervention':
            atlas=json.loads((ROOT/'outputs/encoder_v3/hierarchical/descriptor_free/reconstruction_best_v1/report.json').read_text())
            cases=[c for c in atlas['cases'] if c['role']=='TRAIN' or 'P50' in c['selection']]
            # Add the same two TRAIN optical controls used by the fixed-cohort engineering gate.
            learning=json.loads((ROOT/'outputs/encoder_v3/hierarchical/descriptor_free/learning_v2/report.json').read_text())
            cases += [dict(c,role='TRAIN',selection='fixed TRAIN cohort') for c in learning['after'] if c['source'] in ('S2','L8')]
            for case in cases:
                ds=sets[case['role']];idx=next(i for i,r in enumerate(ds.rows) if r['id']==case['id'])
                sample=sample_from_dataset(ds,idx,resolver,args.device);seed=1729 if case['role']=='VAL' else 17
                captured=[];handle=model.decoder[0].register_forward_hook(lambda mod,inputs,out:captured.append(out.detach()))
                score_chunks=[];score_handle=model.online.fusion_key[0].register_forward_hook(lambda mod,inputs,out:score_chunks.append(out.detach().squeeze(-1)))
                loss,reference=evaluate(sample,seed);handle.remove();score_handle.remove();features=captured[0]
                layout=model.layout(sample);keeps,_,_=model.masks(sample,layout,'A',seed,0)
                with torch.no_grad():encoded=model.online(sample,layout,keeps)
                present=encoded.tokens[0].support_ratio>0;scores=torch.cat(score_chunks,1);alpha=masked_softmax(scores,present,0);active=present.any(0)
                entropy=-(alpha*alpha.clamp_min(1e-30).log()).sum(0)
                entry={'id':sample.identifier,'source':case['source'],'role':case['role'],'A_mse':loss,
                    'visible_mean_mse':visible_baseline(sample,reference),'fusion_effective_band_count_mean':float(entropy[active].exp().mean()),
                    'fusion_largest_weight_mean':float(alpha[:,active].max(0).values.mean()),'active_input_bands':len(scores),'interventions':[]}
                for intervention in ('zero_visible','shuffle_visible','mean_visible','plus_one_visible','reverse_fusion_identity','reverse_output_identity','zero_decoder_fine','zero_decoder_middle','zero_decoder_coarse','zero_all_decoder_inputs'):
                    altered=copy.copy(sample);altered.channels=[copy.copy(c) for c in sample.channels]
                    old_decode=model.decode;old_conditions=model.online.conditions
                    if intervention.endswith('_visible'):
                        for i,(dst,src,keep) in enumerate(zip(altered.channels,sample.channels,keeps)):
                            dst.values=src.values.clone();values=src.values[keep]
                            if intervention=='zero_visible':dst.values[keep]=0
                            elif intervention=='mean_visible' and len(values):dst.values[keep]=values.mean()
                            elif intervention=='plus_one_visible':dst.values[keep]+=1
                            elif intervention=='shuffle_visible':
                                permutation=torch.randperm(len(values),generator=torch.Generator().manual_seed(1729+i)).to(values.device)
                                dst.values[keep]=values[permutation]
                    elif intervention.startswith('reverse_'):
                        key='fusion' if intervention=='reverse_fusion_identity' else 'output'
                        def changed_conditions(s,key=key):
                            values=old_conditions(s);return dict(values,**{key:values[key].flip(0)})
                        model.online.conditions=changed_conditions
                    else:
                        ports={'zero_decoder_fine':[0],'zero_decoder_middle':[1],'zero_decoder_coarse':[2],'zero_all_decoder_inputs':[0,1,2]}[intervention]
                        model.decode=lambda e,v,l,q,ports=ports:old_decode([torch.zeros_like(x) if i in ports else x for i,x in enumerate(e)],v,l,q)
                    try:
                        changed_loss,detail=evaluate(altered,seed)
                        entry['interventions'].append({'name':intervention,'A_mse':changed_loss,'prediction_change_rms_z':prediction_delta(detail,reference)})
                    finally:model.decode=old_decode;model.online.conditions=old_conditions
                if case['role']=='VAL':
                    fits=[];native_features={};f=features.cpu().numpy().astype(np.float64)
                    for c,mapping,mask in zip(sample.channels,layout[2],reference['target_masks']):
                        if not bool(mask.any()):continue
                        m=mapping[0];matrix=coo_matrix(((m.area/m.pixel_area).cpu().numpy().astype(np.float64),(m.source.cpu().numpy(),m.destination.cpu().numpy())),shape=(m.pixels,len(f))).tocsr()
                        target=c.values[mask].cpu().numpy().astype(np.float64);selected=mask.flatten().cpu().numpy();a=matrix[selected]
                        if c.grid not in native_features:native_features[c.grid]=np.column_stack((matrix@f,np.ones(m.pixels)))
                        x=native_features[c.grid][selected];coeff=np.linalg.lstsq(x,target,rcond=1e-8)[0]
                        oracle=lsqr(a,target,atol=1e-9,btol=1e-9,iter_lim=2000)
                        fits.append({'band':c.name,'target_mean_fit_mse':float(np.mean((target-target.mean())**2)),
                            'decoder_band_linear_fit_mse':float(np.mean((target-x@coeff)**2)),
                            'free_fine_target_fit_mse':float(np.mean((target-a@oracle[0])**2)),'lsqr_stop':int(oracle[1])})
                    entry['target_informed_fit_diagnostics']=fits
                report['cases'].append(entry);flush();print('INTERVENTION',case['source'],case['role'],loss,flush=True)
            assert parameter_sha(model)==baseline_sha
        else:
            cohort_path=ROOT/'outputs/encoder_v3/hierarchical/structure_diagnosis/cohort.json'
            cohort=json.loads(cohort_path.read_text());samples=[]
            for row in cohort['members']:
                idx=next(i for i,r in enumerate(sets['TRAIN'].rows) if r['id']==row['id'])
                samples.append(sample_from_dataset(sets['TRAIN'],idx,resolver,args.device))
            initial=[evaluate(s,17)[0] for s in samples]
            if args.arm=='value_film':
                model.online.value_gates=nn.ModuleList(nn.Linear(32,w) for w in (64,96,128)).to(args.device)
                for gate in model.online.value_gates:nn.init.zeros_(gate.weight);nn.init.zeros_(gate.bias)
                def fuse(summary,support,condition,level):
                    h=model.online.projection[level](torch.cat((summary,support[...,None]),-1))
                    content=torch.cat((h,condition[:,None].expand(-1,h.shape[1],-1)),-1)
                    weights=masked_softmax(model.online.fusion_key[level](content).squeeze(-1),support>0,0)
                    modulated=h*(1+model.online.value_gates[level](condition)[:,None])
                    values=model.online.fusion_value[level](torch.cat((modulated,condition[:,None].expand(-1,h.shape[1],-1)),-1))
                    return (values*weights[...,None]).sum(0)
                model.online.fuse=fuse
                for expected,s in zip(initial,samples):assert abs(evaluate(s,17)[0]-expected)<1e-6
            elif args.arm=='fully_visible':
                original=model.masks
                def masks(s,l,kind,seed,epoch):
                    _,targets,patch=original(s,l,kind,seed,epoch)
                    return [c.valid for c in s.channels],targets,patch
                model.masks=masks
            report['cohort']=cohort;report['initial_losses']=initial
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3,weight_decay=0.)
            history=[];model.train()
            for step in range(args.steps):
                opt.zero_grad(set_to_none=True);value=0.;start=time.perf_counter()
                for s in samples:
                    losses,_=model([s],kind='A',seed=17,epoch=0);(losses.mean()/len(samples)).backward();value+=float(losses.detach().mean())/len(samples)
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],3.,error_if_nonfinite=True);opt.step()
                history.append({'step':step+1,'loss':value,'gradient_norm':float(norm),'seconds':time.perf_counter()-start});report['engineering_updates']+=1
                if (step+1)%32==0:report['history']=history;flush();print('FIT',args.arm,step+1,value,flush=True)
            model.eval();report['final_losses']=[evaluate(s,17)[0] for s in samples]
            report['limits']='Fixed TRAIN only, seed17 epoch0 mask, lr1e-3/no decay control. value_film changes only fusion value; fully_visible deliberately reveals targets as an engineering upper control. No formal VAL result or generalization claim.'
        for name,value in hashes.items():assert hashlib.sha256((RELEASE/'formal_run'/name).read_bytes()).hexdigest()==value
        assert json.loads((RELEASE/'formal_run/status.json').read_text())['status']=='TRIAL_COMPLETE'
        report.update(status='PASS',source_artifacts_unchanged=True)
    except BaseException as exc:report.update(status='FAILED',error=repr(exc));raise
    finally:flush()


if __name__=='__main__':main()
