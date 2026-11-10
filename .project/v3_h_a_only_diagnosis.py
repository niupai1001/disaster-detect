"""CPU-only reproduction of the fixed step-525 A-only baseline diagnosis."""
import ssl
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

root=Path(__file__).resolve().parents[1]/'earth-representation'
release=(root/'outputs/encoder_v3/hierarchical/release_a_only_v1').resolve()
sys.path.insert(0,str(release/'runtime/src'))
from encoder_v3.hierarchical.train import load_data,build_model,sample_from_dataset,identity
from encoder_v3.hierarchical.geometry import partitions,area_map


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output.resolve()
    torch.set_num_threads(2)
    config=json.loads((release/'formal_run/config.json').read_text())
    datasets,resolver=load_data();model=build_model(config).cpu().eval()
    visual=root/'outputs/encoder_v3/hierarchical/a_only/reconstruction_node_525_v2/report.json'
    vis=json.loads(visual.read_text());assert identity(config,datasets,resolver)==vis['identity']
    out.mkdir(parents=True,exist_ok=False)
    records=json.loads((release/'formal_run/validation_525.json').read_text())['members']
    official={r['id']:r for r in records}
    selected={r['id']:r for r in vis['cases'] if r['role']=='VAL' and r['selection']!='S2最高综合误差观测'}
    assert len(selected)==6
    members=[];cases=[]
    for index,row in enumerate(datasets['VAL'].rows):
        sample=sample_from_dataset(datasets['VAL'],index,resolver,'cpu')
        geo,parts,_=partitions(tuple(c.grid for c in sample.channels),config['patch_metres'])
        coarse={c.grid:area_map(c.grid,geo,parts[2],'cpu') for c in sample.channels}
        maps=[(None,None,coarse[c.grid]) for c in sample.channels]
        keeps,targets,_=model.masks(sample,(geo,parts,maps,[],[]),'A',config['validation_seed'],0)
        channels=[]
        for c,mapping,keep,mask in zip(sample.channels,maps,keeps,targets):
            if not bool(mask.any()):continue
            assert bool(keep.any()),'Baseline requires visible support in every target channel'
            truth=c.values.numpy().ravel()[mask.numpy().ravel()].astype(np.float64)
            values={'channel':c.name,'zero_mse':float(np.mean(truth**2)),
                    'visible_mean_mse':float(np.mean((truth-float(c.values[keep].mean()))**2))}
            if row['id'] in selected:
                for name,level in [('fine_target_fit',0),('patch_target_fit',2)]:
                    m=area_map(c.grid,geo,parts[0],'cpu') if level==0 else mapping[2]
                    matrix=coo_matrix(((m.area/m.pixel_area).numpy().astype(np.float64),
                        (m.source.numpy(),m.destination.numpy())),shape=(m.pixels,len(parts[level]))).tocsr()[mask.numpy().ravel()]
                    fit=lsqr(matrix,truth,atol=1e-10,btol=1e-10,iter_lim=10000)
                    values[name]=float(np.mean((truth-matrix@fit[0])**2))
                    values[name+'_stop']=int(fit[1]);values[name+'_iterations']=int(fit[2])
                assert values['fine_target_fit']<=values['patch_target_fit']+1e-6
            channels.append(values)
        entry={'id':row['id'],'sensor':row['sensor'],'model_mse':official[row['id']]['loss'],
            'visible_mean_mse':float(np.mean([c['visible_mean_mse'] for c in channels])),
            'train_mean_mse':float(np.mean([c['zero_mse'] for c in channels]))}
        members.append(entry)
        if row['id'] in selected:
            case=selected[row['id']]
            assert abs(entry['model_mse']-case['loss'])<1e-8
            assert abs(entry['train_mean_mse']-case['same_target_train_mean_baseline_mse'])<1e-5
            cases.append(dict(entry,source=row['sensor'],selection=case['selection'],channels=channels,
                **{name+'_mse':float(np.mean([c[name] for c in channels])) for name in ('fine_target_fit','patch_target_fit')}))
        if (index+1)%100==0:print('BASELINE',index+1,flush=True)
    summary={'status':'PASS','checkpoint_cursor':525,'observations':len(members),'members':members,
        **{name:float(np.mean([r[name] for r in members])) for name in ('model_mse','visible_mean_mse','train_mean_mse')},
        'model_worse_than_visible_mean_count':sum(r['model_mse']>r['visible_mean_mse'] for r in members),
        'method':'Same official fixed A mask/QA; predict hidden pixels using visible valid per-channel mean. Pixel, channel, observation means; no target leakage; CPU only.'}
    assert len(members)==383 and abs(summary['model_mse']-vis['full_val_loss'])<1e-8
    for sensor in ('S2','L8'):
        rows=[r for r in members if r['sensor']==sensor]
        summary[sensor]={'observations':len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in ('model_mse','visible_mean_mse')},
            'worse_count':sum(r['model_mse']>r['visible_mean_mse'] for r in rows)}
    base={r['id']:r for r in members};nodes=[]
    for p in sorted((release/'formal_run').glob('validation_*.json'),key=lambda p:int(p.stem.split('_')[-1])):
        val=json.loads(p.read_text());assert {r['id'] for r in val['members']}==set(base)
        nodes.append({'cursor':int(p.stem.split('_')[-1]),'model_mse':val['metric'],'visible_mean_mse':summary['visible_mean_mse'],
            'worse_count':sum(r['loss']>base[r['id']]['visible_mean_mse'] for r in val['members'])})
    summary['same_fixed_mask_validation_nodes']=nodes
    report={'status':'PASS','checkpoint_cursor':525,'gpu_used':False,'optimizer_updates':0,'cases':cases,
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_visual_report_sha256':hashlib.sha256(visual.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'method':'Target-informed per-channel LSQR through exact Fine/Coarse area readout on hidden rows. Attainable fit uses hidden truth: not a usable predictor, not proof the shared neural head can fit it, not a certified minimum. LSQR atol/btol=1e-10, iter_lim=10000; stop codes retained.'}
    for name,value in [('report.json',report),('full_val_visible_mean.json',summary)]:
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DIAGNOSIS_PASS',json.dumps({k:v for k,v in summary.items() if k not in ('members','same_fixed_mask_validation_nodes')}),flush=True)


if __name__=='__main__':main()
