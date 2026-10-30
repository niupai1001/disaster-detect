"""TRAIN-only fixed loss calibration and common initialization; no GPU use."""
import json
from pathlib import Path
import sys
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import yaml
from landslide_seg.models.use_next_p2.encoder import ScaleEncoder
from landslide_seg.models.use_next_v2.readout import state_digest


def main():
    output=ROOT/'outputs/use_next_v2/p2_engineering';output.mkdir(parents=True,exist_ok=True)
    release=ROOT/'outputs/use_next_v2/data/data_release_v1'
    contract=json.loads((release/'release.json').read_text())
    rows=[json.loads(l) for l in (release/'manifest.jsonl').read_text().splitlines()]
    groups=defaultdict(list)
    for row in rows:
        if row['role']=='TRAIN' and row['sensor'] in ('s2','landsat'):groups[row['sensor']+'/'+','.join(row['band_ids'])].append(row)
    rng=np.random.default_rng(1234);calibration={}
    for key,records in sorted(groups.items()):
        records=sorted(records,key=lambda r:r['id']);ids=rng.choice(len(records),size=min(32,len(records)),replace=False)
        count=total=squared=None
        for index in ids:
            with np.load(records[index]['cache_uri']) as arr:
                x=arr['x'].astype('float64');valid=arr['pixel_valid'];x=np.where(valid,x,0)
                sums=x.sum((1,2));sq=(x*x).sum((1,2));n=valid.sum((1,2))
            if count is None:count=n;total=sums;squared=sq
            else:count+=n;total+=sums;squared+=sq
        mean=total/count;std=np.sqrt(np.maximum(squared/count-mean*mean,1e-8))
        calibration[key]=dict(mean=mean.tolist(),std=std.tolist(),fit_role='TRAIN',fit_ids=[records[i]['id'] for i in ids],count=count.tolist())
    (output/'calibration.json').write_text(json.dumps(calibration,indent=2))
    torch.manual_seed(1234);pixel=ScaleEncoder('pixel');torch.manual_seed(1234);physical=ScaleEncoder('physical')
    assert state_digest(pixel)==state_digest(physical)
    torch.save(dict(local=pixel.state_dict(),seed=1234,updates=0),output/'u0.pt')
    (output/'initialization.json').write_text(json.dumps(dict(pixel=state_digest(pixel),physical=state_digest(physical),same=True)))
    base=yaml.safe_load((ROOT/'configs/use_next_v2/p1_dofa.yaml').read_text());base.pop('use_next_v2')
    for mode in ('pixel','physical'):
        for task,steps in (('pretrain',6000),('head',3000)):
            cfg=json.loads(json.dumps(base));cfg['segformer']['model']=dict(type='use_next_p2',in_chans=10)
            cfg['segformer']['train'].update(max_steps=steps,schedule_updates=steps,batch_size=8,eval_every_updates=500)
            cfg['segformer']['optim'].update(encoder_lr=.0003 if task=='pretrain' else 0.,head_lr=.0006)
            cfg['use_next_p2']=dict(stage='P2',mode=mode,task=task,steps=steps,seed=1234,head_seed=1235,
                width=192,layers=2,heads=6,physical_patch_m=120,subcell_m=30,pixel_block=4,microbatch=2,effective_batch=8,
                data_release='outputs/use_next_v2/data/data_release_v1',data_manifest_sha256=contract['manifest_sha256'],
                calibration='outputs/use_next_v2/p2_engineering/calibration.json',u0='outputs/use_next_v2/p2_engineering/u0.pt',
                schedule='outputs/use_next_v2/data/schedule_seed1234.jsonl',selection='fixed_u6000' if task=='pretrain' else 'dev_event_mean_iou',
                frozen_source=f'outputs/use_next_v2/runs/p2_{mode}/last.pt',precision='FP32',tf32=False,
                loss='same TRAIN-calibrated physical120m low-frequency MSE; no unsupported high-frequency target',
                sar_target='NOT_AVAILABLE_NO_VERIFIED_NOISE_CALIBRATION',schedule_type='cosine',warmup_updates=round(steps*.05))
            cfg['use_next_p2']['milestones']=[1000,3000,6000] if task=='pretrain' else [3000]
            (ROOT/f'configs/use_next_p2/{mode}_{task}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    print('calibration groups',len(calibration))


if __name__=='__main__':main()
