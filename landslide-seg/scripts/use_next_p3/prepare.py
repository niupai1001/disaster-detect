import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import torch
import yaml
from landslide_seg.models.use_next_p2.encoder import ScaleEncoder


def main():
    output=ROOT/'outputs/use_next_v2/p3_engineering';output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(1234);model=ScaleEncoder('physical')
    torch.save(dict(local=model.state_dict(),seed=1234,updates=0),output/'u0.pt')
    source=torch.load(ROOT/'outputs/use_next_v2/p2_engineering/u0.pt',weights_only=True)
    assert all(torch.equal(v,source['local'][k]) for k,v in model.state_dict().items())
    base=yaml.safe_load((ROOT/'configs/use_next_p2/physical_pretrain.yaml').read_text());v=base.pop('use_next_p2')
    for objective in ('scale','latent','joint'):
        for task,stage,steps in (('pretrain','P3',10000),('head','P5',3000)):
            cfg=json.loads(json.dumps(base));values=json.loads(json.dumps(v))
            cfg['segformer']['model']=dict(type='use_next_p3',in_chans=10)
            cfg['segformer']['train'].update(max_steps=steps,schedule_updates=steps)
            cfg['segformer']['optim'].update(encoder_lr=.0003 if task=='pretrain' else 0.,head_lr=.0006)
            values.update(stage=stage,task=task,steps=steps,warmup_updates=steps//20,objective=objective,source_objective=objective,
                u0='outputs/use_next_v2/p3_engineering/u0.pt',selection='fixed_u10000' if task=='pretrain' else 'dev_event_mean_iou',
                frozen_source=f'outputs/use_next_v2/runs/p3_{objective}/last.pt',
                loss=dict(scale='same TRAIN-calibrated120m low-frequency MSE',latent='per-feature LayerNorm MSE of detached EMA120m patch target',
                          joint='scale+latent, fixed unit weights; absent scale on SAR does not rescale latent'),
                ema=dict(start=.996,end=1.,schedule='cosine',boundary='new local encoder only; anchors excluded'))
            values['milestones']=[1000,3000,5000,7500,10000] if task=='pretrain' else [3000]
            values['band_drop']=dict(probability=.5,minimum_fraction=.2,maximum_fraction=.4,integer_policy='clamp_round_to_ceil_floor')
            cfg['use_next_p3']=values
            (ROOT/f'configs/use_next_p3/{objective}_{task}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))


if __name__=='__main__':main()
